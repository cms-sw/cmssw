// -*- C++ -*-
//
// Package:    FastSimulation/CTPPSSimHitProducer
// Class:      CTPPSSimHitProducer
//
/**\class CTPPSSimHitProducer CTPPSSimHitProducer.cc FastSimulation/CTPPSSimHitProducer/plugins/CTPPSSimHitProducer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Dilson De Jesus Damiao
//         Created:  Mon, 05 Sep 2016 18:49:10 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// SimHitContainer
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// STL headers
#include <vector>
#include <iostream>

// HepMC headers
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Forward/interface/LHCTransportLink.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "Utilities/PPS/interface/PPSUnitConversion.h"

//
// class declaration
//

class CTPPSSimHitProducer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSSimHitProducer(const edm::ParameterSet&);
  ~CTPPSSimHitProducer() override;

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  edm::EDGetTokenT<edm::HepMCProduct> mcEventToken;  // label of MC event
  edm::Handle<edm::HepMCProduct> EvtHandle;
  // ----------member data ---------------------------
  double fz_tracker1, fz_tracker2, fz_timing;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CTPPSSimHitProducer::CTPPSSimHitProducer(const edm::ParameterSet& iConfig) {
  produces<edm::PSimHitContainer>("CTPPSHits");
  // consumes
  mcEventToken =
      mayConsume<edm::HepMCProduct>(iConfig.getUntrackedParameter<edm::InputTag>("MCEvent", std::string("")));

  // Read the position of the trackers and timing detectors
  fz_tracker1 = iConfig.getParameter<double>("Z_Tracker1");
  fz_tracker2 = iConfig.getParameter<double>("Z_Tracker2");
  fz_timing = iConfig.getParameter<double>("Z_Timing");
}

CTPPSSimHitProducer::~CTPPSSimHitProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CTPPSSimHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::vector<PSimHit> theCTPPSHits;
  iEvent.getByToken(mcEventToken, EvtHandle);

  const HepMC::GenEvent* Evt = EvtHandle->GetEvent();
  std::vector<math::XYZTLorentzVector> protonCTPPS;
  protonCTPPS.clear();
  for (HepMC::GenEvent::vertex_const_iterator ivtx = Evt->vertices_begin(); ivtx != Evt->vertices_end(); ivtx++) {
    if ((*ivtx)->id() != 0)
      continue;
    double prim_vtxZ = (*ivtx)->position().z() * mm_to_m;  //in meters
    // Get the vertices at the entrance of CTPPS and get the protons coming out of them (propagated by Hector)
    for (HepMC::GenVertex::particles_out_const_iterator i = (*ivtx)->particles_out_const_begin();
         i != (*ivtx)->particles_out_const_end();
         ++i) {
      int pid = (*i)->pdg_id();
      if (pid != 2212)
        continue;

      HepMC::GenVertex* pv = (*i)->production_vertex();
      const HepMC::FourVector& vertex = pv->position();
      const HepMC::FourVector p((*i)->momentum());
      protonCTPPS.push_back(math::XYZTLorentzVector(p.x(), p.y(), p.z(), p.t()));

      LocalPoint initialPosition_tr1, initialPosition_tr2, initialPosition_tof;

      double t0_tr1 = 0., t0_tr2 = 0., t0_tof = 0.;
      //  t0_* has dimensions of mm
      //  Convert to ns for internal calculations.
      const double c_light_s = 2.99792458e+11;  // mm/s;
      const double s_to_ns = 1.e9;
      const double m_to_mm = 1.e3;
      double x_tr1 = 0., x_tr2 = 0., x_tof = 0., y_tr1 = 0., y_tr2 = 0., y_tof = 0., z_tr1 = 0.;
      double z_tr2 = fz_tracker2;  //m
      double z_tof = fz_timing;    //m
      int Direction = 0;

      // Read the vertex made by SimTransport at the entrance of det1
      if (std::abs(vertex.eta()) > 8. && (*i)->status() == 1) {
        if (vertex.z() > 0)
          Direction = 1;
        else if (vertex.z() < 0)
          Direction = -1;

        //Get the global coordinates at Tracker1, equal to those of the vertex at CTPPS
        x_tr1 = vertex.x();
        y_tr1 = vertex.y();
        z_tr1 = vertex.z() * mm_to_m;  //move z from mm to meters
        Local3DPoint xyzzy_tr1(x_tr1, y_tr1, z_tr1);
        initialPosition_tr1 = xyzzy_tr1;
        t0_tr1 = vertex.t() / c_light_s * s_to_ns;
        //Get the global coordinates at Tracker2 by propagating as a straight line from Tracker1
        t0_tr2 = z_tr2 * m_to_mm / c_light_s * s_to_ns;  //discuss latter if needs to be corrected with vertex position
        t0_tr2 = t0_tr1 + (z_tr2 - z_tr1) * m_to_mm / c_light_s * s_to_ns;  //corrected with vertex position
        z_tr2 *= Direction;
        x_tr2 = x_tr1 + (p.x() / p.z()) * (z_tr2 - z_tr1) * m_to_mm;
        y_tr2 = y_tr1 + (p.y() / p.z()) * (z_tr2 - z_tr1) * m_to_mm;
        Local3DPoint xyzzy_tr2(x_tr2, y_tr2, z_tr2);
        initialPosition_tr2 = xyzzy_tr2;
        //Propagate as a straight line from Tracker1
        t0_tof = z_tof * m_to_mm / c_light_s * s_to_ns;  //discuss latter if needs to be corrected with vertex position
        t0_tof = (z_tof - prim_vtxZ) * m_to_mm / c_light_s * s_to_ns;  //corrected with vertex position
        z_tof *= Direction;
        x_tof = x_tr1 + (p.x() / p.z()) * (z_tof - z_tr1) * m_to_mm;
        y_tof = y_tr1 + (p.y() / p.z()) * (z_tof - z_tr1) * m_to_mm;
        Local3DPoint xyzzy_tof(x_tof, y_tof, z_tof);
        initialPosition_tof = xyzzy_tof;

        // DetId codification for PSimHit from CTPPSPixel- It will be replaced by CTPPSDetId
        // 2014314496 -> Tracker1 zPositive
        // 2014838784 -> Tracker2 zPositive
        // 2046820352 -> Timing   zPositive
        // 2031091712 -> Tracker1 zNegative
        // 2031616000 -> Tracker2 zNegative
        // 2063597568 -> Timing   zNegative

        if (Direction > 0.) {
          PSimHit hit_tr1(xyzzy_tr1, xyzzy_tr1, 0., t0_tr1, 0., pid, 2014314496, 0, 0., 0., 2);
          PSimHit hit_tr2(xyzzy_tr2, xyzzy_tr2, 0., t0_tr2, 0., pid, 2014838784, 0, 0., 0., 2);
          PSimHit hit_tof(xyzzy_tof, xyzzy_tof, 0., t0_tof, 0., pid, 2046820352, 0, 0., 0., 2);
          theCTPPSHits.push_back(hit_tr1);
          theCTPPSHits.push_back(hit_tr2);
          theCTPPSHits.push_back(hit_tof);
        }
        if (Direction < 0.) {
          PSimHit hit_tr1(xyzzy_tr1, xyzzy_tr1, 0., t0_tr1, 0., pid, 2031091712, 0, 0., 0., 2);
          PSimHit hit_tr2(xyzzy_tr2, xyzzy_tr2, 0., t0_tr2, 0., pid, 2031616000, 0, 0., 0., 2);
          PSimHit hit_tof(xyzzy_tof, xyzzy_tof, 0., t0_tof, 0., pid, 2063597568, 0, 0., 0., 2);
          theCTPPSHits.push_back(hit_tr1);
          theCTPPSHits.push_back(hit_tr2);
          theCTPPSHits.push_back(hit_tof);
        }
      }
    }
  }
  std::unique_ptr<edm::PSimHitContainer> pctpps(new edm::PSimHitContainer);
  for (std::vector<PSimHit>::const_iterator i = theCTPPSHits.begin(); i != theCTPPSHits.end(); i++) {
    pctpps->push_back(*i);
  }
  iEvent.put(std::move(pctpps), "CTPPSHits");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void CTPPSSimHitProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void CTPPSSimHitProducer::endStream() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CTPPSSimHitProducer);
