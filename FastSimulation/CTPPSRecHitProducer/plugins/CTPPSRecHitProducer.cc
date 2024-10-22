// -*- C++ -*-
//
// Package:    FastSimulation/CTPPSRecHitProducer
// Class:      CTPPSRecHitProducer
//
/**\class CTPPSRecHitProducer CTPPSRecHitProducer.cc FastSimulation/CTPPSRecHitProducer/plugins/CTPPSRecHitProducer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Dilson De Jesus Damiao
//         Created:  Mon, 19 Sep 2016 18:10:30 GMT
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
/////
#include <cmath>

#include <CLHEP/Random/RandGauss.h>
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <vector>
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastRecHitContainer.h"
#include "FastSimulation/CTPPSFastGeometry/interface/CTPPSToFDetector.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include <iostream>

#include <string>

//Mixing Collection and CF info
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "HepMC/GenEvent.h"

//
// class declaration
//
class TRandom3;
class CTPPSToFDetector;

class CTPPSRecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSRecHitProducer(const edm::ParameterSet&);
  ~CTPPSRecHitProducer() override;

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  // ----------member data ---------------------------
  typedef std::vector<PSimHit> PSimHitContainer;

  //	std::string mix_;
  //   	std::string collection_for_XF;

  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token;

  double fTrackerWidth, fTrackerHeight, fTrackerInsertion, fBeamXRMS_Trk1, fBeamXRMS_Trk2, fTrk1XOffset, fTrk2XOffset;
  double fHitSigmaX, fHitSigmaY, fHitSigmaZ;
  std::vector<double> fToFCellWidth;
  double fToFCellHeight, fToFPitchX, fToFPitchY;
  int fToFNCellX, fToFNCellY;
  double fToFInsertion, fBeamXRMS_ToF, fToFXOffset, fTimeSigma;
};

CTPPSRecHitProducer::CTPPSRecHitProducer(const edm::ParameterSet& iConfig) {
  produces<edm::CTPPSFastRecHitContainer>("CTPPSFastRecHits");

  //Name of Collection use for create the XF
  std::string mix_ = iConfig.getParameter<std::string>("mixLabel");
  std::string collection_for_XF = iConfig.getParameter<std::string>("InputCollection");
  cf_token = consumes<CrossingFrame<PSimHit> >(edm::InputTag(mix_, collection_for_XF));

  // Read the detector parameters
  fTrackerWidth = iConfig.getParameter<double>("TrackerWidth");
  fTrackerHeight = iConfig.getParameter<double>("TrackerHeight");
  fTrackerInsertion = iConfig.getParameter<double>("TrackerInsertion");
  fBeamXRMS_Trk1 = iConfig.getParameter<double>("BeamXRMS_Trk1");
  fBeamXRMS_Trk2 = iConfig.getParameter<double>("BeamXRMS_Trk2");
  fTrk1XOffset = iConfig.getParameter<double>("Trk1XOffset");
  fTrk2XOffset = iConfig.getParameter<double>("Trk2XOffset");
  fHitSigmaX = iConfig.getParameter<double>("HitSigmaX");
  fHitSigmaY = iConfig.getParameter<double>("HitSigmaY");
  fHitSigmaZ = iConfig.getParameter<double>("HitSigmaZ");
  fToFCellWidth = iConfig.getUntrackedParameter<std::vector<double> >("ToFCellWidth");
  fToFCellHeight = iConfig.getParameter<double>("ToFCellHeight");
  fToFPitchX = iConfig.getParameter<double>("ToFPitchX");
  fToFPitchY = iConfig.getParameter<double>("ToFPitchY");
  fToFNCellX = iConfig.getParameter<int>("ToFNCellX");
  fToFNCellY = iConfig.getParameter<int>("ToFNCellY");
  fToFInsertion = iConfig.getParameter<double>("ToFInsertion");
  fBeamXRMS_ToF = iConfig.getParameter<double>("BeamXRMS_ToF");
  fToFXOffset = iConfig.getParameter<double>("ToFXOffset");
  fTimeSigma = iConfig.getParameter<double>("TimeSigma");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "CTPPSRecHitProducer requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  }
}

CTPPSRecHitProducer::~CTPPSRecHitProducer() {}

void CTPPSRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());
  if (engine->name() != "TRandom3") {
    throw cms::Exception("Configuration") << "The TRandom3 engine type must be used with CTPPSRecHitProducer, Random "
                                             "Number Generator Service not correctly configured!";
  }

  //Mix the PSimHits for pileup
  Handle<CrossingFrame<PSimHit> > xFrame;
  iEvent.getByToken(cf_token, xFrame);

  std::unique_ptr<MixCollection<PSimHit> > simHits(new MixCollection<PSimHit>(xFrame.product()));

  std::vector<CTPPSFastRecHit> theCTPPSFastRecHit;

  for (MixCollection<PSimHit>::MixItr ihit = simHits->begin(); ihit != simHits->end(); ihit++)

  {
    const PSimHit* simHit = &(*ihit);
    unsigned int detlayerId = simHit->detUnitId();
    // DetId layer codification for PSimHit
    // 2014314496 -> Tracker1 zPositive
    // 2014838784 -> Tracker2 zPositive
    // 2046820352 -> Timing   zPositive
    // 2031091712 -> Tracker1 zNegative
    // 2031616000 -> Tracker2 zNegative
    // 2063597568 -> Timing   zNegative
    bool outside = false;

    //Make Tracker RecHits by smearing the SimHits positions and applying fiducial cuts
    CTPPSFastRecHit rechit;  // (const Local3DPoint& entry, unsigned int detId, float tof, unsigned int cellId)
    if (detlayerId == 2014314496 || detlayerId == 2031091712) {
      // Apply position smearing
      float x_trk1 = simHit->entryPoint().x() + CLHEP::RandGauss::shoot(engine, 0, fHitSigmaX * um_to_mm);
      float y_trk1 = simHit->entryPoint().y() + CLHEP::RandGauss::shoot(engine, 0, fHitSigmaY * um_to_mm);
      float z_trk1 = simHit->entryPoint().z();

      // Apply fiducial cuts
      double pos_trk1 = fTrackerInsertion * fBeamXRMS_Trk1 + fTrk1XOffset;
      if (x_trk1 > 0 || fabs(x_trk1) < pos_trk1 || fabs(x_trk1) > (fTrackerWidth + pos_trk1) ||
          fabs(y_trk1) > fTrackerHeight / 2.)
        outside = true;

      float tof = 0.0;
      unsigned int cellId = 1;
      Local3DPoint xyzzy = Local3DPoint(x_trk1, y_trk1, z_trk1);
      if (!outside) {
        rechit.setLocal3DPoint(xyzzy);
        rechit.setTof(tof);
        rechit.setDetUnitId(detlayerId);
        rechit.setCellId(cellId);
        theCTPPSFastRecHit.push_back(rechit);
      }
    }
    if (detlayerId == 2014838784 || detlayerId == 2031616000) {
      // Apply position smearing
      double x_trk2 = simHit->entryPoint().x() + CLHEP::RandGauss::shoot(engine, 0, fHitSigmaX * um_to_mm);
      double y_trk2 = simHit->entryPoint().y() + CLHEP::RandGauss::shoot(engine, 0, fHitSigmaY * um_to_mm);
      double z_trk2 = simHit->entryPoint().z();
      // Apply fiducial cuts
      double pos_trk2 = fTrackerInsertion * fBeamXRMS_Trk2 + fTrk2XOffset;
      if (x_trk2 > 0 || fabs(x_trk2) < pos_trk2 || fabs(x_trk2) > (fTrackerWidth + pos_trk2) ||
          fabs(y_trk2) > fTrackerHeight / 2.)
        outside = true;

      float tof = 0.0;
      unsigned int cellId = 2;
      Local3DPoint xyzzy = Local3DPoint(x_trk2, y_trk2, z_trk2);
      if (!outside) {
        rechit.setLocal3DPoint(xyzzy);
        rechit.setTof(tof);
        rechit.setDetUnitId(detlayerId);
        rechit.setCellId(cellId);
        theCTPPSFastRecHit.push_back(rechit);
      }
    }
    //Make Timing RecHits by smearing the SimHits time of flight and checking the cell of the hit
    //The RecHit position is the centre of the cell
    if (detlayerId == 2046820352 || detlayerId == 2063597568) {
      float t = simHit->tof();
      unsigned int cellId = 0;
      float tof = CLHEP::RandGauss::shoot(engine, t, fTimeSigma);
      double x_tof = simHit->entryPoint().x();
      double y_tof = simHit->entryPoint().y();

      double pos_tof = fToFInsertion * fBeamXRMS_ToF + fToFXOffset;

      std::vector<double> vToFCellWidth;
      vToFCellWidth.reserve(8);
      for (int i = 0; i < 8; i++) {
        vToFCellWidth.push_back(fToFCellWidth[i]);
      }
      CTPPSToFDetector* ToFDet = new CTPPSToFDetector(
          fToFNCellX, fToFNCellY, vToFCellWidth, fToFCellHeight, fToFPitchX, fToFPitchY, pos_tof, fTimeSigma);
      cellId = ToFDet->findCellId(x_tof, y_tof);
      if (cellId > 0) {
        double xc_tof = 0., yc_tof = 0.;
        ToFDet->get_CellCenter(cellId, xc_tof, yc_tof);
        Local3DPoint xyzzy = Local3DPoint(xc_tof, yc_tof, simHit->entryPoint().z());
        rechit.setLocal3DPoint(xyzzy);
        rechit.setTof(tof);
        rechit.setDetUnitId(detlayerId);
        rechit.setCellId(cellId);
        theCTPPSFastRecHit.push_back(rechit);
      }
    }
  }

  std::unique_ptr<CTPPSFastRecHitContainer> output_recHits(new edm::CTPPSFastRecHitContainer);
  output_recHits->reserve(simHits->size());

  for (std::vector<CTPPSFastRecHit>::const_iterator i = theCTPPSFastRecHit.begin(); i != theCTPPSFastRecHit.end();
       i++) {
    output_recHits->push_back(*i);
  }

  iEvent.put(std::move(output_recHits), "CTPPSFastRecHits");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void CTPPSRecHitProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void CTPPSRecHitProducer::endStream() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CTPPSRecHitProducer);
