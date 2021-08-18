// -*- C++ -*-
//
//
// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013
// $Id$
//
//
// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

////////////////////////////
// HepMC products
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "TH1F.h"

using namespace l1t;

//
// class declaration
//

class L1TkFastVertexProducer : public edm::global::EDProducer<> {
public:
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollectionType;

  explicit L1TkFastVertexProducer(const edm::ParameterSet&);
  ~L1TkFastVertexProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  float zMax_;   // in cm
  float DeltaZ;  // in cm
  float chi2Max_;
  float pTMinTra_;  // in GeV

  float pTMax_;       // in GeV, saturation / truncation value
  int highPtTracks_;  // saturate or truncate

  int nVtx_;         // the number of vertices to return
  int nStubsmin_;    // minimum number of stubs
  int nStubsPSmin_;  // minimum number of stubs in PS modules

  int nBinning_;  // number of bins used in the temp histogram

  bool monteCarloVertex_;  //
                           //const StackedTrackerGeometry*                   theStackedGeometry;

  bool doPtComp_;
  bool doTightChi2_;
  float trkPtTightChi2_;
  float trkChi2dofTightChi2_;

  int weight_;  // weight (power) of pT 0 , 1, 2

  constexpr static float xmin_ = -30;
  constexpr static float xmax_ = +30;

  const edm::EDGetTokenT<edm::HepMCProduct> hepmcToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle> > genparticleToken_;
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > trackToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
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
L1TkFastVertexProducer::L1TkFastVertexProducer(const edm::ParameterSet& iConfig)
    : hepmcToken_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("HepMCInputTag"))),
      genparticleToken_(
          consumes<std::vector<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("GenParticleInputTag"))),
      trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(
          iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))) {
  zMax_ = (float)iConfig.getParameter<double>("ZMAX");
  chi2Max_ = (float)iConfig.getParameter<double>("CHI2MAX");
  pTMinTra_ = (float)iConfig.getParameter<double>("PTMINTRA");

  pTMax_ = (float)iConfig.getParameter<double>("PTMAX");
  highPtTracks_ = iConfig.getParameter<int>("HighPtTracks");

  nVtx_ = iConfig.getParameter<int>("nVtx");
  nStubsmin_ = iConfig.getParameter<int>("nStubsmin");
  nStubsPSmin_ = iConfig.getParameter<int>("nStubsPSmin");
  nBinning_ = iConfig.getParameter<int>("nBinning");

  monteCarloVertex_ = iConfig.getParameter<bool>("MonteCarloVertex");
  doPtComp_ = iConfig.getParameter<bool>("doPtComp");
  doTightChi2_ = iConfig.getParameter<bool>("doTightChi2");
  trkPtTightChi2_ = (float)iConfig.getParameter<double>("trk_ptTightChi2");
  trkChi2dofTightChi2_ = (float)iConfig.getParameter<double>("trk_chi2dofTightChi2");

  weight_ = iConfig.getParameter<int>("WEIGHT");

  produces<TkPrimaryVertexCollection>();
}

L1TkFastVertexProducer::~L1TkFastVertexProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1TkFastVertexProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  auto result = std::make_unique<TkPrimaryVertexCollection>();

  // Tracker Topology
  const TrackerTopology& tTopo = iSetup.getData(topoToken_);

  TH1F htmp("htmp", ";z (cm); Tracks", nBinning_, xmin_, xmax_);
  TH1F htmp_weight("htmp_weight", ";z (cm); Tracks", nBinning_, xmin_, xmax_);

  // ----------------------------------------------------------------------

  if (monteCarloVertex_) {
    // MC info  ... retrieve the zvertex
    edm::Handle<edm::HepMCProduct> HepMCEvt;
    iEvent.getByToken(hepmcToken_, HepMCEvt);

    edm::Handle<std::vector<reco::GenParticle> > GenParticleHandle;
    iEvent.getByToken(genparticleToken_, GenParticleHandle);

    const double mm = 0.1;
    float zvtx_gen = -999;

    if (HepMCEvt.isValid()) {
      // using HepMCEvt

      const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();
      for (HepMC::GenEvent::vertex_const_iterator ivertex = MCEvt->vertices_begin(); ivertex != MCEvt->vertices_end();
           ++ivertex) {
        bool hasParentVertex = false;

        // Loop over the parents looking to see if they are coming from a production vertex
        for (HepMC::GenVertex::particle_iterator iparent = (*ivertex)->particles_begin(HepMC::parents);
             iparent != (*ivertex)->particles_end(HepMC::parents);
             ++iparent)
          if ((*iparent)->production_vertex()) {
            hasParentVertex = true;
            break;
          }

        // Reject those vertices with parent vertices
        if (hasParentVertex)
          continue;
        // Get the position of the vertex
        HepMC::FourVector pos = (*ivertex)->position();
        zvtx_gen = pos.z() * mm;
        break;  // there should be one single primary vertex
      }         // end loop over gen vertices

    } else if (GenParticleHandle.isValid()) {
      for (const auto& genpart : *GenParticleHandle) {
        int status = genpart.status();
        if (status != 3)
          continue;
        if (genpart.numberOfMothers() == 0)
          continue;  // the incoming hadrons
        float part_zvertex = genpart.vz();
        zvtx_gen = part_zvertex;
        break;  //
      }
    } else {
      throw cms::Exception("L1TkFastVertexProducer")
          << "\nerror: try to retrieve the MC vertex (monteCarloVertex_ = True) "
          << "\nbut the input file contains neither edm::HepMCProduct> nor vector<reco::GenParticle>. Exit"
          << std::endl;
    }

    TkPrimaryVertex genvtx(zvtx_gen, -999.);

    result->push_back(genvtx);
    iEvent.put(std::move(result));
    return;
  }

  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken_, L1TTTrackHandle);

  if (!L1TTTrackHandle.isValid()) {
    throw cms::Exception("L1TkFastVertexProducer")
        << "\nWarning: L1TkTrackCollection with not found in the event. Exit" << std::endl;
    return;
  }

  for (const auto& track : *L1TTTrackHandle) {
    float z = track.POCA().z();
    float chi2 = track.chi2();
    float pt = track.momentum().perp();
    float eta = track.momentum().eta();

    //..............................................................
    float wt = pow(pt, weight_);  // calculating the weight for tks in as pt^0,pt^1 or pt^2 based on weight_

    if (std::abs(z) > zMax_)
      continue;
    if (chi2 > chi2Max_)
      continue;
    if (pt < pTMinTra_)
      continue;

    // saturation or truncation :
    if (pTMax_ > 0 && pt > pTMax_) {
      if (highPtTracks_ == 0)
        continue;  // ignore this track
      if (highPtTracks_ == 1)
        pt = pTMax_;  // saturate
    }

    // get the number of stubs and the number of stubs in PS layers
    float nPS = 0.;  // number of stubs in PS modules
    float nstubs = 0;

    // get pointers to stubs associated to the L1 track
    const std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >&
        theStubs = track.getStubRefs();

    int tmp_trk_nstub = (int)theStubs.size();
    if (tmp_trk_nstub < 0) {
      LogTrace("L1TkFastVertexProducer")
          << " ... could not retrieve the vector of stubs in L1TkFastVertexProducer::SumPtVertex " << std::endl;
      continue;
    }

    // loop over the stubs
    for (const auto& stub : theStubs) {
      nstubs++;
      bool isPS = false;
      DetId detId(stub->getDetId());
      if (detId.det() == DetId::Detector::Tracker) {
        if (detId.subdetId() == StripSubdetector::TOB && tTopo.tobLayer(detId) <= 3)
          isPS = true;
        else if (detId.subdetId() == StripSubdetector::TID && tTopo.tidRing(detId) <= 9)
          isPS = true;
      }
      if (isPS)
        nPS++;
    }  // end loop over stubs
    if (nstubs < nStubsmin_)
      continue;
    if (nPS < nStubsPSmin_)
      continue;

    // quality cuts from Louise S, based on the pt-stub compatibility (June 20, 2014)
    int trk_nstub = (int)track.getStubRefs().size();
    float chi2dof = chi2 / (2 * trk_nstub - 4);

    if (doPtComp_) {
      float trk_consistency = track.stubPtConsistency();
      if (trk_nstub == 4) {
        if (std::abs(eta) < 2.2 && trk_consistency > 10)
          continue;
        else if (std::abs(eta) > 2.2 && chi2dof > 5.0)
          continue;
      }
    }
    if (doTightChi2_) {
      if (pt > trkPtTightChi2_ && chi2dof > trkChi2dofTightChi2_)
        continue;
    }

    htmp.Fill(z);
    htmp_weight.Fill(z, wt);  // changed from "pt" to "wt" which is some power of pt (0,1 or 2)

  }  // end loop over tracks

  // sliding windows... maximize bin i + i-1  + i+1

  float zvtx_sliding;
  float sigma_max;
  int imax;
  int nb = htmp.GetNbinsX();
  std::vector<int> found;
  found.reserve(nVtx_);
  for (int ivtx = 0; ivtx < nVtx_; ivtx++) {
    zvtx_sliding = -999;
    sigma_max = -999;
    imax = -999;
    for (int i = 2; i <= nb - 1; i++) {
      float a0 = htmp_weight.GetBinContent(i - 1);
      float a1 = htmp_weight.GetBinContent(i);
      float a2 = htmp_weight.GetBinContent(i + 1);
      float sigma = a0 + a1 + a2;
      if ((sigma > sigma_max) && (find(found.begin(), found.end(), i) == found.end())) {
        sigma_max = sigma;
        imax = i;
        float z0 = htmp_weight.GetBinCenter(i - 1);
        float z1 = htmp_weight.GetBinCenter(i);
        float z2 = htmp_weight.GetBinCenter(i + 1);
        zvtx_sliding = (a0 * z0 + a1 * z1 + a2 * z2) / sigma;
      }
    }
    found.push_back(imax);
    TkPrimaryVertex vtx4(zvtx_sliding, sigma_max);
    result->push_back(vtx4);
  }

  iEvent.put(std::move(result));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TkFastVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkFastVertexProducer);
