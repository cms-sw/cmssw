// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMissFwd.h"

using namespace l1t;

class L1TkHTMissProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TkHTMissProducer(const edm::ParameterSet&);
  ~L1TkHTMissProducer() override;

private:
  //void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endJob() override;

  // ----------member data ---------------------------
  float jetMinPt_;            // [GeV]
  float jetMaxEta_;           // [rad]
  bool doVtxConstrain_;       // require vertex constraint
  bool primaryVtxConstrain_;  // use event primary vertex instead of leading jet (if doVtxConstrain)
  bool useCaloJets_;          // Determines whether or not calo jets are used
  float deltaZ_;              // for jets [cm] (if DoTvxConstrain)
  float minJetEtLowPt_;       // for track jets, minimum et required, depending on number of low pT tracks
  float minJetEtHighPt_;
  bool displaced_;  // Use prompt/displaced tracks
  unsigned int minNtracksHighPt_;
  unsigned int minNtracksLowPt_;
  const edm::EDGetTokenT<TkPrimaryVertexCollection> pvToken_;
  const edm::EDGetTokenT<TkJetCollection> jetToken_;
};

L1TkHTMissProducer::L1TkHTMissProducer(const edm::ParameterSet& iConfig)
    : pvToken_(consumes<TkPrimaryVertexCollection>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
      jetToken_(consumes<TkJetCollection>(iConfig.getParameter<edm::InputTag>("L1TkJetInputTag"))) {
  jetMinPt_ = (float)iConfig.getParameter<double>("jet_minPt");
  jetMaxEta_ = (float)iConfig.getParameter<double>("jet_maxEta");
  doVtxConstrain_ = iConfig.getParameter<bool>("doVtxConstrain");
  useCaloJets_ = iConfig.getParameter<bool>("useCaloJets");
  primaryVtxConstrain_ = iConfig.getParameter<bool>("primaryVtxConstrain");
  deltaZ_ = (float)iConfig.getParameter<double>("deltaZ");
  minNtracksHighPt_ = iConfig.getParameter<int>("jet_minNtracksHighPt");
  minNtracksLowPt_ = iConfig.getParameter<int>("jet_minNtracksLowPt");
  minJetEtLowPt_ = iConfig.getParameter<double>("jet_minJetEtLowPt");
  minJetEtHighPt_ = iConfig.getParameter<double>("jet_minJetEtHighPt");
  displaced_ = iConfig.getParameter<bool>("displaced");

  if (useCaloJets_)
    produces<TkHTMissCollection>("TkCaloHTMiss");
  else if (displaced_)
    produces<TkHTMissCollection>("L1TrackerHTMissExtended");
  else
    produces<TkHTMissCollection>("L1TrackerHTMiss");
}

L1TkHTMissProducer::~L1TkHTMissProducer() {}

void L1TkHTMissProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::unique_ptr<TkHTMissCollection> MHTCollection(new TkHTMissCollection);

  // L1 primary vertex
  edm::Handle<TkPrimaryVertexCollection> L1VertexHandle;
  iEvent.getByToken(pvToken_, L1VertexHandle);

  // L1 track-trigger jets
  edm::Handle<TkJetCollection> L1TkJetsHandle;
  iEvent.getByToken(jetToken_, L1TkJetsHandle);
  std::vector<TkJet>::const_iterator jetIter;

  if (!L1TkJetsHandle.isValid() && !displaced_) {
    LogError("TkHTMissProducer") << "\nWarning: TkJetCollection not found in the event. Exit\n";
    return;
  }

  if (!L1TkJetsHandle.isValid() && displaced_) {
    LogError("TkHTMissProducer") << "\nWarning: TkJetExtendedCollection not found in the event. Exit\n";
    return;
  }

  // ----------------------------------------------------------------------------------------------
  // if primaryVtxConstrain_, use the primary vertex instead of z position from leading jet
  // ----------------------------------------------------------------------------------------------
  float evtZVtx = 999;
  bool foundVtx = false;
  edm::Ref<TkPrimaryVertexCollection> L1VtxRef;  // null reference

  if (useCaloJets_) {
    if (doVtxConstrain_ && primaryVtxConstrain_) {
      if (!L1VertexHandle.isValid()) {
        LogError("L1TkHTMissProducer") << "\nWarning: TkPrimaryVertexCollection not found in the event. Exit\n";
        return;
      } else {
        std::vector<TkPrimaryVertex>::const_iterator vtxIter = L1VertexHandle->begin();
        // by convention, the first vertex in the collection is the one that should
        // be used by default
        evtZVtx = vtxIter->zvertex();
        foundVtx = true;
        int ivtx = 0;
        edm::Ref<TkPrimaryVertexCollection> vtxRef(L1VertexHandle, ivtx);
        L1VtxRef = vtxRef;
      }
    }  //endif primaryVtxConstrain_

    // ----------------------------------------------------------------------------------------------
    // using z position of leading jet to define "event vertex"
    // ----------------------------------------------------------------------------------------------
    float zvtx_jetpt = -1.0;  //pt of jet determining the event vertex
    float jetVtxMax = 99.;    //find z position of leading jet that has a z vertex!

    if (doVtxConstrain_ && !primaryVtxConstrain_) {
      for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {
        int ibx = jetIter->bx();  // only consider jets from the central BX
        if (ibx != 0)
          continue;

        float tmp_jet_vtx = jetIter->jetVtx();
        float tmp_jet_pt = jetIter->pt();
        float tmp_jet_eta = jetIter->eta();
        if (tmp_jet_pt < jetMinPt_)
          continue;
        if (fabs(tmp_jet_eta) > jetMaxEta_)
          continue;
        if (fabs(tmp_jet_vtx) > jetVtxMax)
          continue;

        // find vertex position of leading jet
        if (tmp_jet_pt > zvtx_jetpt) {
          evtZVtx = tmp_jet_vtx;
          zvtx_jetpt = tmp_jet_pt;
          foundVtx = true;
        }
      }  //end loop over jets
    }    //endif z position from leading jet

    float sumPx_calo = 0;
    float sumPy_calo = 0;
    float HT_calo = 0;

    if (doVtxConstrain_ && !foundVtx)
      LogWarning("L1TkHTMissProducer") << "Didn't find any z vertex (based on jet vertices) for this event!\n";

    // loop over jets
    for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {
      int ibx = jetIter->bx();  // only consider jets from the central BX
      if (ibx != 0)
        continue;

      float tmp_jet_px = jetIter->px();
      float tmp_jet_py = jetIter->py();
      float tmp_jet_et = jetIter->et();
      float tmp_jet_vtx = jetIter->jetVtx();
      if (jetIter->pt() < jetMinPt_)
        continue;
      if (fabs(jetIter->eta()) > jetMaxEta_)
        continue;

      // vertex consistency requirement
      bool VtxRequirement = false;
      if (foundVtx)
        VtxRequirement = fabs(tmp_jet_vtx - evtZVtx) < deltaZ_;

      if (!doVtxConstrain_ || VtxRequirement) {
        sumPx_calo += tmp_jet_px;
        sumPy_calo += tmp_jet_py;
        HT_calo += tmp_jet_et;
      }
    }  //end loop over jets

    // define missing HT
    float et = sqrt(sumPx_calo * sumPx_calo + sumPy_calo * sumPy_calo);
    math::XYZTLorentzVector missingEt(-sumPx_calo, -sumPy_calo, 0, et);
    edm::RefProd<TkJetCollection> jetCollRef(L1TkJetsHandle);
    TkHTMiss tkHTM(missingEt, HT_calo, jetCollRef, L1VtxRef);

    if (doVtxConstrain_ && !primaryVtxConstrain_) {
      tkHTM.setVtx(evtZVtx);
    }

    MHTCollection->push_back(tkHTM);
    iEvent.put(std::move(MHTCollection), "L1TkCaloHTMiss");
  }

  else {  // Using standalone jets
    float sumPx = 0;
    float sumPy = 0;
    float HT = 0;

    // loop over jets
    for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {
      float tmp_jet_px = jetIter->px();
      float tmp_jet_py = jetIter->py();
      float tmp_jet_et = jetIter->et();
      float tmp_jet_pt = jetIter->pt();
      if (tmp_jet_pt < jetMinPt_)
        continue;
      if (fabs(jetIter->eta()) > jetMaxEta_)
        continue;
      if (jetIter->ntracks() < minNtracksLowPt_ && tmp_jet_et > minJetEtLowPt_)
        continue;
      if (jetIter->ntracks() < minNtracksHighPt_ && tmp_jet_et > minJetEtHighPt_)
        continue;
      sumPx += tmp_jet_px;
      sumPy += tmp_jet_py;
      HT += tmp_jet_pt;
    }  // end jet loop

    // define missing HT
    float et = sqrt(sumPx * sumPx + sumPy * sumPy);
    math::XYZTLorentzVector missingEt(-sumPx, -sumPy, 0, et);
    edm::RefProd<TkJetCollection> jetCollRef(L1TkJetsHandle);
    TkHTMiss tkHTM(missingEt, HT, jetCollRef, L1VtxRef);

    MHTCollection->push_back(tkHTM);
    if (displaced_)
      iEvent.put(std::move(MHTCollection), "L1TrackerHTMissExtended");
    else
      iEvent.put(std::move(MHTCollection), "L1TrackerHTMiss");
  }
}  //end producer

//void L1TkHTMissProducer::beginJob() {}

//void L1TkHTMissProducer::endJob() {}

DEFINE_FWK_MODULE(L1TkHTMissProducer);
