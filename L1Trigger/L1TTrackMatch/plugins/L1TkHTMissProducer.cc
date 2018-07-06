// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticleFwd.h"


using namespace l1t;

class L1TkHTMissProducer : public edm::EDProducer {
  public:

    explicit L1TkHTMissProducer(const edm::ParameterSet&);
    ~L1TkHTMissProducer();

  private:
    virtual void beginJob() ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;

    // ----------member data ---------------------------
    float jet_minPt;                // [GeV]
    float jet_maxEta;               // [rad]
    bool DoVtxConstrain;            // require vertex constraint
    bool PrimaryVtxConstrain;       // use event primary vertex instead of leading jet (if DoVtxConstrain)
    bool UseCaloJets;            // Determines whether or not calo jets are used
    float DeltaZ;                   // for jets [cm] (if DoTvxConstrain)
    const edm::EDGetTokenT< L1TkPrimaryVertexCollection > pvToken;
    const edm::EDGetTokenT< L1TkJetParticleCollection > jetToken;
};

///////////////
//constructor//
///////////////
L1TkHTMissProducer::L1TkHTMissProducer(const edm::ParameterSet& iConfig) :
  pvToken(consumes<L1TkPrimaryVertexCollection>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
  jetToken(consumes<L1TkJetParticleCollection>(iConfig.getParameter<edm::InputTag>("L1TkJetInputTag")))
{
  jet_minPt  = (float)iConfig.getParameter<double>("jet_minPt");
  jet_maxEta = (float)iConfig.getParameter<double>("jet_maxEta");
  DoVtxConstrain      = iConfig.getParameter<bool>("DoVtxConstrain");
  UseCaloJets      = iConfig.getParameter<bool>("UseCaloJets");
  PrimaryVtxConstrain = iConfig.getParameter<bool>("PrimaryVtxConstrain");
  DeltaZ              = (float)iConfig.getParameter<double>("DeltaZ");

  if (UseCaloJets) produces<L1TkHTMissParticleCollection>("L1TkCaloHTMiss");
  else produces<L1TkHTMissParticleCollection>("L1TrackerHTMiss");
}

//////////////
//destructor//
//////////////
L1TkHTMissProducer::~L1TkHTMissProducer() {
}

////////////
//producer//
////////////
void L1TkHTMissProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<L1TkHTMissParticleCollection> MHTCollection(new L1TkHTMissParticleCollection);

  // L1 primary vertex
  edm::Handle<L1TkPrimaryVertexCollection> L1VertexHandle;
  iEvent.getByToken(pvToken, L1VertexHandle);
  std::vector<L1TkPrimaryVertex>::const_iterator vtxIter;

  // L1 track-trigger jets
  edm::Handle<L1TkJetParticleCollection> L1TkJetsHandle;
  iEvent.getByToken(jetToken, L1TkJetsHandle);
  std::vector<L1TkJetParticle>::const_iterator jetIter;

  if ( ! L1TkJetsHandle.isValid() ) {
    LogError("L1TkHTMissProducer")<< "\nWarning: L1TkJetParticleCollection not found in the event. Exit"<< std::endl;
    return;
  }

  // ----------------------------------------------------------------------------------------------
  // if PrimaryVtxConstrain, use the primary vertex instead of z position from leading jet
  // ----------------------------------------------------------------------------------------------
  float evt_zvtx = 999;
  bool found_vtx = false;
  edm::Ref< L1TkPrimaryVertexCollection > L1VtxRef; 	// null reference
  if ( DoVtxConstrain && PrimaryVtxConstrain && UseCaloJets) {
    if( !L1VertexHandle.isValid() ) {
      LogError("L1TkHTMissProducer")<< "\nWarning: L1TkPrimaryVertexCollection not found in the event. Exit."<< std::endl;
      return ;
    }
    else {
      std::vector<L1TkPrimaryVertex>::const_iterator vtxIter = L1VertexHandle->begin();
      // by convention, the first vertex in the collection is the one that should
      // be used by default
      evt_zvtx = vtxIter->getZvertex();
      found_vtx = true;
      int ivtx = 0;
      edm::Ref< L1TkPrimaryVertexCollection > vtxRef( L1VertexHandle, ivtx );
      L1VtxRef = vtxRef;
    }
  } //endif PrimaryVtxConstrain

  // ----------------------------------------------------------------------------------------------
  // using z position of leading jet to define "event vertex"
  // ----------------------------------------------------------------------------------------------
  float zvtx_jetpt = -1.0; //pt of jet determining the event vertex
  float JET_VTXMAX = 99.;  //find z position of leading jet that has a z vertex!

  if ( DoVtxConstrain && !PrimaryVtxConstrain && UseCaloJets) {
    for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {
      int ibx = jetIter->bx(); // only consider jets from the central BX
      if (ibx != 0) continue;

      float tmp_jet_vtx = jetIter->getJetVtx();
      float tmp_jet_pt  = jetIter->pt();
      float tmp_jet_eta = jetIter->eta();
      if (tmp_jet_pt < jet_minPt) continue;
      if (fabs(tmp_jet_eta) > jet_maxEta) continue;
      if (fabs(tmp_jet_vtx) > JET_VTXMAX) continue;

      // find vertex position of leading jet
      if (tmp_jet_pt > zvtx_jetpt) {
        evt_zvtx = tmp_jet_vtx;
        zvtx_jetpt = tmp_jet_pt;
        found_vtx = true;
      }
    } //end loop over jets
  } //endif z position from leading jet

  if (UseCaloJets){
    float sumPx_calo = 0;
    float sumPy_calo = 0;
    float HT_calo = 0;

    if (DoVtxConstrain && !found_vtx) std::cout << "WARNING from L1TkHTMissProducer: didn't find any z vertex (based on jet vertices) for this event!" << std::endl;

    // loop over jets
    for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {
      int ibx = jetIter->bx(); // only consider jets from the central BX
      if (ibx != 0) continue;

      float px = jetIter->px();
      float py = jetIter->py();
      float et = jetIter->et();
      float tmp_jet_vtx = jetIter->getJetVtx();
      float tmp_jet_pt  = jetIter->pt();
      float tmp_jet_eta = jetIter->eta();
      if (tmp_jet_pt < jet_minPt) continue;
      if (fabs(tmp_jet_eta) > jet_maxEta) continue;

      // vertex consistency requirement
      bool VtxRequirement = false;
      if (found_vtx) VtxRequirement = fabs(tmp_jet_vtx - evt_zvtx) < DeltaZ;

      if (!DoVtxConstrain || VtxRequirement) {
        sumPx_calo += px;
        sumPy_calo += py;
        HT_calo += et;
      }
    } //end loop over jets

    // define missing HT
    float et = sqrt(sumPx_calo*sumPx_calo + sumPy_calo*sumPy_calo);
    math::XYZTLorentzVector missingEt(-sumPx_calo, -sumPy_calo, 0, et);
    edm::RefProd<L1TkJetParticleCollection> jetCollRef(L1TkJetsHandle);
    L1TkHTMissParticle tkHTM(missingEt, HT_calo, jetCollRef, L1VtxRef);

    if (DoVtxConstrain && !PrimaryVtxConstrain) {
      tkHTM.setVtx(evt_zvtx);
    }

    MHTCollection->push_back(tkHTM);
    iEvent.put( std::move(MHTCollection), "L1TkCaloHTMiss" );
  }

  // Using standalone jets
  else {
    float sumPx = 0;
    float sumPy = 0;
    float HT = 0;

    // loop over jets
    for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {
      int ibx = jetIter->bx(); // only consider jets from the central BX
      if (ibx != 0) continue;

      float px = jetIter->px();
      float py = jetIter->py();
      float et = jetIter->et();
      float tmp_jet_pt  = jetIter->pt();
      float tmp_jet_eta = jetIter->eta();
      if (tmp_jet_pt < jet_minPt) continue;
      if (fabs(tmp_jet_eta) > jet_maxEta) continue;

      sumPx += px;
      sumPy += py;
      HT += et;
    } // end jet loop

    // define missing HT
    float et = sqrt(sumPx*sumPx + sumPy*sumPy);
    math::XYZTLorentzVector missingEt(-sumPx, -sumPy, 0, et);
    edm::RefProd<L1TkJetParticleCollection> jetCollRef(L1TkJetsHandle);
    L1TkHTMissParticle tkHTM(missingEt, HT, jetCollRef, L1VtxRef);

    MHTCollection->push_back(tkHTM);
    iEvent.put( std::move(MHTCollection), "L1TrackerHTMiss" );
  }

} //end producer

// ------------ method called once each job just before starting event loop  ------------
void L1TkHTMissProducer::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void L1TkHTMissProducer::endJob() {
}


DEFINE_FWK_MODULE(L1TkHTMissProducer);
