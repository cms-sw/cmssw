#include "RecoTauTag/HLTProducers/interface/L1THLTTauMatching.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/TauReco/interface/PFTau.h"

//
// class decleration
//
using namespace reco   ;
using namespace std    ;
using namespace edm    ;
using namespace trigger;

L1THLTTauMatching::L1THLTTauMatching(const edm::ParameterSet& iConfig):
  jetSrc    ( consumes<PFTauCollection>                     (iConfig.getParameter<InputTag>("JetSrc"      ) ) ),
  tauTrigger( consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("L1TauTrigger") ) ),
  mEt_Min   (                                                iConfig.getParameter<double>  ("EtMin"       )   )
{  
  produces<PFTauCollection>();
}
L1THLTTauMatching::~L1THLTTauMatching(){ }

void L1THLTTauMatching::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const
{
    
  auto_ptr<PFTauCollection> tauL2jets(new PFTauCollection);
    
  double deltaR    = 1.0;
  double matchingR = 0.5;
  
  // Getting HLT jets to be matched
  edm::Handle<PFTauCollection > tauJets;
  iEvent.getByToken( jetSrc, tauJets );
        
  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;
  iEvent.getByToken(tauTrigger,l1TriggeredTaus);
                
  l1t::TauVectorRef tauCandRefVec;
  l1TriggeredTaus->getObjects( trigger::TriggerL1Tau,tauCandRefVec);

  math::XYZPoint a(0.,0.,0.);
        
  for(unsigned int iL1Tau = 0; iL1Tau < tauCandRefVec.size(); iL1Tau++){  
    for(unsigned int iJet = 0; iJet < tauJets->size(); iJet++){
      // Find the relative L2TauJets, to see if it has been reconstructed
      const PFTau &  myJet = (*tauJets)[iJet];
      deltaR = ROOT::Math::VectorUtil::DeltaR(myJet.p4().Vect(), (tauCandRefVec[iL1Tau]->p4()).Vect());
      if(deltaR < matchingR ) {
        if(myJet.leadPFChargedHadrCand().isNonnull()){
          a =  myJet.leadPFChargedHadrCand()->vertex();  
        }
        PFTau myPFTau(std::numeric_limits<int>::quiet_NaN(), myJet.p4(), a);
        if(myJet.pt() > mEt_Min) {
          tauL2jets->push_back(myPFTau);
        }
        break;
      }
    }
  }  
   
  iEvent.put(tauL2jets);
}

void L1THLTTauMatching::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1TauTrigger", edm::InputTag("hltL1sDoubleIsoTau40er"                     ))->setComment("Name of trigger filter"    );
  desc.add<edm::InputTag>("JetSrc"      , edm::InputTag("hltSelectedPFTausTrackPt1MediumIsolationReg"))->setComment("Input collection of PFTaus");
  desc.add<double>       ("EtMin",0.0)->setComment("Minimal pT of PFTau to match");
  descriptions.setComment("This module produces collection of PFTaus matched to L1 Taus / Jets passing a HLT filter (Only p4 and vertex of returned PFTaus are set).");
  descriptions.add       ("L1THLTTauMatching",desc);
}
