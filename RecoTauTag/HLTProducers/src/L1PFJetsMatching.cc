#include "RecoTauTag/HLTProducers/interface/L1PFJetsMatching.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/JetReco/interface/PFJet.h"

//
// class decleration
//
using namespace reco   ;
using namespace std    ;
using namespace edm    ;
using namespace trigger;

std::pair<PFJetCollection,PFJetCollection> Categorise(PFJetCollection CaloL2jets, double pt1, double pt2, double Mjj)
{

int A=0;
    std::pair<PFJetCollection,PFJetCollection> output;
   unsigned int i1 = 0;
   unsigned int i2 = 0;
    double mjj = -9999;
    if (CaloL2jets.size()>1){
    for (unsigned int i = 0; i < CaloL2jets.size()-1; i++)
        for (unsigned int j = i+1; j < CaloL2jets.size(); j++)
    {
        const PFJet &  myJet1 = (CaloL2jets)[i];
        const PFJet &  myJet2 = (CaloL2jets)[j];
        
        
        if ((myJet1.p4()+myJet2.p4()).M()>mjj){
            
         mjj =(myJet1.p4()+myJet2.p4()).M();
            i1 = i;
            i2 = j;
        }
    }
        //std::cout<<"mjj:= "<<mjj<<"; i1,i2: = "<<i1<<" "<<i2<<std::endl;
        
        
            
            const PFJet &  myJet1 = (CaloL2jets)[i1];
            const PFJet &  myJet2 = (CaloL2jets)[i2];
        
        if ((myJet1.p4().Pt() >= pt1) && (myJet2.p4().Pt() > pt2) && (mjj > Mjj))
        {
            
            output.first.push_back(myJet1);
            output.first.push_back(myJet2);
            A++;
	}
        
        if ((myJet1.p4().Pt() < pt1) && (myJet1.p4().Pt() > pt2) && (myJet2.p4().Pt() > pt2) && (mjj > Mjj))
        {
            
            const PFJet &  myJetTest = (CaloL2jets)[0];
         if (myJetTest.p4().Pt()>90){   
            output.second.push_back(myJet1);
            output.second.push_back(myJet2);
            output.second.push_back(myJetTest);
            A+=2;   
		}
        }
        
    }
    //std::cout<<" "<<A<<std::endl;
            return output;
        
}

L1PFJetsMatching::L1PFJetsMatching(const edm::ParameterSet& iConfig):
  jetSrc    ( consumes<PFJetCollection>                     (iConfig.getParameter<InputTag>("JetSrc"      ) ) ),
  jetTrigger( consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("L1JetTrigger") ) ),
  pt1_Min   ( iConfig.getParameter<double>("pt1_Min")),
  pt2_Min   ( iConfig.getParameter<double>("pt2_Min")),
  mjj_Min   ( iConfig.getParameter<double>("mjj_Min"))
{  
  produces<PFJetCollection>("TwoJets");
  produces<PFJetCollection>("ThreeJets");
}
L1PFJetsMatching::~L1PFJetsMatching(){ }

void L1PFJetsMatching::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const
{
    
  unique_ptr<PFJetCollection> caloL2jets(new PFJetCollection);
    std::pair<PFJetCollection,PFJetCollection> Output;
    
  double deltaR    = 1.0;
  double matchingR = 0.5;
  
  // Getting HLT jets to be matched
  edm::Handle<PFJetCollection > caloJets;
  iEvent.getByToken( jetSrc, caloJets );
        
  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredJets;
  iEvent.getByToken(jetTrigger,l1TriggeredJets);
                
  //l1t::TauVectorRef jetCandRefVec;
  l1t::JetVectorRef jetCandRefVec;
  l1TriggeredJets->getObjects( trigger::TriggerL1Jet,jetCandRefVec);

  math::XYZPoint a(0.,0.,0.);
        
 //std::cout<<"PFsize= "<<caloJets->size()<<endl<<" L1size= "<<jetCandRefVec.size()<<std::endl;
 for(unsigned int iJet = 0; iJet < caloJets->size(); iJet++){
    for(unsigned int iL1Jet = 0; iL1Jet < jetCandRefVec.size(); iL1Jet++){
      // Find the relative L2caloJets, to see if it has been reconstructed
      const PFJet &  myJet = (*caloJets)[iJet];
    //  if ((iJet<3) && (iL1Jet==0))  std::cout<<myJet.p4().Pt()<<" ";
      deltaR = ROOT::Math::VectorUtil::DeltaR(myJet.p4().Vect(), (jetCandRefVec[iL1Jet]->p4()).Vect());
      if(deltaR < matchingR ) {
       
        //if(myJet.pt() > mEt_Min) {
            caloL2jets->push_back(myJet);
       // }
        break;
      }
    }
  }  
   
    
    
    
    Output= Categorise(*caloL2jets,pt1_Min,pt2_Min, mjj_Min);
    unique_ptr<PFJetCollection> Output1(new PFJetCollection(Output.first));
    unique_ptr<PFJetCollection> Output2(new PFJetCollection(Output.second));
    //unique_ptr<std::pair<PFJetCollection,PFJetCollection>> Output(new std::pair<PFJetCollection,PFJetCollection>);
    //std::cout<<"Storing the event: "<<std::endl;
   
    
    iEvent.put(std::move(Output1),"TwoJets");
    iEvent.put(std::move(Output2),"ThreeJets");
    

}

void L1PFJetsMatching::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1JetTrigger", edm::InputTag("hltL1sDoubleIsoTau40er"                     ))->setComment("Name of trigger filter"    );
  desc.add<edm::InputTag>("JetSrc"      , edm::InputTag("hltSelectedCaloJetsTrackPt1MediumIsolationReg"))->setComment("Input collection of PFJets");
  desc.add<double>       ("pt1_Min",95.0)->setComment("Minimal pT1 of PFJets to match");
  desc.add<double>       ("pt2_Min",35.0)->setComment("Minimal pT2 of PFJets to match");
  desc.add<double>       ("mjj_Min",650.0)->setComment("Minimal mjj of matched PFjets");
  descriptions.setComment("This module produces collection of PFJetss matched to L1 Taus / Jets passing a HLT filter (Only p4 and vertex of returned PFJetss are set).");
  descriptions.add       ("L1PFJetsMatching",desc);
}
