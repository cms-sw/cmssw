#include "RecoTauTag/HLTProducers/interface/L2TauJetsMerger.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;

L2TauJetsMerger::L2TauJetsMerger(const edm::ParameterSet& iConfig)
{
  jetSrc = iConfig.getParameter<vtag>("JetSrc");
  for(vtag::const_iterator it = jetSrc.begin(); it != jetSrc.end(); ++it) {
    edm::EDGetTokenT<CaloJetCollection> aToken = consumes<CaloJetCollection>(*it);
    jetSrc_token.push_back(aToken);
  }
  mEt_Min = iConfig.getParameter<double>("EtMin");
  
  produces<CaloJetCollection>();
}

L2TauJetsMerger::~L2TauJetsMerger(){ }

void L2TauJetsMerger::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

 using namespace edm;
 using namespace std;
 using namespace reco;

 //Getting all the L1Seeds

 
 //Getting the Collections of L2ReconstructedJets from L1Seeds
 //and removing the collinear jets
 myL2L1JetsMap.clear();
 CaloJetCollection myTmpJets;
 myTmpJets.clear();

 int iL1Jet = 0;
 for( vtoken_cjets::const_iterator s = jetSrc_token.begin(); s != jetSrc_token.end(); ++ s ) {
   edm::Handle<CaloJetCollection> tauJets;
   iEvent.getByToken( * s, tauJets );
   for(CaloJetCollection::const_iterator iTau = tauJets->begin(); iTau !=tauJets->end(); ++iTau)
     { 
     //Create a Map to associate to every Jet its L1SeedId, i.e. 0,1,2 or 3
       if(iTau->et() > mEt_Min) {

	 //Add the Pdg Id here 
	 CaloJet myJet = *iTau;
	 myJet.setPdgId(15);
	 myTmpJets.push_back(myJet);
       }
     }
   iL1Jet++;
 }

 std::auto_ptr<CaloJetCollection> tauL2jets(new CaloJetCollection); 

 //Removing the collinear jets correctly!

 //First sort the jets you have merged
 SorterByPt sorter;
 std::sort(myTmpJets.begin(),myTmpJets.end(),sorter);
 
//Remove Collinear Jets by prefering the highest ones!

   while(myTmpJets.size()>0) {
     tauL2jets->push_back(myTmpJets.at(0));
     CaloJetCollection tmp;
     for(unsigned int i=1 ;i<myTmpJets.size();++i) {
       double DR = ROOT::Math::VectorUtil::DeltaR(myTmpJets.at(0).p4(),myTmpJets.at(i).p4());
       if(DR>0.1) 
	 tmp.push_back(myTmpJets.at(i));
     }
     myTmpJets.swap(tmp);
     tmp.clear();
   }


  iEvent.put(tauL2jets);

}
