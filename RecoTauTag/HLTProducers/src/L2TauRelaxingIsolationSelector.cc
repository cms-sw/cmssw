
#include "RecoTauTag/HLTProducers/interface/L2TauRelaxingIsolationSelector.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"

using namespace reco; 

L2TauRelaxingIsolationSelector::L2TauRelaxingIsolationSelector(const edm::ParameterSet& iConfig):
  associationInput_(consumes<L2TauInfoAssociation>(iConfig.getParameter<edm::InputTag>("L2InfoAssociation"))),
  ecalIsolEt_(iConfig.getParameter<std::vector<double> >("EcalIsolationEt")),
  towerIsolEt_(iConfig.getParameter<std::vector<double> >("TowerIsolationEt")),
  nClusters_(iConfig.getParameter<std::vector<double> >("NumberOfClusters")),
  phiRMS_(iConfig.getParameter<std::vector<double> >("ClusterPhiRMS")),
  etaRMS_(iConfig.getParameter<std::vector<double> >("ClusterEtaRMS")),
  drRMS_(iConfig.getParameter<std::vector<double> >("ClusterDRRMS")),
  et_(iConfig.getParameter<double>("MinJetEt")),
  seedTowerEt_(iConfig.getParameter<double>("SeedTowerEt"))
{
  produces<CaloJetCollection>("Isolated");
}


L2TauRelaxingIsolationSelector::~L2TauRelaxingIsolationSelector()
{

}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
L2TauRelaxingIsolationSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   edm::Handle<L2TauInfoAssociation> Imap;

   std::auto_ptr<CaloJetCollection> l2IsolCaloJets( new CaloJetCollection );
   iEvent.getByToken(associationInput_ ,Imap); 

   if(Imap->size()>0)
	 for(L2TauInfoAssociation::const_iterator p = Imap->begin();p!=Imap->end();++p)
	   {
	     //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	     const L2TauIsolationInfo l2info = p->val;
	     //Retrieve the Jet
	     const CaloJet jet =*(p->key);
	     
	     //If The Cuts are Satisfied
 	   if(jet.et()>et_) 
	     if(l2info.seedHcalHitEt()>seedTowerEt_)
	       if(l2info.ecalIsolEt()< ecalIsolEt_[0]+ecalIsolEt_[1]*jet.et()+ecalIsolEt_[2]*jet.et()*jet.et())
		 if(l2info.nEcalHits() <(int)(nClusters_[0]+nClusters_[1]*jet.et()+nClusters_[2]*jet.et()*jet.et()))
		   if(l2info.ecalClusterShape()[0] <etaRMS_[0]+etaRMS_[1]*jet.et()+etaRMS_[2]*jet.et()*jet.et())
		     if(l2info.ecalClusterShape()[1] <phiRMS_[0]+phiRMS_[1]*jet.et()+phiRMS_[2]*jet.et()*jet.et())
		       if(l2info.ecalClusterShape()[2] <drRMS_[0]+drRMS_[1]*jet.et()+drRMS_[2]*jet.et()*jet.et())
			 if(l2info.hcalIsolEt()<towerIsolEt_[0]+towerIsolEt_[1]*jet.et()+towerIsolEt_[2]*jet.et()*jet.et())
			     {
			       l2IsolCaloJets->push_back(jet);
			     }

	   }
 
        iEvent.put(l2IsolCaloJets, "Isolated");
}

// ------------ method called once each job just before starting event loop  ------------
void 
L2TauRelaxingIsolationSelector::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L2TauRelaxingIsolationSelector::endJob() {
}



