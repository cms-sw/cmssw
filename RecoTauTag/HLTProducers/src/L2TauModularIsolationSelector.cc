
#include "RecoTauTag/HLTProducers/interface/L2TauModularIsolationSelector.h"

using namespace reco; 

L2TauModularIsolationSelector::L2TauModularIsolationSelector(const edm::ParameterSet& iConfig):
  associationInput_(consumes<L2TauInfoAssociation>(iConfig.getParameter<edm::InputTag>("L2InfoAssociation"))),

  ecalIsolEt_(iConfig.getParameter<std::vector<double> >("EcalIsolationEt")),
  nEcalClusters_(iConfig.getParameter<std::vector<double> >("NumberOfECALClusters")),
  ecalClusterPhiRMS_(iConfig.getParameter<std::vector<double> >("ECALClusterPhiRMS")),
  ecalClusterEtaRMS_(iConfig.getParameter<std::vector<double> >("ECALClusterEtaRMS")),
  ecalClusterDrRMS_(iConfig.getParameter<std::vector<double> >("ECALClusterDRRMS")),
  hcalIsolEt_(iConfig.getParameter<std::vector<double> >("HcalIsolationEt")),
  nHcalClusters_(iConfig.getParameter<std::vector<double> >("NumberOfHCALClusters")),
  hcalClusterPhiRMS_(iConfig.getParameter<std::vector<double> >("HCALClusterPhiRMS")),
  hcalClusterEtaRMS_(iConfig.getParameter<std::vector<double> >("HCALClusterEtaRMS")),
  hcalClusterDrRMS_(iConfig.getParameter<std::vector<double> >("HCALClusterDRRMS")),
  et_(iConfig.getParameter<double>("MinJetEt")),
  seedTowerEt_(iConfig.getParameter<double>("SeedTowerEt"))
{
  produces<CaloJetCollection>("Isolated");
}


L2TauModularIsolationSelector::~L2TauModularIsolationSelector()
{

}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
L2TauModularIsolationSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
		 if(l2info.nEcalHits() <(int)(nEcalClusters_[0]+nEcalClusters_[1]*jet.et()+nEcalClusters_[2]*jet.et()*jet.et()))
		   if(l2info.ecalClusterShape()[0] <ecalClusterEtaRMS_[0]+ecalClusterEtaRMS_[1]*jet.et()+ecalClusterEtaRMS_[2]*jet.et()*jet.et())
		     if(l2info.ecalClusterShape()[1] <ecalClusterPhiRMS_[0]+ecalClusterPhiRMS_[1]*jet.et()+ecalClusterPhiRMS_[2]*jet.et()*jet.et())
		       if(l2info.ecalClusterShape()[2] <ecalClusterDrRMS_[0]+ecalClusterDrRMS_[1]*jet.et()+ecalClusterDrRMS_[2]*jet.et()*jet.et())
			 if(l2info.hcalIsolEt()<hcalIsolEt_[0]+hcalIsolEt_[1]*jet.et()+hcalIsolEt_[2]*jet.et()*jet.et())
			   if(l2info.nHcalHits() <(int)(nHcalClusters_[0]+nHcalClusters_[1]*jet.et()+nHcalClusters_[2]*jet.et()*jet.et()))
			     if(l2info.hcalClusterShape()[0] <hcalClusterEtaRMS_[0]+hcalClusterEtaRMS_[1]*jet.et()+hcalClusterEtaRMS_[2]*jet.et()*jet.et())
			       if(l2info.hcalClusterShape()[1] <hcalClusterPhiRMS_[0]+hcalClusterPhiRMS_[1]*jet.et()+hcalClusterPhiRMS_[2]*jet.et()*jet.et())
				 if(l2info.hcalClusterShape()[2] <hcalClusterDrRMS_[0]+hcalClusterDrRMS_[1]*jet.et()+hcalClusterDrRMS_[2]*jet.et()*jet.et())
				   {
				     l2IsolCaloJets->push_back(jet);
				   }

	   }
 
        iEvent.put(l2IsolCaloJets, "Isolated");
}

// ------------ method called once each job just before starting event loop  ------------
void 
L2TauModularIsolationSelector::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L2TauModularIsolationSelector::endJob() {
}



