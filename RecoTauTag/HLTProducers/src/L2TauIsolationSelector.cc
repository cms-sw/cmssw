
#include "RecoTauTag/HLTProducers/interface/L2TauIsolationSelector.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"

using namespace reco; 

L2TauIsolationSelector::L2TauIsolationSelector(const edm::ParameterSet& iConfig):
  associationInput_(iConfig.getParameter<edm::InputTag>("L2InfoAssociation")),
  ECALIsolEt_(iConfig.getParameter<double>("ECALIsolEt")),
  TowerIsolEt_(iConfig.getParameter<double>("TowerIsolEt")),
  Cluster_etaRMS_(iConfig.getParameter<double>("ClusterEtaRMS")),
  Cluster_phiRMS_(iConfig.getParameter<double>("ClusterPhiRMS")),
  Cluster_drRMS_(iConfig.getParameter<double>("ClusterDRRMS")),
  Cluster_nClusters_(iConfig.getParameter<int>("ClusterNClusters")),
  JetEt_(iConfig.getParameter<double>("MinJetEt")),
  SeedTowerEt_(iConfig.getParameter<double>("SeedTowerEt"))

{




  produces<CaloJetCollection>("Isolated");
}


L2TauIsolationSelector::~L2TauIsolationSelector()
{
 

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L2TauIsolationSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<L2TauInfoAssociation> Imap;
   

   iEvent.getByLabel(associationInput_ ,Imap);//get the handle

     if(&(*Imap)) 
       {
	 //Create the CaloJet Collection
	
	 std::auto_ptr<CaloJetCollection> l2IsolCaloJets( new CaloJetCollection );
	 l2IsolCaloJets->reserve(Imap->size());

	 for(L2TauInfoAssociation::const_iterator p = Imap->begin();p!=Imap->end();++p)
	   {
	     //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	     const L2TauIsolationInfo l2info = p->val;
	     //Retrieve the Jet
	     const CaloJet& jet =*(p->key);
	     
	
	     
	     //If The Cuts are Satisfied
	   if(jet.et()>JetEt_) 
	     if(l2info.ECALIsolConeCut< ECALIsolEt_)
	       if(l2info.SeedTowerEt>SeedTowerEt_)
	        if(l2info.ECALClusterNClusters <Cluster_nClusters_)
		   if(l2info.ECALClusterNClusters <Cluster_nClusters_)
		     if(l2info.ECALClusterEtaRMS <Cluster_etaRMS_)
		       if(l2info.ECALClusterPhiRMS <Cluster_phiRMS_)
			 if(l2info.ECALClusterDRRMS <Cluster_drRMS_)
			   if(l2info.TowerIsolConeCut<TowerIsolEt_)
			     {
			         //Retrieve the Jet From the AssociationMap

	   		       l2IsolCaloJets->push_back(*(jet.clone()));
			     }

	   }
 
	        iEvent.put(l2IsolCaloJets, "Isolated");

       }



}

// ------------ method called once each job just before starting event loop  ------------
void 
L2TauIsolationSelector::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L2TauIsolationSelector::endJob() {
}



