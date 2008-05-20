#include "RecoTauTag/HLTProducers/interface/L2TauIsolationProducer.h"
#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"



L2TauIsolationProducer::L2TauIsolationProducer(const edm::ParameterSet& iConfig):
  l2CaloJets_(iConfig.getParameter<edm::InputTag>("L2TauJetCollection")),
  EBRecHits_(iConfig.getParameter<edm::InputTag>("EBRecHits")),
  EERecHits_(iConfig.getParameter<edm::InputTag>("EERecHits")),
  crystalThreshold_(iConfig.getParameter<double>("crystalThreshold")),
  towerThreshold_(iConfig.getParameter<double>("towerThreshold"))
 {
        
  //ECAL Isolation
  edm::ParameterSet ECALIsolParams = iConfig.getParameter<edm::ParameterSet>("ECALIsolation") ;
    
  ECALIsolation_innerCone_ =  ECALIsolParams.getParameter<double>( "innerCone" );
  ECALIsolation_outerCone_ =  ECALIsolParams.getParameter<double>( "outerCone" );
  ECALIsolation_run_    =  ECALIsolParams.getParameter<bool>( "runAlgorithm" );
  

  //ECAL Clustering
  edm::ParameterSet ECALClusterParams = iConfig.getParameter<edm::ParameterSet>("ECALClustering") ;
      
  ECALClustering_clusterRadius_ =  ECALClusterParams.getParameter<double>( "clusterRadius" );
    
  //Tower Isolation

  edm::ParameterSet TowerIsolParams = iConfig.getParameter<edm::ParameterSet>("TowerIsolation") ;
      
  TowerIsolation_innerCone_ =  TowerIsolParams.getParameter<double>( "innerCone" );
  TowerIsolation_outerCone_ =  TowerIsolParams.getParameter<double>( "outerCone" );
  TowerIsolation_run_ =  TowerIsolParams.getParameter<bool>( "runAlgorithm" );
 

  //Add the products
  produces<L2TauInfoAssociation>( "L2TauIsolationInfoAssociator" );

}


L2TauIsolationProducer::~L2TauIsolationProducer()
{
  //Destruction

}



void
L2TauIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


   Handle<CaloJetCollection> l2CaloJets; //Handle to the input (L2TauCaloJets);



   iEvent.getByLabel(l2CaloJets_ ,l2CaloJets);//get the handle


   //If the JetCrystalsAssociation exists -> RUN The Producer
   if(&(*l2CaloJets))
     {


       //Create the Association
       std::auto_ptr<L2TauInfoAssociation> l2InfoAssoc( new L2TauInfoAssociation);
     
       CaloJetCollection::const_iterator jcStart = l2CaloJets->begin();


       //Loop on Jets
       for(CaloJetCollection::const_iterator jc = jcStart ;jc!=l2CaloJets->end();++jc)
	 {

	   L2TauIsolationInfo l2info; //Create Info Object

	   //Run ECALIsolation 
	   if( (ECALIsolation_run_))
	     {
	       L2TauECALIsolation ecal_isolation(ECALIsolation_innerCone_,ECALIsolation_outerCone_);
	       ecal_isolation.run(getECALHits(*jc,iEvent,iSetup),*jc,l2info);
	     }


	   //Run ECALClustering 
	   if(ECALClustering_run_)
	     {
	       L2TauECALClustering ecal_clustering(ECALClustering_clusterRadius_);
	       ecal_clustering.run(getECALHits(*jc,iEvent,iSetup),*jc,l2info);
	     }


	   //Run CaloTower Isolation
           if(TowerIsolation_run_)
	     {
	       L2TauTowerIsolation tower_isolation(towerThreshold_,TowerIsolation_innerCone_,TowerIsolation_outerCone_);
	       tower_isolation.run(*jc,l2info);

	     }
	  

      

	   //Store the info Class
	   edm::Ref<CaloJetCollection> jcRef(l2CaloJets,jc-jcStart);
	   l2InfoAssoc->insert(jcRef, l2info);


	         
	 }


       //Store The staff in the event

       iEvent.put(l2InfoAssoc, "L2TauIsolationInfoAssociator");
     
     } //end of if(*jetCrystalsObj)



}


void 
L2TauIsolationProducer::beginJob(const edm::EventSetup&)
{
}


void 
L2TauIsolationProducer::endJob() {
}


math::PtEtaPhiELorentzVectorCollection 
L2TauIsolationProducer::getECALHits(const CaloJet& jet,const edm::Event& iEvent,const edm::EventSetup& iSetup)
{

  //Init Geometry
  ESHandle<CaloGeometry> geometry;
  iSetup.get<CaloGeometryRecord>().get(geometry);

  //Create ECAL Geometry
  const CaloSubdetectorGeometry* EB = geometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorGeometry* EE = geometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);

  //Handle To the ECAL
  Handle<EBRecHitCollection> EBRecHits;
  Handle<EERecHitCollection> EERecHits;

  //Read From File
 
  iEvent.getByLabel( EBRecHits_, EBRecHits );
  iEvent.getByLabel( EERecHits_, EERecHits );
  
 

  //Create a container for the hits
  math::PtEtaPhiELorentzVectorCollection jetRecHits;


  //Create the Machinery to put the highest et crystal at the begining
  double ref_pt = 0.;//max et
  int ref_id=0; //id of the max element
  int count=0; //counter

  //Code Copied from JetCrystalsAssociator    
      const std::vector<CaloTowerRef>  myTowers=jet.getConstituents();
        for (unsigned int iTower = 0; iTower < myTowers.size(); iTower++)
	{
	  CaloTowerRef theTower = myTowers[iTower];
	  size_t numRecHits = theTower->constituentsSize();
	  // access CaloRecHits
	  for (size_t j = 0; j < numRecHits; j++) {
	    DetId RecHitDetID=theTower->constituent(j);
	    DetId::Detector DetNum=RecHitDetID.det();
	    if( DetNum == DetId::Ecal ){
	      int EcalNum =  RecHitDetID.subdetId();
	      if( EcalNum == 1 ){
		EBDetId EcalID = RecHitDetID;
		EBRecHitCollection::const_iterator theRecHit=EBRecHits->find(EcalID);
		if(theRecHit != EBRecHits->end()){
		  DetId id = theRecHit->detid();
		  const CaloCellGeometry* this_cell = EB->getGeometry(id);
		  if (this_cell) {
		    GlobalPoint posi = this_cell->getPosition();
		    double energy = theRecHit->energy();
		    double eta = posi.eta();
		    double phi = posi.phi();
		    double theta = posi.theta();
		    if(theta > M_PI) theta = 2 * M_PI- theta;
		    double et = energy * sin(theta);
		    //Apply Thresholds Here
		    math::PtEtaPhiELorentzVector p(et, eta, phi, energy);
		     if(p.pt()>crystalThreshold_)
		      { 
			if(p.pt()>ref_pt)
			 {
			   ref_id=count;
			   ref_pt = p.pt();
			 }
			jetRecHits.push_back(p);
			count++;
		      }
		    
		  }
		}
	      } else if ( EcalNum == 2 ) {
		EEDetId EcalID = RecHitDetID;
		EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);    
		if(theRecHit != EBRecHits->end()){
		  DetId id = theRecHit->detid();
		  const CaloCellGeometry* this_cell = EE->getGeometry(id);
		  if (this_cell) {
		    GlobalPoint posi = this_cell->getPosition();
		    double energy = theRecHit->energy();
		    double eta = posi.eta();
		    double phi = posi.phi();
		    double theta = posi.theta();
		    if (theta > M_PI) theta = 2 * M_PI - theta;
		    double et = energy * sin(theta);
		    // cout <<"Et "<<et<<endl;
		    math::PtEtaPhiELorentzVector p(et, eta, phi, energy);
	    	     if(p.pt()>crystalThreshold_)
		      { 
			if(p.pt()>ref_pt)
			 {
			   ref_id=count;
			   ref_pt = p.pt();
			 }
			jetRecHits.push_back(p);
			count++;
		      }
		  }
		}
	      }
	    }
	  }
	}

	//bring it to the front
	if(jetRecHits.size()>0)
	  std::swap(jetRecHits[ref_id],jetRecHits[0]);

	return jetRecHits;
}
     



