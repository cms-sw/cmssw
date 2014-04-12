#include "RecoTauTag/HLTProducers/interface/L2TauIsolationProducer.h"
#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "RecoTauTag/HLTProducers/interface/L2TauSimpleClustering.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

using namespace reco;

L2TauIsolationProducer::L2TauIsolationProducer(const edm::ParameterSet& iConfig):
  l2CaloJets_(consumes<CaloJetCollection>(iConfig.getParameter<edm::InputTag>("L2TauJetCollection"))),
  EBRecHits_(consumes<EBRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHits"))),
  EERecHits_(consumes<EERecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHits"))),
  crystalThreshold_(iConfig.getParameter<double>("crystalThreshold")),
  towerThreshold_(iConfig.getParameter<double>("towerThreshold"))
 {
        
  //ECAL Isolation
  edm::ParameterSet ECALIsolParams = iConfig.getParameter<edm::ParameterSet>("ECALIsolation") ;
    
  ECALIsolation_innerCone_ =  ECALIsolParams.getParameter<double>( "innerCone" );
  ECALIsolation_outerCone_ =  ECALIsolParams.getParameter<double>( "outerCone" );
  ECALIsolation_run_       =  ECALIsolParams.getParameter<bool>( "runAlgorithm" );
  

  //ECAL Clustering
  edm::ParameterSet ECALClusterParams = iConfig.getParameter<edm::ParameterSet>("ECALClustering") ;

  ECALClustering_run_                 =  ECALClusterParams.getParameter<bool>( "runAlgorithm" );
  ECALClustering_clusterRadius_       =  ECALClusterParams.getParameter<double>( "clusterRadius" );
    

  //Tower Isolation
  edm::ParameterSet TowerIsolParams = iConfig.getParameter<edm::ParameterSet>("TowerIsolation") ;
  TowerIsolation_innerCone_         =  TowerIsolParams.getParameter<double>( "innerCone" );
  TowerIsolation_outerCone_         =  TowerIsolParams.getParameter<double>( "outerCone" );
  TowerIsolation_run_               =  TowerIsolParams.getParameter<bool>( "runAlgorithm" );
 

  //Add the products
  produces<L2TauInfoAssociation>();

}


L2TauIsolationProducer::~L2TauIsolationProducer()
{
  //Destruction

}



void
L2TauIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


   edm::Handle<CaloJetCollection> l2CaloJets; //Handle to the input (L2TauCaloJets);
   iEvent.getByToken(l2CaloJets_ ,l2CaloJets);//get the handle

   //Create the Association
   std::auto_ptr<L2TauInfoAssociation> l2InfoAssoc( new L2TauInfoAssociation);

   //If the JetCrystalsAssociation exists -> RUN The Producer
   if(l2CaloJets->size()>0)
     {
      CaloJetCollection::const_iterator jcStart = l2CaloJets->begin();

       //Loop on Jets
       for(CaloJetCollection::const_iterator jc = jcStart ;jc!=l2CaloJets->end();++jc)
	 {

	   //Create Algorithm Object
	   L2TauIsolationAlgs alg;
	   
	   //Create Info Object
	   L2TauIsolationInfo info; 

	   //get Hits
	   math::PtEtaPhiELorentzVectorCollection hitsECAL = getECALHits(*jc,iEvent,iSetup);
	   math::PtEtaPhiELorentzVectorCollection hitsHCAL = getHCALHits(*jc);


	   //Run ECALIsolation 
	   if(ECALIsolation_run_)
	     {
	       info.setEcalIsolEt( alg.isolatedEt(hitsECAL , jc->p4().Vect(), ECALIsolation_innerCone_,ECALIsolation_outerCone_) );
	       if(hitsECAL.size()>0)
		 info.setSeedEcalHitEt(hitsECAL[0].pt());
  	     }

	   //Run ECALClustering 
	   if(ECALClustering_run_)
	     {
	       //load simple clustering algorithm
	       L2TauSimpleClustering clustering(ECALClustering_clusterRadius_);
	       math::PtEtaPhiELorentzVectorCollection clusters = clustering.clusterize(hitsECAL);
	       info.setEcalClusterShape(alg.clusterShape(clusters,jc->p4().Vect(),0,0.5) );
	       info.setNEcalHits(clusters.size());
	     }

	   //Run CaloTower Isolation
           if(TowerIsolation_run_)
	     {
	       info.setHcalIsolEt( alg.isolatedEt(hitsHCAL , jc->p4().Vect(), TowerIsolation_innerCone_,TowerIsolation_outerCone_) );
	       if(hitsHCAL.size()>0)
		 info.setSeedHcalHitEt(hitsHCAL[0].pt());
	     }

	   //Store the info Class
	   edm::Ref<CaloJetCollection> jcRef(l2CaloJets,jc-jcStart);
	   l2InfoAssoc->insert(jcRef, info);
	 }

     
     } //end of if(*jetCrystalsObj)

    iEvent.put(l2InfoAssoc);
}


void 
L2TauIsolationProducer::beginJob()
{
}


void 
L2TauIsolationProducer::endJob() {
}



math::PtEtaPhiELorentzVectorCollection 
L2TauIsolationProducer::getHCALHits(const CaloJet& jet)
{


  std::vector<CaloTowerPtr> towers = jet.getCaloConstituents();

  math::PtEtaPhiELorentzVectorCollection towers2;

  for(size_t i=0;i<towers.size();++i)
    if(towers[i]->energy()>towerThreshold_)
      towers2.push_back(math::PtEtaPhiELorentzVector(towers[i]->et(),towers[i]->eta(),towers[i]->phi(),towers[i]->energy()));

  return towers2;
}





math::PtEtaPhiELorentzVectorCollection 
L2TauIsolationProducer::getECALHits(const CaloJet& jet,const edm::Event& iEvent,const edm::EventSetup& iSetup)
{

  //Init Geometry
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<CaloGeometryRecord>().get(geometry);

  //Create ECAL Geometry
  const CaloSubdetectorGeometry* EB = geometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorGeometry* EE = geometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);

  //Handle To the ECAL
  edm::Handle<EBRecHitCollection> EBRecHits;
  edm::Handle<EERecHitCollection> EERecHits;

  //Read From File
 
  iEvent.getByToken( EBRecHits_, EBRecHits );
  iEvent.getByToken( EERecHits_, EERecHits );
  
 

  //Create a container for the hits
  math::PtEtaPhiELorentzVectorCollection jetRecHits;


  //Create the Machinery to put the highest et crystal at the begining
  double ref_pt = 0.;//max et
  int ref_id=0; //id of the max element
  int count=0; //counter

  //Code Copied from JetCrystalsAssociator    
      const std::vector<CaloTowerPtr>  myTowers=jet.getCaloConstituents();
        for (unsigned int iTower = 0; iTower < myTowers.size(); iTower++)
	{
	  CaloTowerPtr theTower = myTowers[iTower];
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
		    // std::cout <<"Et "<<et<<std::endl;
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
     



