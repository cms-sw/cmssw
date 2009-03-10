#include "RecoTauTag/HLTProducers/interface/L2TauNarrowConeIsolationProducer.h"
#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


L2TauNarrowConeIsolationProducer::L2TauNarrowConeIsolationProducer(const edm::ParameterSet& iConfig):
  l2CaloJets_(iConfig.getParameter<edm::InputTag>("L2TauJetCollection")),
  EBRecHits_(iConfig.getParameter<edm::InputTag>("EBRecHits")),
  EERecHits_(iConfig.getParameter<edm::InputTag>("EERecHits")),
  CaloTowers_(iConfig.getParameter<edm::InputTag>("CaloTowers")),
  associationRadius_(iConfig.getParameter<double>("associationRadius")),
  crystalThresholdE_(iConfig.getParameter<double>("crystalThresholdEE")),
  crystalThresholdB_(iConfig.getParameter<double>("crystalThresholdEB")),
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


L2TauNarrowConeIsolationProducer::~L2TauNarrowConeIsolationProducer()
{
  //Destruction

}



void
L2TauNarrowConeIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


   Handle<CaloJetCollection> l2CaloJets; //Handle to the input (L2TauCaloJets);
   iEvent.getByLabel(l2CaloJets_ ,l2CaloJets);//get the handle

   //Create the Association
   std::auto_ptr<L2TauInfoAssociation> l2InfoAssoc( new L2TauInfoAssociation);

   //If the JetCrystalsAssociation exists -> RUN The Producer
   if(l2CaloJets->size()>0)
     {
      CaloJetCollection::const_iterator jcStart = l2CaloJets->begin();
       //Loop on Jets
       for(CaloJetCollection::const_iterator jc = jcStart ;jc!=l2CaloJets->end();++jc)
	 {
	   L2TauIsolationInfo l2info; //Create Info Object

	   //Run ECALIsolation 
	   if(ECALIsolation_run_)
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
	       L2TauTowerIsolation tower_isolation(TowerIsolation_innerCone_,TowerIsolation_outerCone_);
	       tower_isolation.run(*jc,getHCALHits(*jc,iEvent),l2info);

	     }

	   //Store the info Class
	   edm::Ref<CaloJetCollection> jcRef(l2CaloJets,jc-jcStart);
	   l2InfoAssoc->insert(jcRef, l2info);
	 }


     
     } //end of if(*jetCrystalsObj)

    iEvent.put(l2InfoAssoc);
}


void 
L2TauNarrowConeIsolationProducer::beginJob(const edm::EventSetup&)
{
}


void 
L2TauNarrowConeIsolationProducer::endJob() {
}



math::PtEtaPhiELorentzVectorCollection 
L2TauNarrowConeIsolationProducer::getHCALHits(const CaloJet& jet,const edm::Event& iEvent)
{
  edm::Handle<CaloTowerCollection> towers;
  math::PtEtaPhiELorentzVectorCollection towers2;

  if(iEvent.getByLabel(CaloTowers_,towers))
    for(size_t i=0;i<towers->size();++i)
      {
	math::PtEtaPhiELorentzVector tower((*towers)[i].et(),(*towers)[i].eta(),(*towers)[i].phi(),(*towers)[i].energy());
	if(ROOT::Math::VectorUtil::DeltaR(tower,jet.p4()) <associationRadius_)
	  {
	    if(tower.energy()>towerThreshold_)
	      towers2.push_back(tower);

	  }	  

      }

 std::sort(towers2.begin(),towers2.end(),comparePt);

  return towers2;
}


math::PtEtaPhiELorentzVectorCollection 
L2TauNarrowConeIsolationProducer::getECALHits(const CaloJet& jet,const edm::Event& iEvent,const edm::EventSetup& iSetup)
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

  //Create a container for the hits
  math::PtEtaPhiELorentzVectorCollection jetRecHits;

  //Loop on the barrel hits
  if(iEvent.getByLabel( EBRecHits_, EBRecHits))
     for(EBRecHitCollection::const_iterator hit = EBRecHits->begin();hit!=EBRecHits->end();++hit)
       {
	 //get Detector Geometry
	 const CaloCellGeometry* this_cell = EB->getGeometry(hit->detid());
	 GlobalPoint posi = this_cell->getPosition();
	 double energy = hit->energy();
	 double eta = posi.eta();
	 double phi = posi.phi();
	 double theta = posi.theta();
	 if(theta > M_PI) theta = 2 * M_PI- theta;
	 double et = energy * sin(theta);
	 math::PtEtaPhiELorentzVector p(et, eta, phi, energy);
	 if(ROOT::Math::VectorUtil::DeltaR(p,jet.p4()) <associationRadius_)
	   if(p.energy()>crystalThresholdB_)
	     jetRecHits.push_back(p);
       }

 if(iEvent.getByLabel( EERecHits_, EERecHits))
     for(EERecHitCollection::const_iterator hit = EERecHits->begin();hit!=EERecHits->end();++hit)
       {
	 //get Detector Geometry
	 const CaloCellGeometry* this_cell = EE->getGeometry(hit->detid());
	 GlobalPoint posi = this_cell->getPosition();
	 double energy = hit->energy();
	 double eta = posi.eta();
	 double phi = posi.phi();
	 double theta = posi.theta();
	 if(theta > M_PI) theta = 2 * M_PI- theta;
	 double et = energy * sin(theta);
	 math::PtEtaPhiELorentzVector p(et, eta, phi, energy);
	 if(ROOT::Math::VectorUtil::DeltaR(p,jet.p4()) < associationRadius_)
	   if(p.energy()>crystalThresholdE_)
	     jetRecHits.push_back(p);
       }


 std::sort(jetRecHits.begin(),jetRecHits.end(),comparePt);

 return jetRecHits;
}



