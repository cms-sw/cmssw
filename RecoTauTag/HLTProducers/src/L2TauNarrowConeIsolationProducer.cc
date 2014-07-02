#include "RecoTauTag/HLTProducers/interface/L2TauNarrowConeIsolationProducer.h"
#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "RecoTauTag/HLTProducers/interface/L2TauSimpleClustering.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

using namespace reco;

L2TauNarrowConeIsolationProducer::L2TauNarrowConeIsolationProducer(const edm::ParameterSet& iConfig):
  l2CaloJets_(consumes<CaloJetCollection>(iConfig.getParameter<edm::InputTag>("L2TauJetCollection"))),
  EBRecHits_(consumes<EBRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHits"))),
  EERecHits_(consumes<EERecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHits"))),
  CaloTowers_(consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("CaloTowers"))),
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
	   math::PtEtaPhiELorentzVectorCollection hitsHCAL = getHCALHits(*jc,iEvent);



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
L2TauNarrowConeIsolationProducer::beginJob()
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

  if(iEvent.getByToken(CaloTowers_,towers))
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
  edm::ESHandle<CaloGeometry> geometry;
  iSetup.get<CaloGeometryRecord>().get(geometry);

  //Create ECAL Geometry
  const CaloSubdetectorGeometry* EB = geometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorGeometry* EE = geometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);

  //Handle To the ECAL
  edm::Handle<EBRecHitCollection> EBRecHits;
  edm::Handle<EERecHitCollection> EERecHits;

  //Create a container for the hits
  math::PtEtaPhiELorentzVectorCollection jetRecHits;

  //Loop on the barrel hits
  if(iEvent.getByToken( EBRecHits_, EBRecHits))
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

 if(iEvent.getByToken( EERecHits_, EERecHits))
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



