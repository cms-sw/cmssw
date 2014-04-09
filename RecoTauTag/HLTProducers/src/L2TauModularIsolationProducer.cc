#include "RecoTauTag/HLTProducers/interface/L2TauModularIsolationProducer.h"
#include "RecoTauTag/HLTProducers/interface/L2TauIsolationAlgs.h"
#include "RecoTauTag/HLTProducers/interface/L2TauSimpleClustering.h"


using namespace reco;
using namespace edm;


L2TauModularIsolationProducer::L2TauModularIsolationProducer(const edm::ParameterSet& iConfig):
  l2CaloJets_(consumes<CaloJetCollection>(iConfig.getParameter<edm::InputTag>("L2TauJetCollection"))),
  EBRecHits_(consumes<EBRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHits"))),
  EERecHits_(consumes<EERecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHits"))),
  caloTowers_(consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("CaloTowers"))),
  pfClustersECAL_(consumes<PFClusterCollection>(iConfig.getParameter<edm::InputTag>("pfClustersECAL"))),
  pfClustersHCAL_(consumes<PFClusterCollection>(iConfig.getParameter<edm::InputTag>("pfClustersHCAL"))),
  ecalIsolationAlg_(iConfig.getParameter<std::string>("ecalIsolationAlgorithm")),
  hcalIsolationAlg_(iConfig.getParameter<std::string>("hcalIsolationAlgorithm")),
  ecalClusteringAlg_(iConfig.getParameter<std::string>("ecalClusteringAlgorithm")),
  hcalClusteringAlg_(iConfig.getParameter<std::string>("hcalClusteringAlgorithm")),
  associationRadius_(iConfig.getParameter<double>("associationRadius")),
  simpleClusterRadiusECAL_(iConfig.getParameter<double>("simpleClusterRadiusEcal")),
  simpleClusterRadiusHCAL_(iConfig.getParameter<double>("simpleClusterRadiusHcal")),
  innerConeECAL_(iConfig.getParameter<double>("innerConeECAL")),
  outerConeECAL_(iConfig.getParameter<double>("outerConeECAL")),
  innerConeHCAL_(iConfig.getParameter<double>("innerConeHCAL")),
  outerConeHCAL_(iConfig.getParameter<double>("outerConeHCAL")),
  crystalThresholdE_(iConfig.getParameter<double>("crystalThresholdEE")),
  crystalThresholdB_(iConfig.getParameter<double>("crystalThresholdEB")),
  towerThreshold_(iConfig.getParameter<double>("towerThreshold"))

 {
       
  //Add the products
  produces<L2TauInfoAssociation>();

}


L2TauModularIsolationProducer::~L2TauModularIsolationProducer()
{
  //Destruction

}



void
L2TauModularIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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

	   //Define objects to be loaded from file:
	   math::PtEtaPhiELorentzVectorCollection hitsECAL;
	   math::PtEtaPhiELorentzVectorCollection hitsHCAL;
	   math::PtEtaPhiELorentzVectorCollection pfClustersECAL;
	   math::PtEtaPhiELorentzVectorCollection pfClustersHCAL;

	   if(ecalIsolationAlg_=="recHits" || ecalClusteringAlg_=="recHits" || ecalIsolationAlg_ =="simpleClusters" ||ecalClusteringAlg_ =="simpleClusters")
	     hitsECAL =getECALHits(*jc,iEvent,iSetup); 

	   if(hcalIsolationAlg_=="recHits" || hcalClusteringAlg_=="recHits" || hcalIsolationAlg_ =="simpleClusters" ||hcalClusteringAlg_ =="simpleClusters")
	     hitsHCAL =getHCALHits(*jc,iEvent); 

	   if(ecalIsolationAlg_=="particleFlow" || ecalClusteringAlg_=="particleFlow")
	     pfClustersECAL =getPFClusters(*jc,iEvent,pfClustersECAL_);

	   if(hcalIsolationAlg_=="particleFlow" || hcalClusteringAlg_=="particleFlow")
	     pfClustersHCAL =getPFClusters(*jc,iEvent,pfClustersHCAL_);



	   //Do ECAL Isolation

	   if(ecalIsolationAlg_ == "recHits")
	     {
	       //Use Rechits
       	       info.setEcalIsolEt( alg.isolatedEt(hitsECAL , jc->p4().Vect(), innerConeECAL_,outerConeECAL_) );
	     }
	   else if(ecalIsolationAlg_ == "simpleClusters")
	     {
	       //create the simple clusters
	       L2TauSimpleClustering clustering(simpleClusterRadiusECAL_);
	       math::PtEtaPhiELorentzVectorCollection clusters = clustering.clusterize(hitsECAL);
       	       info.setEcalIsolEt( alg.isolatedEt(clusters , jc->p4().Vect(), innerConeECAL_,outerConeECAL_) );

	     }
	   else if(ecalIsolationAlg_ == "particleFlow")
	     {
	       //Use ParticleFlow
       	       info.setEcalIsolEt( alg.isolatedEt(pfClustersECAL , jc->p4().Vect(), innerConeECAL_,outerConeECAL_) );
	     }

	   //Do ECAL Clustering

	   if(ecalClusteringAlg_ == "recHits")
	     {
	       //Use Rechits
               info.setEcalClusterShape(alg.clusterShape(hitsECAL,jc->p4().Vect(),0.,outerConeECAL_) );
	       info.setNEcalHits( alg.nClustersAnnulus(hitsECAL , jc->p4().Vect(), innerConeECAL_,outerConeECAL_));
	       if(hitsECAL.size()>0)
		 info.setSeedEcalHitEt(hitsECAL[0].pt());

	     }
	   else if(ecalClusteringAlg_ == "simpleClusters")
	     {
	       //create the simple clusters
	       L2TauSimpleClustering clustering(simpleClusterRadiusECAL_);
	       math::PtEtaPhiELorentzVectorCollection clusters = clustering.clusterize(hitsECAL);
               info.setEcalClusterShape(alg.clusterShape(clusters,jc->p4().Vect(),0.,outerConeECAL_) );
	       info.setNEcalHits( alg.nClustersAnnulus(clusters, jc->p4().Vect(), innerConeECAL_,outerConeECAL_));

	       if(clusters.size()>0)
		 info.setSeedEcalHitEt(clusters[0].pt());
	     }
	   else if(ecalClusteringAlg_ == "particleFlow")
	     {
	       //Use ParticleFlow
               info.setEcalClusterShape(alg.clusterShape(pfClustersECAL,jc->p4().Vect(),0.,outerConeECAL_) );
	       info.setNEcalHits( alg.nClustersAnnulus(pfClustersECAL, jc->p4().Vect(), innerConeECAL_,outerConeECAL_));
	       if(pfClustersECAL.size()>0)
		 info.setSeedEcalHitEt(pfClustersECAL[0].pt());
	     }


	   //Do HCAL Isolation

	   if(hcalIsolationAlg_ == "recHits")
	     {
	       //Use Rechits
       	       info.setHcalIsolEt( alg.isolatedEt(hitsHCAL , jc->p4().Vect(), innerConeHCAL_,outerConeHCAL_) );
	     }
	   else if(hcalIsolationAlg_ == "simpleClusters")
	     {
	       //create the simple clusters
	       L2TauSimpleClustering clustering(simpleClusterRadiusHCAL_);
	       math::PtEtaPhiELorentzVectorCollection clusters = clustering.clusterize(hitsHCAL);
       	       info.setHcalIsolEt( alg.isolatedEt(clusters , jc->p4().Vect(), innerConeHCAL_,outerConeHCAL_) );

	     }
	   else if(hcalIsolationAlg_ == "particleFlow")
	     {
	       //Use ParticleFlow
       	       info.setHcalIsolEt( alg.isolatedEt(pfClustersHCAL , jc->p4().Vect(), innerConeHCAL_,outerConeHCAL_) );
	     }



	   //Do HCAL Clustering

	   if(hcalClusteringAlg_ == "recHits")
	     {
	       //Use Rechits
               info.setHcalClusterShape(alg.clusterShape(hitsHCAL,jc->p4().Vect(),0.,outerConeHCAL_) );
	       info.setNHcalHits( alg.nClustersAnnulus(hitsHCAL, jc->p4().Vect(), innerConeHCAL_,outerConeHCAL_));
	       if(hitsHCAL.size()>0)
		 info.setSeedHcalHitEt(hitsHCAL[0].pt());
	     }
	   else if(hcalClusteringAlg_ == "simpleClusters")
	     {
	       //create the simple clusters
	       L2TauSimpleClustering clustering(simpleClusterRadiusHCAL_);
	       math::PtEtaPhiELorentzVectorCollection clusters = clustering.clusterize(hitsHCAL);
               info.setHcalClusterShape(alg.clusterShape(clusters,jc->p4().Vect(),0.,outerConeHCAL_) );
	       info.setNHcalHits( alg.nClustersAnnulus(clusters, jc->p4().Vect(), innerConeHCAL_,outerConeHCAL_));
	       if(clusters.size()>0)
		 info.setSeedHcalHitEt(clusters[0].pt());
	     }
	   else if(hcalClusteringAlg_ == "particleFlow")
	     {
	       //Use ParticleFlow
               info.setHcalClusterShape(alg.clusterShape(pfClustersHCAL,jc->p4().Vect(),0.,outerConeHCAL_) );
	       info.setNHcalHits( alg.nClustersAnnulus(pfClustersHCAL, jc->p4().Vect(), innerConeHCAL_,outerConeHCAL_));
	       if(pfClustersHCAL.size()>0)
		 info.setSeedHcalHitEt(pfClustersHCAL[0].pt());
	     }

	   //Store the info Class
	   edm::Ref<CaloJetCollection> jcRef(l2CaloJets,jc-jcStart);
	   l2InfoAssoc->insert(jcRef, info);
	 }

     } //end of if(*jetCrystalsObj)

    iEvent.put(l2InfoAssoc);
}


void 
L2TauModularIsolationProducer::beginJob()
{
}


void 
L2TauModularIsolationProducer::endJob() {
}




math::PtEtaPhiELorentzVectorCollection 
L2TauModularIsolationProducer::getHCALHits(const CaloJet& jet,const edm::Event& iEvent)
{
  edm::Handle<CaloTowerCollection> towers;

  math::PtEtaPhiELorentzVectorCollection towers2;

  if(iEvent.getByToken(caloTowers_,towers))
    if(towers->size()>0)
    for(size_t i=0;i<towers->size();++i)
      {
	math::PtEtaPhiELorentzVector tower((*towers)[i].et(),(*towers)[i].eta(),(*towers)[i].phi(),(*towers)[i].energy());
	if(ROOT::Math::VectorUtil::DeltaR(tower,jet.p4()) <associationRadius_)
	  {
	    if(tower.pt()>towerThreshold_)
	      towers2.push_back(tower);
	  }	  
      }

  if(towers2.size()>0)
    std::sort(towers2.begin(),towers2.end(),comparePt);
  return towers2;
}


math::PtEtaPhiELorentzVectorCollection 
L2TauModularIsolationProducer::getECALHits(const CaloJet& jet,const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  //Init Geometry
  ESHandle<CaloGeometry> geometry;
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
  if(iEvent.getByToken(EBRecHits_, EBRecHits))
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
	   if(p.pt()>crystalThresholdB_)
	     jetRecHits.push_back(p);
       }

 if(iEvent.getByToken(EERecHits_, EERecHits))
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
	   if(p.pt()>crystalThresholdE_)
	     jetRecHits.push_back(p);
       }

 if(jetRecHits.size()>0)
   std::sort(jetRecHits.begin(),jetRecHits.end(),comparePt);

 return jetRecHits;
}



math::PtEtaPhiELorentzVectorCollection 
L2TauModularIsolationProducer::getPFClusters(const CaloJet& jet,const edm::Event& iEvent,const edm::EDGetTokenT<PFClusterCollection>& input)
{
  edm::Handle<PFClusterCollection> clusters;
  math::PtEtaPhiELorentzVectorCollection clusters2;

  //get Clusters near the jet
  if(iEvent.getByToken(input,clusters))
    if(clusters->size()>0)
    for(PFClusterCollection::const_iterator c = clusters->begin();c!=clusters->end();++c)
    {
      double energy = c->energy();
      double eta = c->eta();
      double phi = c->phi();
      double theta = c->position().theta();
      if(theta > M_PI) theta = 2 * M_PI- theta;
      double et = energy * sin(theta);
      math::PtEtaPhiELorentzVector p(et, eta, phi, energy);

      if(ROOT::Math::VectorUtil::DeltaR(p,jet.p4()) < associationRadius_)
	clusters2.push_back(p);
    }

  if(clusters2.size()>0)
    std::sort(clusters2.begin(),clusters2.end(),comparePt);
  return clusters2;
}

