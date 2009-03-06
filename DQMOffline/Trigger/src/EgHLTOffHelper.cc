#include "DQMOffline/Trigger/interface/EgHLTOffHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

void EgHLTOffHelper::setup(const edm::ParameterSet& conf)
{
  // barrelShapeAssocProd_ = conf.getParameter<edm::InputTag>("barrelClusterShapeAssociation");
  //barrelShapeAssocProd_ = conf.getParameter<edm::InputTag>("barrelClusterShapeAssociation");
  //endcapShapeAssocProd_ = conf.getParameter<edm::InputTag>("endcapClusterShapeAssociation");


  ecalRecHitsEBTag_ = conf.getParameter<edm::InputTag>("BarrelRecHitCollection");
  ecalRecHitsEETag_ = conf.getParameter<edm::InputTag>("EndcapRecHitCollection");
  
  caloJetsTag_ = conf.getParameter<edm::InputTag>("CaloJetCollection");
 

  cuts_.setHighNrgy();
  tagCuts_.setHighNrgy();
  probeCuts_.setPreSel();
}


//this function coverts GsfElectrons to a format which is actually useful to me
void EgHLTOffHelper::fillEgHLTOffEleVec(edm::Handle<reco::GsfElectronCollection> gsfElectrons,std::vector<EgHLTOffEle>& egHLTOffEles)
{
  egHLTOffEles.clear();
  egHLTOffEles.reserve(gsfElectrons->size());
  for(reco::PixelMatchGsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin(); gsfIter!=gsfElectrons->end();++gsfIter){
    //for now use dummy isolation data
    EgHLTOffEle::IsolData isolData;
    isolData.nrTrks=1;
    isolData.ptTrks=1.;
    isolData.em= 0.42;
    isolData.had=0.42;
    
    //get cluster shape and we're done construction 
    //edm::LogInfo("EgHLTOffHelper") << "getting clus shape "<<std::endl;
    //const reco::ClusterShape* clusShape = getClusterShape(&*gsfIter);
    //edm::LogInfo("EgHLTOffHelper") << "clus shape "<<clusShape<<std::endl;
    EgHLTOffEle::ClusShapeData clusShapeData;
    clusShapeData.sigmaEtaEta=999.;
    //    clusShapeData.sigmaIEtaIEta=999.;
    //clusShapeData.e2x5MaxOver5x5=-1.; //not defined in endcap yet
    //need to figure out if its in the barrel or endcap
    //classification variable is unrelyable so get the first hit of the cluster and figure out if its barrel or endcap
    const reco::BasicCluster& seedClus = *(gsfIter->superCluster()->seed());
    const DetId seedDetId = seedClus.getHitsByDetId()[0]; //note this may not actually be the seed hit but it doesnt matter because all hits will be in the barrel OR endcap (it is also incredably inefficient as it getHitsByDetId passes the vector by value not reference
    if(seedDetId.subdetId()==EcalBarrel){
      std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,ebRecHits_,caloTopology_,caloGeom_);
      // std::vector<float> crysCov = EcalClusterTools::crystalCovariances(seedClus,ebRecHits_,caloTopology_,caloGeom_);
      clusShapeData.sigmaEtaEta = sqrt(stdCov[0]);
      //  clusShapeData.sigmaIEtaIEta =  sqrt(crysCov[0]);
      clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
      //clusShapeData.sigmaIPhiIPhi =  sqrt(crysCov[2]);
       
      
     
    }else{
     std::vector<float> stdCov = EcalClusterTools::covariances(seedClus,eeRecHits_,caloTopology_,caloGeom_);
     clusShapeData.sigmaEtaEta = sqrt(stdCov[0]);  
     clusShapeData.sigmaPhiPhi = sqrt(stdCov[2]);
     
    }

    egHLTOffEles.push_back(EgHLTOffEle(*gsfIter,clusShapeData,isolData));
    
    //now we would like to set the cut results
    EgHLTOffEle& ele =  egHLTOffEles.back();
    ele.setTagCutCode(tagCuts_.getCutCode(ele));
    ele.setProbeCutCode(probeCuts_.getCutCode(ele));
    ele.setCutCode(cuts_.getCutCode(ele));
      

  }//end loop over gsf electron collection

}

void EgHLTOffHelper::getHandles(const edm::Event& event,const edm::EventSetup& setup)
{
  // try{
//     event.getByLabel(barrelShapeAssocProd_, clusterShapeHandleBarrel_) ;
//   }catch(...){} //the worlds most pointless try, catch pair, damn you CMSSW framework,  DAMN you
  
//   try{
//     event.getByLabel(endcapShapeAssocProd_, clusterShapeHandleEndcap_) ;
//   }catch(...){}

//   if (!clusterShapeHandleBarrel_.isValid()) {
//     edm::LogError ("EgHLTOffHelper") << "Can't get ECAL barrel Cluster Shape Collection" ; 
//   }
//   if (!clusterShapeHandleEndcap_.isValid()) {
//     edm::LogError ("EgHLTOffHelper") << "Can't get ECAL endcap Cluster Shape Collection" ; 
//   }

  //yay, now in 2_1 we dont have to program by exception
  edm::Handle<EcalRecHitCollection> ecalBarrelRecHitsHandle;
  event.getByLabel(ecalRecHitsEBTag_,ecalBarrelRecHitsHandle);
  ebRecHits_ = ecalBarrelRecHitsHandle.product();

  edm::Handle<EcalRecHitCollection> ecalEndcapRecHitsHandle;
  event.getByLabel(ecalRecHitsEETag_,ecalEndcapRecHitsHandle);
  eeRecHits_ = ecalEndcapRecHitsHandle.product();

  edm::Handle<reco::CaloJetCollection> caloJetsHandle;
  event.getByLabel(caloJetsTag_,caloJetsHandle);
  jets_ = caloJetsHandle.product();
  
  edm::ESHandle<CaloGeometry> geomHandle;
  setup.get<CaloGeometryRecord>().get(geomHandle);
  caloGeom_ = geomHandle.product();

  edm::ESHandle<CaloTopology> topologyHandle;
  setup.get<CaloTopologyRecord>().get(topologyHandle);
  caloTopology_ = topologyHandle.product();
}



// //ripped of from the electronIDAlgo (there must be a better way, I *cannot* believe that there isnt a better way)
// //I've made some minor mods for speed and robustness (it could still be faster though)
// //I'm sorry for the pain you are about to go through
// //in summary it determines where the electron is barrel or endcap and if the clusterShape association map handle is valid for it
// //it then looks in the map for the electrons seed cluster and if found, returns a pointer to the shape
// //a complication arrises as electrons which are endcap may be classified as in the barrel-endcap gap and therefore have classification 40
// //and therefore be labeled barrel someplaces (like here) and endcap others
// const reco::ClusterShape* EgHLTOffHelper::getClusterShape(const reco::GsfElectron* electron)
// {
//   // Find entry in map corresponding to seed BasicCluster of SuperCluster
//   reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr;
 
//   if ( electron->classification() < 100 && clusterShapeHandleBarrel_.isValid() ) {
//     const reco::BasicClusterShapeAssociationCollection& barrelClShpMap = *clusterShapeHandleBarrel_;
//     reco::SuperClusterRef sclusRef = electron->get<reco::SuperClusterRef> () ;
//     seedShpItr = barrelClShpMap.find ( sclusRef->seed () ) ;
//     if (seedShpItr!=barrelClShpMap.end()) return &*(seedShpItr->val);
//     else if (clusterShapeHandleEndcap_.isValid()){
//       const reco::BasicClusterShapeAssociationCollection& endcapClShpMap = *clusterShapeHandleEndcap_;
//       seedShpItr = endcapClShpMap.find ( sclusRef->seed ()) ;
//       if(seedShpItr!=endcapClShpMap.end()) return &*(seedShpItr->val);
//     }//end check of valid endcap cluster shape in barrel section
//   } else if(electron->classification()>=100 && clusterShapeHandleEndcap_.isValid()) {
//     const reco::BasicClusterShapeAssociationCollection& endcapClShpMap = *clusterShapeHandleEndcap_;
//     reco::SuperClusterRef sclusRef = electron->get<reco::SuperClusterRef> () ;
//     seedShpItr = endcapClShpMap.find ( sclusRef->seed () ) ;
//     if(seedShpItr!=endcapClShpMap.end()) return &*(seedShpItr->val); 
//   }//end check of endcap electron with valid shape map
  
//   return NULL;
// }

