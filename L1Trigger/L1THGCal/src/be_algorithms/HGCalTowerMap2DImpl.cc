///
/// \class HGCalTowerMap2DImpl
///
/// \author: Thomas Strebler
///
/// Description: first iteration of HGCal Tower Maps

#include "FWCore/Utilities/interface/EDMException.h"

#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTowerMap2DImpl.h"



HGCalTowerMap2DImpl::HGCalTowerMap2DImpl( const edm::ParameterSet& conf ) :
  nEtaBins_(conf.getParameter<int>("nEtaBins")),
  nPhiBins_(conf.getParameter<int>("nPhiBins")),
  etaBins_(conf.getParameter<std::vector<double> >("etaBins")),
  phiBins_(conf.getParameter<std::vector<double> >("phiBins")),
  useLayerWeights_(conf.getParameter<bool>("useLayerWeights")),
  layerWeights_(conf.getParameter< std::vector<double> >("layerWeights"))
{
  edm::LogInfo("HGCalTowerMap2DImpl") << "Number of eta bins for the tower maps: " << nEtaBins_<<endl;
  edm::LogInfo("HGCalTowerMap2DImpl") << "Number of phi bins for the tower maps: " << nPhiBins_<<endl;

  if(etaBins_.size()>0 && int(etaBins_.size())!=nEtaBins_+1){
    throw edm::Exception(edm::errors::Configuration, "Configuration")
      << "HGCalTowerMap2DImpl nEtaBins for the tower map not consistent with etaBins size"<<endl;
  }
  if(phiBins_.size()>0 && int(phiBins_.size())!=nPhiBins_+1){
    throw edm::Exception(edm::errors::Configuration, "Configuration")
      << "HGCalTowerMap2DImpl nPhiBins for the tower map not consistent with phiBins size"<<endl;
  }


  std::vector<l1t::HGCalTowerMap> towerMaps = newTowerMaps();
  edm::LogInfo("HGCalTowerMap2DImpl") << "Eta bins for the tower maps: {";
  for(auto eta : towerMaps[0].etaBins()) edm::LogInfo("HGCalTowerMap2DImpl") << eta << ",";
  edm::LogInfo("HGCalTowerMap2DImpl") << "}" <<endl;
  edm::LogInfo("HGCalTowerMap2DImpl") << "Phi bins for the tower maps: {";
  for(auto phi : towerMaps[0].phiBins()) edm::LogInfo("HGCalTowerMap2DImpl") << phi << ",";
  edm::LogInfo("HGCalTowerMap2DImpl") << "}" <<endl;  

}


std::vector<l1t::HGCalTowerMap> HGCalTowerMap2DImpl::newTowerMaps(){

  //If no custom binning specified, assume uniform one
  l1t::HGCalTowerMap map;
  if(etaBins_.size()==0 || phiBins_.size()==0){
    l1t::HGCalTowerMap mapTmp(nEtaBins_,nPhiBins_);
    map = mapTmp;
  }
  else{
    l1t::HGCalTowerMap mapTmp(etaBins_,phiBins_);
    map = mapTmp;
  }

  std::vector<l1t::HGCalTowerMap> towerMaps(max(triggerTools_.lastLayerBH(),unsigned(1)),map); //Always create at least 1 towerMap
  for(unsigned layer=0; layer<triggerTools_.lastLayerBH(); layer++) towerMaps[layer].setLayer(layer+1);
  return towerMaps;

}




void HGCalTowerMap2DImpl::buildTowerMap2D(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>> & triggerCellsPtrs,
					  l1t::HGCalTowerMapBxCollection & towerMaps
				       ){

  std::vector<l1t::HGCalTowerMap> towerMapsTmp = newTowerMaps();

  for( std::vector<edm::Ptr<l1t::HGCalTriggerCell>>::const_iterator tc = triggerCellsPtrs.begin(); tc != triggerCellsPtrs.end(); ++tc ){

    unsigned layer = triggerTools_.layerWithOffset((*tc)->detId());
    int iEta = towerMapsTmp[layer-1].iEta((*tc)->eta());
    int iPhi = towerMapsTmp[layer-1].iPhi((*tc)->phi());

    double calibPt = (*tc)->pt();
    if(useLayerWeights_) calibPt = layerWeights_[layer]*((*tc)->mipPt());
    math::PtEtaPhiMLorentzVector p4(calibPt,
				    (*tc)->eta(),
				    (*tc)->phi(),
				    0. );

    double etEm = layer<=triggerTools_.lastLayerEE() ? calibPt : 0;
    double etHad = layer>triggerTools_.lastLayerEE() ? calibPt : 0;

    l1t::HGCalTower tower;
    tower.setP4(p4);
    tower.setEtEm(etEm);
    tower.setEtHad(etHad);
    tower.setHwEta(iEta);
    tower.setHwPhi(iPhi);

    towerMapsTmp[layer-1].addTower(iEta,iPhi,tower);

  }

  /* store towerMaps in the persistent collection */
  towerMaps.resize(0, triggerTools_.lastLayerBH());
  int i=0;
  for(auto towerMap : towerMapsTmp){
    towerMaps.set( 0, i, towerMap);
    i++;
  }


}



