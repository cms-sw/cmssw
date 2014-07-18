///
/// \class l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2EGammaAlgorithmFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"



l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::Stage2Layer2EGammaAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{

}


l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::~Stage2Layer2EGammaAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster> & clusters,
							      const std::vector<l1t::CaloTower>& towers,
                    std::vector<l1t::EGamma> & egammas) {

  egammas.clear();
  for(size_t clusNr=0;clusNr<clusters.size();clusNr++){
    // Keep only actual clusters
    if(clusters[clusNr].isValid()){ 

      egammas.push_back(clusters[clusNr]);

      // Identification part 
      bool hOverEBit = idHOverE(clusters[clusNr]);
      bool shapeBit = idShape(clusters[clusNr]);
      bool fgBit = !(clusters[clusNr].hwSeedPt()>6 && clusters[clusNr].fgECAL()); 
      int qual = 0;
      if(fgBit) qual |= (0x1); // first bit = FG
      if(hOverEBit) qual |= (0x1<<1); // second bit = H/E
      if(shapeBit) qual |= (0x1<<2); // third bit = shape
      egammas.back().setHwQual( qual ); 


      // Isolation part
      int hwEtSum = CaloTools::calHwEtSum(clusters[clusNr].hwEta(),clusters[clusNr].hwPhi(),towers,
          -1*params_->egIsoAreaNrTowersEta(),params_->egIsoAreaNrTowersEta(),
          -1*params_->egIsoAreaNrTowersPhi(),params_->egIsoAreaNrTowersPhi(),
          params_->egIsoMaxEtaAbsForIsoSum());
      int hwFootPrint = isoCalEgHwFootPrint(clusters[clusNr],towers);

      int nrTowers = CaloTools::calNrTowers(-1*params_->egIsoMaxEtaAbsForTowerSum(),
          params_->egIsoMaxEtaAbsForTowerSum(),
          1,72,towers,1,999,CaloTools::CALO);
      unsigned int lutAddress = isoLutIndex(egammas.back().hwEta(),nrTowers);

      int isolBit = hwEtSum-hwFootPrint <= params_->egIsolationLUT()->data(lutAddress); 
      // std::cout <<"hwEtSum "<<hwEtSum<<" hwFootPrint "<<hwFootPrint<<" isol "<<hwEtSum-hwFootPrint<<" bit "<<isolBit<<" area "<<params_->egIsoAreaNrTowersEta()<<" "<<params_->egIsoAreaNrTowersPhi()<< " veto "<<params_->egIsoVetoNrTowersPhi()<<std::endl;

      egammas.back().setHwIso(isolBit);
      //  egammas.back().setHwIso(hwEtSum-hwFootPrint); //naughtly little debug hack, shouldnt be in release, comment out if it is

      // calibration part
      int calibPt = calibratedPt(clusters[clusNr]);

      // physical eta/phi
      double eta = 0.;
      double phi = 0.;
      double seedEta     = CaloTools::towerEta(clusters[clusNr].hwEta());
      double seedEtaSize = CaloTools::towerEtaSize(clusters[clusNr].hwEta());
      double seedPhi     = CaloTools::towerPhi(clusters[clusNr].hwEta(), clusters[clusNr].hwPhi());
      double seedPhiSize = CaloTools::towerPhiSize(clusters[clusNr].hwEta());
      if(clusters[clusNr].fgEta()==0)      eta = seedEta; // center
      else if(clusters[clusNr].fgEta()==2) eta = seedEta + seedEtaSize*0.25; // center + 1/4
      else if(clusters[clusNr].fgEta()==1) eta = seedEta - seedEtaSize*0.25; // center - 1/4
      if(clusters[clusNr].fgPhi()==0)      phi = seedPhi; // center
      else if(clusters[clusNr].fgPhi()==2) phi = seedPhi + seedPhiSize*0.25; // center + 1/4
      else if(clusters[clusNr].fgPhi()==1) phi = seedPhi - seedPhiSize*0.25; // center - 1/4

      // Set 4-vector
      math::PtEtaPhiMLorentzVector calibP4((double)calibPt*params_->egLsb(), eta, phi, 0.);
      egammas.back().setP4(calibP4);

    }//end of cuts on cluster to make EGamma
  }//end of cluster loop
}

bool l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::idHOverE(const l1t::CaloCluster& clus)
{
  unsigned int lutAddress = idHOverELutIndex(clus.hwEta()); 
  bool hOverEBit = ( clus.hOverE() <= params_->egMaxHOverELUT()->data(lutAddress) );
  hOverEBit |= ( clus.hwPt()>=floor(params_->egEtToRemoveHECut()/params_->egLsb()) );
  return hOverEBit;
}

unsigned int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::idHOverELutIndex(int iEta)
{
  unsigned int iEtaNormed = abs(iEta);
  return iEtaNormed-1;
}

bool l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::idShape(const l1t::CaloCluster& clus)
{
  unsigned int shape = 0;
  if( !(clus.checkClusterFlag(CaloCluster::TRIM_N)) ) shape |= (0x1);
  if( !(clus.checkClusterFlag(CaloCluster::TRIM_S)) ) shape |= (0x1<<1);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && !(clus.checkClusterFlag(CaloCluster::TRIM_NE)) ) shape |= (0x1<<2);
  if( clus.checkClusterFlag(CaloCluster::TRIM_RIGHT) && !(clus.checkClusterFlag(CaloCluster::TRIM_NW)) ) shape |= (0x1<<2);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && !(clus.checkClusterFlag(CaloCluster::TRIM_E))  ) shape |= (0x1<<3);
  if( clus.checkClusterFlag(CaloCluster::TRIM_RIGHT) && !(clus.checkClusterFlag(CaloCluster::TRIM_W))  ) shape |= (0x1<<3);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && !(clus.checkClusterFlag(CaloCluster::TRIM_SE)) ) shape |= (0x1<<4);
  if( clus.checkClusterFlag(CaloCluster::TRIM_RIGHT) && !(clus.checkClusterFlag(CaloCluster::TRIM_SW)) ) shape |= (0x1<<4);
  if( clus.checkClusterFlag(CaloCluster::EXT_UP)   ) shape |= (0x1<<5);
  if( clus.checkClusterFlag(CaloCluster::EXT_DOWN) ) shape |= (0x1<<6);

  unsigned int lutAddress = idShapeLutIndex(shape, clus.hwEta()); 
  bool shapeBit = params_->egShapeIdLUT()->data(lutAddress);
  return shapeBit;
}

unsigned int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::idShapeLutIndex(unsigned int shape, int iEta)
{
  unsigned int iEtaNormed = abs(iEta)-1;
  if(iEtaNormed>31) iEtaNormed = 31;
  if(shape>127) shape = 127;
  unsigned int index = iEtaNormed*128+shape;
  return index;
}

//calculates the footprint of the electron in hardware values
int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::isoCalEgHwFootPrint(const l1t::CaloCluster& clus,const std::vector<l1t::CaloTower>& towers) 
{
  int iEta=clus.hwEta();
  int iPhi=clus.hwPhi();

  // hwEmEtSumLeft =  CaloTools::calHwEtSum(iEta,iPhi,towers,-1,-1,-1,1,CaloTools::ECAL);
  // int hwEmEtSumRight = CaloTools::calHwEtSum(iEta,iPhi,towers,1,1,-1,1,CaloTools::ECAL);
  
  int etaSide = clus.checkClusterFlag(CaloCluster::TRIM_LEFT) ? 1 : -1; //if we trimed left, its the right (ie +ve) side we want
  int phiSide = iEta>0 ? 1 : -1;
  
  int ecalHwFootPrint = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,
					      -1*params_->egIsoVetoNrTowersPhi(),params_->egIsoVetoNrTowersPhi(),
					      params_->egIsoMaxEtaAbsForIsoSum(),CaloTools::ECAL) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,etaSide,etaSide,
			  -1*params_->egIsoVetoNrTowersPhi(),params_->egIsoVetoNrTowersPhi(),
			  params_->egIsoMaxEtaAbsForIsoSum(),CaloTools::ECAL);
  int hcalHwFootPrint = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,0,0,params_->egIsoMaxEtaAbsForIsoSum(),CaloTools::HCAL) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,phiSide,phiSide,params_->egIsoMaxEtaAbsForIsoSum(),CaloTools::HCAL);
  return ecalHwFootPrint+hcalHwFootPrint;
}

//ieta =-28, nrTowers 0 is 0, increases to ieta28, nrTowers=kNrTowersInSum
unsigned l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::isoLutIndex(int iEta,unsigned int nrTowers)
{
  const unsigned int kNrTowersInSum=72*params_->egIsoMaxEtaAbsForTowerSum()*2;
  const unsigned int kTowerGranularity=params_->egIsoPUEstTowerGranularity();
  const unsigned int kMaxAddress = kNrTowersInSum%kTowerGranularity==0 ? (kNrTowersInSum/kTowerGranularity+1)*28*2 : 
                                                                         (kNrTowersInSum/kTowerGranularity)*28*2;
  
  unsigned int nrTowersNormed = nrTowers/kTowerGranularity;
  
  unsigned int iEtaNormed = iEta+28;
  if(iEta>0) iEtaNormed--; //we skip zero
  
  if(std::abs(iEta)>28 || iEta==0 || nrTowers>kNrTowersInSum) return kMaxAddress;
  else return iEtaNormed*(kNrTowersInSum/kTowerGranularity+1)+nrTowersNormed;
  
}

int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::calibratedPt(const l1t::CaloCluster& clus)
{
  unsigned int lutAddress = calibrationLutIndex(clus.hwEta()); 
  int corr = params_->egCalibrationLUT()->data(lutAddress); // 9 bits. [0,1]. corrPt = (1+corr)*rawPt
  // the correction can only increase the energy, and it cannot increase it more than a factor two
  int rawPt = clus.hwPt();
  int corrXrawPt = corr*rawPt;// 17 bits
  // round corr*rawPt
  int addPt = corrXrawPt>>9;// 8 MS bits (truncation)
  int remainder = corrXrawPt & 0x1FF; // 9 LS bits
  if(remainder>=0x100) addPt += 1;
  int corrPt = rawPt + addPt;
  if(corrPt>255) corrPt = 255;// 8 bits threshold
  return corrPt;
}

unsigned int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::calibrationLutIndex(int iEta)
{
  unsigned int iEtaNormed = abs(iEta);
  return iEtaNormed-1;
}
