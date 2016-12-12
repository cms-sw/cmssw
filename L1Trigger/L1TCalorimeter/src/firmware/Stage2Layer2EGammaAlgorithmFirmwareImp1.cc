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
#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"
#include "L1Trigger/L1TCalorimeter/interface/AccumulatingSort.h"


namespace l1t {
  bool operator > ( const l1t::EGamma& a, const l1t::EGamma& b )
  {
      return a.pt() > b.pt();
  }
}


/*****************************************************************/
l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::Stage2Layer2EGammaAlgorithmFirmwareImp1(CaloParamsHelper* params) :
  params_(params)
/*****************************************************************/
{

}

/*****************************************************************/
l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::~Stage2Layer2EGammaAlgorithmFirmwareImp1() 
/*****************************************************************/
{
}

/*****************************************************************/
void l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster>& clusters, const std::vector<l1t::CaloTower>& towers, std::vector<l1t::EGamma>& egammas) 
/*****************************************************************/
{
  l1t::CaloStage2Nav caloNav;
  egammas.clear();

  //EGammas without check of FG and shape ID
  std::vector<l1t::EGamma> egammas_raw; 

  for(const auto& cluster : clusters)
  {
    // Keep only valid clusters
    if(cluster.isValid())
    { 
      // need tower energies to recompute egamma trimmed energy
      int iEta = cluster.hwEta();
      int iPhi = cluster.hwPhi();
      int iEtaP  = caloNav.offsetIEta(iEta,  1);
      int iEtaM  = caloNav.offsetIEta(iEta, -1);
      int iPhiP  = caloNav.offsetIPhi(iPhi,  1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi,  2);
      int iPhiM  = caloNav.offsetIPhi(iPhi, -1);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      const l1t::CaloTower& seed    = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEta) , iPhi );
      const l1t::CaloTower& towerNW = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEtaM), iPhiM);
      const l1t::CaloTower& towerN  = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEta) , iPhiM);
      const l1t::CaloTower& towerNE = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEtaP), iPhiM);
      const l1t::CaloTower& towerE  = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEtaP), iPhi );
      const l1t::CaloTower& towerSE = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEtaP), iPhiP);
      const l1t::CaloTower& towerS  = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEta) , iPhiP);
      const l1t::CaloTower& towerSW = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEtaM), iPhiP);
      const l1t::CaloTower& towerW  = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEtaM), iPhi );
      const l1t::CaloTower& towerNN = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEta) , iPhiM2);
      const l1t::CaloTower& towerSS = l1t::CaloTools::getTower(towers, CaloTools::caloEta(iEta) , iPhiP2);
      //

      int seedEt    = seed   .hwPt();
      int towerEtNW = towerNW.hwPt();
      int towerEtN  = towerN .hwPt();
      int towerEtNE = towerNE.hwPt();
      int towerEtE  = towerE .hwPt();
      int towerEtSE = towerSE.hwPt();
      int towerEtS  = towerS .hwPt();
      int towerEtSW = towerSW.hwPt();
      int towerEtW  = towerW .hwPt();
      int towerEtNN = towerNN.hwPt();
      int towerEtSS = towerSS.hwPt();

      if(abs(iEta)> 28 )
	continue;

      // initialize egamma from cluster
      egammas_raw.push_back(cluster);
      l1t::EGamma& egamma = egammas_raw.back();     

      // Trim cluster (only for egamma energy computation, the original cluster is unchanged)
      l1t::CaloCluster clusterTrim = trimCluster(cluster);

      // Recompute hw energy (of the trimmed cluster) from towers
      egamma.setHwPt(seedEt);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NW)) egamma.setHwPt(egamma.hwPt() + towerEtNW);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_N))  egamma.setHwPt(egamma.hwPt() + towerEtN);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NE)) egamma.setHwPt(egamma.hwPt() + towerEtNE);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_E))  egamma.setHwPt(egamma.hwPt() + towerEtE);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SE)) egamma.setHwPt(egamma.hwPt() + towerEtSE);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_S))  egamma.setHwPt(egamma.hwPt() + towerEtS);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SW)) egamma.setHwPt(egamma.hwPt() + towerEtSW);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_W))  egamma.setHwPt(egamma.hwPt() + towerEtW);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NN)) egamma.setHwPt(egamma.hwPt() + towerEtNN);
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SS)) egamma.setHwPt(egamma.hwPt() + towerEtSS);


      // Identification of the egamma
      // Based on the seed tower FG bit, the H/E ratio of the seed tower, and the shape of the cluster
      bool hOverEBit = cluster.hOverE()>0;
      bool shapeBit  = idShape(cluster, egamma.hwPt());
      bool fgBit     = !(cluster.fgECAL()); 
      int qual = 0;
      if(fgBit)     qual |= (0x1); // first bit = FG
      if(hOverEBit) qual |= (0x1<<1); // second bit = H/E
      if(shapeBit)  qual |= (0x1<<2); // third bit = shape
      egamma.setHwQual( qual ); 


      // Isolation 
      int isoLeftExtension = params_->egIsoAreaNrTowersEta();
      int isoRightExtension = params_->egIsoAreaNrTowersEta();

      if(cluster.checkClusterFlag(CaloCluster::TRIM_LEFT))
	isoRightExtension++;
      else
	isoLeftExtension++;

      int hwEtSum = CaloTools::calHwEtSum(cluster.hwEta(), cluster.hwPhi(), towers,
					  -isoLeftExtension,isoRightExtension,
					  -1*params_->egIsoAreaNrTowersPhi(),params_->egIsoAreaNrTowersPhi(),
					  params_->egPUSParam(2));

      int hwFootPrint = isoCalEgHwFootPrint(cluster,towers);

      int nrTowers = CaloTools::calNrTowers(-1*params_->egPUSParam(1),
          params_->egPUSParam(1),
          1,72,towers,1,999,CaloTools::CALO);
      unsigned int lutAddress = isoLutIndex(egamma.hwEta(), nrTowers, egamma.hwPt());

      int isolBit = (((hwEtSum-hwFootPrint) < params_->egIsolationLUT()->data(lutAddress)) || (params_->egIsolationLUT()->data(lutAddress)>255));       
      egamma.setHwIso(isolBit);
      int hwIsoEnergy = hwEtSum-hwFootPrint;

      // development vars
      egamma.setTowerIPhi((short int)CaloTools::towerEta(cluster.hwEta()));
      egamma.setTowerIEta((short int)CaloTools::towerPhi(cluster.hwEta(), cluster.hwPhi()));
      egamma.setRawEt((short int)egamma.hwPt());
      egamma.setIsoEt((short int)hwIsoEnergy);
      egamma.setFootprintEt((short int)hwFootPrint);
      egamma.setNTT((short int)nrTowers);
      egamma.setShape((short int)returnShape(cluster));
      
      // Energy calibration
      // Corrections function of ieta, ET, and cluster shape
      int calibPt = calibratedPt(cluster, egamma.hwPt());
      egamma.setHwPt(calibPt);

      // Physical eta/phi. Computed from ieta/iphi of the seed tower and the fine-grain position within the seed
      double eta = 0.;
      double phi = 0.;
      double seedEta     = CaloTools::towerEta(cluster.hwEta());
      double seedEtaSize = CaloTools::towerEtaSize(cluster.hwEta());
      double seedPhi     = CaloTools::towerPhi(cluster.hwEta(), cluster.hwPhi());
      double seedPhiSize = CaloTools::towerPhiSize(cluster.hwEta());
      if(cluster.fgEta()==0)      eta = seedEta; // center
      //Test
      else if(cluster.fgEta()==2) eta = seedEta + seedEtaSize*0.251; // center + 1/4
      else if(cluster.fgEta()==1) eta = seedEta - seedEtaSize*0.251; // center - 1/4


      //fgPhi is recomputed after trimming
      int fgPhi = 0;
      
      int EtUp   = 0;
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NE)) EtUp += towerEtNE;
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_N))  EtUp += towerEtN;
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NW)) EtUp += towerEtNW;
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NN)) EtUp += towerEtNN;
      int EtDown = 0;
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SE)) EtDown += towerEtSE;
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_S))  EtDown += towerEtS;
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SW)) EtDown += towerEtSW;
      if(clusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SS)) EtDown += towerEtSS;
      //
      if(EtDown>EtUp) fgPhi = 2;
      else if(EtUp>EtDown) fgPhi = 1;


      if(fgPhi==0)      phi = seedPhi; // center
      else if(fgPhi==2) phi = seedPhi + seedPhiSize*0.251; // center + 1/4
      else if(fgPhi==1) phi = seedPhi - seedPhiSize*0.251; // center - 1/4
      

      // Set 4-vector
      math::PtEtaPhiMLorentzVector calibP4((double)calibPt*params_->egLsb(), eta, phi, 0.);
      egamma.setP4(calibP4);

    }//end of cuts on cluster to make EGamma
  }//end of cluster loop

  // prepare content to be sorted -- each phi ring contains 18 elements, with Et = 0 if no candidate exists
  math::PtEtaPhiMLorentzVector emptyP4;
  l1t::EGamma tempEG (emptyP4, 0, 0, 0, 0);
  std::vector< std::vector<l1t::EGamma> > egEtaPos( 28 , std::vector<l1t::EGamma>(18, tempEG));
  std::vector< std::vector<l1t::EGamma> > egEtaNeg( 28 , std::vector<l1t::EGamma>(18, tempEG));
  for (unsigned int iEG = 0; iEG < egammas_raw.size(); iEG++)
  {
      int fgBit     = egammas_raw.at(iEG).hwQual()    & (0x1);
      int hOverEBit = egammas_raw.at(iEG).hwQual()>>1 & (0x1);
      int shapeBit  = egammas_raw.at(iEG).hwQual()>>2 & (0x1);

      bool IDcuts = (fgBit && hOverEBit && shapeBit) || (egammas_raw.at(iEG).pt()>=params_->egMaxPtHOverE()) || (params_->egBypassEGVetos());

      if(!IDcuts) continue;

      if (egammas_raw.at(iEG).hwEta() > 0) egEtaPos.at( egammas_raw.at(iEG).hwEta()-1).at((egammas_raw.at(iEG).hwPhi()-1)/4) = egammas_raw.at(iEG);
      else                                 egEtaNeg.at( -(egammas_raw.at(iEG).hwEta()+1)).at((egammas_raw.at(iEG).hwPhi()-1)/4) = egammas_raw.at(iEG);
  }

  AccumulatingSort <l1t::EGamma> etaPosSorter(6);
  AccumulatingSort <l1t::EGamma> etaNegSorter(6);
  std::vector<l1t::EGamma> accumEtaPos;
  std::vector<l1t::EGamma> accumEtaNeg;

  for( int ieta = 0 ; ieta < 28 ; ++ieta)
  {
      // eta +
      std::vector<l1t::EGamma>::iterator start_, end_;
      start_ = egEtaPos.at(ieta).begin();  
      end_   = egEtaPos.at(ieta).end();
      BitonicSort<l1t::EGamma>(down, start_, end_);
      etaPosSorter.Merge( egEtaPos.at(ieta) , accumEtaPos );
      
      // eta -
      start_ = egEtaNeg.at(ieta).begin();  
      end_   = egEtaNeg.at(ieta).end();
      BitonicSort<l1t::EGamma>(down, start_, end_);
      etaNegSorter.Merge( egEtaNeg.at(ieta) , accumEtaNeg );

  }

  // put all 12 candidates in the original tau vector, removing zero energy ones
  egammas.clear();
  for (l1t::EGamma acceg : accumEtaPos)
  {
      if (acceg.hwPt() > 0) egammas.push_back(acceg);
  }
  for (l1t::EGamma acceg : accumEtaNeg)
  {
      if (acceg.hwPt() > 0) egammas.push_back(acceg);
  }

}


/*****************************************************************/
bool l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::idShape(const l1t::CaloCluster& clus, int hwPt)
/*****************************************************************/
{
  unsigned int shape = 0;
  if( (clus.checkClusterFlag(CaloCluster::INCLUDE_N)) ) shape |= (0x1);
  if( (clus.checkClusterFlag(CaloCluster::INCLUDE_S)) ) shape |= (0x1<<1);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_E))  ) shape |= (0x1<<2);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_W))  ) shape |= (0x1<<2);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_NE)) ) shape |= (0x1<<3);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_NW)) ) shape |= (0x1<<3);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_SE)) ) shape |= (0x1<<4);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_SW)) ) shape |= (0x1<<4);
  if( clus.checkClusterFlag(CaloCluster::INCLUDE_NN) ) shape |= (0x1<<5);
  if( clus.checkClusterFlag(CaloCluster::INCLUDE_SS) ) shape |= (0x1<<6);

  unsigned int lutAddress = idShapeLutIndex(clus.hwEta(), hwPt, shape); 
  bool shapeBit = params_->egShapeIdLUT()->data(lutAddress);
  return shapeBit;
}

/*****************************************************************/
unsigned int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::idShapeLutIndex(int iEta, int E, int shape)
/*****************************************************************/
{
  if(params_->egShapeIdType()=="compressed")
  {
    unsigned int iEtaNormed = abs(iEta);
    if(iEtaNormed>28) iEtaNormed = 28;
    if(E>255) E = 255;
    unsigned int compressedShape = params_->egCompressShapesLUT()->data(shape);
    unsigned int compressedE     = params_->egCompressShapesLUT()->data((0x1<<7)+E);
    unsigned int compressedEta   = params_->egCompressShapesLUT()->data((0x1<<7)+(0x1<<8)+iEtaNormed);
    return (compressedShape | compressedE | compressedEta);
  }
  else // Uncompressed (kept for backward compatibility)
  {
    unsigned int iEtaNormed = abs(iEta);
    if(iEtaNormed>28) iEtaNormed = 28;
    if(E>255) E = 255;
    unsigned int compressedShape = params_->egCompressShapesLUT()->data(shape);
    return E+compressedShape*256+(iEtaNormed-1)*256*64;
  }
}

//calculates the footprint of the electron in hardware values
/*****************************************************************/
int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::isoCalEgHwFootPrint(const l1t::CaloCluster& clus,const std::vector<l1t::CaloTower>& towers) 
/*****************************************************************/
{
  int iEta=clus.hwEta();
  int iPhi=clus.hwPhi();

  // hwEmEtSumLeft =  CaloTools::calHwEtSum(iEta,iPhi,towers,-1,-1,-1,1,CaloTools::ECAL);
  // int hwEmEtSumRight = CaloTools::calHwEtSum(iEta,iPhi,towers,1,1,-1,1,CaloTools::ECAL);

  int etaSide = clus.checkClusterFlag(CaloCluster::TRIM_LEFT) ? 1 : -1; //if we trimed left, its the right (ie +ve) side we want
  int phiSide = iEta>0 ? 1 : -1;

  int ecalHwFootPrint = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,
      -1*params_->egIsoVetoNrTowersPhi(),params_->egIsoVetoNrTowersPhi(),
      params_->egPUSParam(2),CaloTools::ECAL) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,etaSide,etaSide,
        -1*params_->egIsoVetoNrTowersPhi(),params_->egIsoVetoNrTowersPhi(),
        params_->egPUSParam(2),CaloTools::ECAL);
  
  //Because of compression E+H can be different from E + H
  int ecalHwFootPrint_2x1 = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,0,0,params_->egPUSParam(2),CaloTools::ECAL) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,phiSide,phiSide,params_->egPUSParam(2),CaloTools::ECAL);

  int ecalhcal_HwFootPrint_2x1 = CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,0,0,params_->egPUSParam(2)) +
    CaloTools::calHwEtSum(iEta,iPhi,towers,0,0,phiSide,phiSide,params_->egPUSParam(2));

  return ecalHwFootPrint-ecalHwFootPrint_2x1+ecalhcal_HwFootPrint_2x1;
  
}

//ieta =-28, nrTowers 0 is 0, increases to ieta28, nrTowers=kNrTowersInSum
/*****************************************************************/
unsigned l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::isoLutIndex(int iEta,unsigned int nrTowers,int E)
/*****************************************************************/
{

  if(params_->egIsolationType()=="compressed")
    {
      if(nrTowers>255) nrTowers = 255;
      unsigned int iEtaNormed = abs(iEta);
      if(iEtaNormed>28) iEtaNormed = 28;
      if(E>255) E = 255;
      unsigned int compressednTT = params_->egCompressShapesLUT()->data((0x1<<7)+(0x1<<8)+(0x1<<5)+nrTowers);
      unsigned int compressedE     = params_->egCompressShapesLUT()->data((0x1<<7)+E)<<1;
      unsigned int compressedEta   = params_->egCompressShapesLUT()->data((0x1<<7)+(0x1<<8)+iEtaNormed)<<1;

      return (compressednTT | compressedE | compressedEta);
    }
  
  else // Uncompressed (kept for backward compatibility)
    {  
      const unsigned int kNrTowersInSum=72*params_->egPUSParam(1)*2;
      const unsigned int kTowerGranularity=params_->egPUSParam(0);
      const unsigned int kMaxAddress = kNrTowersInSum%kTowerGranularity==0 ? (kNrTowersInSum/kTowerGranularity+1)*28*2 : 
	(kNrTowersInSum/kTowerGranularity)*28*2;
      
      unsigned int nrTowersNormed = nrTowers/kTowerGranularity;
      
      unsigned int iEtaNormed = iEta+28;
      if(iEta>0) iEtaNormed--; //we skip zero
      
      if(std::abs(iEta)>28 || iEta==0 || nrTowers>kNrTowersInSum) return kMaxAddress;
      else return iEtaNormed*(kNrTowersInSum/kTowerGranularity+1)+nrTowersNormed;
    }

}

/*****************************************************************/
int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::calibratedPt(const l1t::CaloCluster& clus, int hwPt)
/*****************************************************************/
{
  unsigned int shape = 0;
  if( (clus.checkClusterFlag(CaloCluster::INCLUDE_N)) ) shape |= (0x1);
  if( (clus.checkClusterFlag(CaloCluster::INCLUDE_S)) ) shape |= (0x1<<1);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_E))  ) shape |= (0x1<<2);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_W))  ) shape |= (0x1<<2);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_NE)) ) shape |= (0x1<<3);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_NW)) ) shape |= (0x1<<3);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_SE)) ) shape |= (0x1<<4);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_SW)) ) shape |= (0x1<<4);
  if( clus.checkClusterFlag(CaloCluster::INCLUDE_NN) ) shape |= (0x1<<5);
  if( clus.checkClusterFlag(CaloCluster::INCLUDE_SS) ) shape |= (0x1<<6);

  unsigned int lutAddress = calibrationLutIndex(clus.hwEta(), hwPt, shape); 
  int corr = params_->egCalibrationLUT()->data(lutAddress); // 9 bits. [0,2]. corrPt = (corr)*rawPt
  // the correction can increase or decrease the energy
  int rawPt = hwPt;
  int corrXrawPt = corr*rawPt;// 17 bits
  // round corr*rawPt
  int corrPt = corrXrawPt>>8;// 8 MS bits (truncation)

  //12 bits saturation
  if(corrPt>4095)
    corrPt = 4095;

  return corrPt;
}

/*****************************************************************/
unsigned int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::calibrationLutIndex(int iEta, int E, int shape)
/*****************************************************************/
{
  if(params_->egCalibrationType()=="compressed")
  {
    unsigned int iEtaNormed = abs(iEta);
    if(iEtaNormed>28) iEtaNormed = 28;
    if(E>255) E = 255;
    unsigned int compressedShape = params_->egCompressShapesLUT()->data(shape);
    unsigned int compressedE     = params_->egCompressShapesLUT()->data((0x1<<7)+E);
    unsigned int compressedEta   = params_->egCompressShapesLUT()->data((0x1<<7)+(0x1<<8)+iEtaNormed);
    return (compressedShape | compressedE | compressedEta);
  }
  else // Uncompressed (kept for backward compatibility)
  {
    unsigned int iEtaNormed = abs(iEta);
    if(iEtaNormed>28) iEtaNormed = 28;
    if(E>255) E = 255;
    if(E<22) E = 22;
    unsigned int compressedShape = params_->egCompressShapesLUT()->data(shape);
    if(compressedShape>31) compressedShape = 31;
    return (E-20)+compressedShape*236+(iEtaNormed-1)*236*32;
  }
}

/*****************************************************************/
l1t::CaloCluster l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::trimCluster(const l1t::CaloCluster& clus)
/*****************************************************************/
{
  l1t::CaloCluster clusCopy = clus;

  unsigned int shape = 0;
  if( (clus.checkClusterFlag(CaloCluster::INCLUDE_N)) ) shape |= (0x1);
  if( (clus.checkClusterFlag(CaloCluster::INCLUDE_S)) ) shape |= (0x1<<1);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_E))  ) shape |= (0x1<<2);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_W))  ) shape |= (0x1<<2);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_NE)) ) shape |= (0x1<<3);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_NW)) ) shape |= (0x1<<3);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_SE)) ) shape |= (0x1<<4);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_SW)) ) shape |= (0x1<<4);
  if( clus.checkClusterFlag(CaloCluster::INCLUDE_NN) ) shape |= (0x1<<5);
  if( clus.checkClusterFlag(CaloCluster::INCLUDE_SS) ) shape |= (0x1<<6);

  unsigned int lutAddress = trimmingLutIndex(shape, clus.hwEta()); 
  unsigned int shapeTrim = params_->egTrimmingLUT()->data(lutAddress);
  // apply trimming flags
  clusCopy.setClusterFlag(CaloCluster::INCLUDE_N,  ( shapeTrim&(0x1) )    ? true : false); 
  clusCopy.setClusterFlag(CaloCluster::INCLUDE_S,  ( shapeTrim&(0x1<<1) ) ? true : false);
  clusCopy.setClusterFlag(CaloCluster::INCLUDE_NN, ( shapeTrim&(0x1<<5) ) ? true : false);
  clusCopy.setClusterFlag(CaloCluster::INCLUDE_SS, ( shapeTrim&(0x1<<6) ) ? true : false);
  if( clusCopy.checkClusterFlag(CaloCluster::TRIM_LEFT) )
  {
    clusCopy.setClusterFlag(CaloCluster::INCLUDE_E,  ( shapeTrim&(0x1<<2) ) ? true : false);
    clusCopy.setClusterFlag(CaloCluster::INCLUDE_NE, ( shapeTrim&(0x1<<3) ) ? true : false);
    clusCopy.setClusterFlag(CaloCluster::INCLUDE_SE, ( shapeTrim&(0x1<<4) ) ? true : false);
  }
  else
  {
    clusCopy.setClusterFlag(CaloCluster::INCLUDE_W,  ( shapeTrim&(0x1<<2) ) ? true : false);
    clusCopy.setClusterFlag(CaloCluster::INCLUDE_NW, ( shapeTrim&(0x1<<3) ) ? true : false);
    clusCopy.setClusterFlag(CaloCluster::INCLUDE_SW, ( shapeTrim&(0x1<<4) ) ? true : false);
  }
  return clusCopy;
}

/*****************************************************************/
unsigned int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::trimmingLutIndex(unsigned int shape, int iEta)
/*****************************************************************/
{
  unsigned int iEtaNormed = abs(iEta)-1;
  if(iEtaNormed>31) iEtaNormed = 31;
  if(shape>127) shape = 127;
  unsigned int index = iEtaNormed*128+shape;
  return index;
}

/*****************************************************************/
unsigned int l1t::Stage2Layer2EGammaAlgorithmFirmwareImp1::returnShape(const l1t::CaloCluster& clus)
/*****************************************************************/
{
  l1t::CaloCluster clusCopy = clus;

  unsigned int shape = 0;
  if( (clus.checkClusterFlag(CaloCluster::INCLUDE_N)) ) shape |= (0x1);
  if( (clus.checkClusterFlag(CaloCluster::INCLUDE_S)) ) shape |= (0x1<<1);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_E))  ) shape |= (0x1<<2);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_W))  ) shape |= (0x1<<2);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_NE)) ) shape |= (0x1<<3);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_NW)) ) shape |= (0x1<<3);
  if( clus.checkClusterFlag(CaloCluster::TRIM_LEFT)  && (clus.checkClusterFlag(CaloCluster::INCLUDE_SE)) ) shape |= (0x1<<4);
  if( !clus.checkClusterFlag(CaloCluster::TRIM_LEFT) && (clus.checkClusterFlag(CaloCluster::INCLUDE_SW)) ) shape |= (0x1<<4);
  if( clus.checkClusterFlag(CaloCluster::INCLUDE_NN) ) shape |= (0x1<<5);
  if( clus.checkClusterFlag(CaloCluster::INCLUDE_SS) ) shape |= (0x1<<6);

  return shape;
}
 
