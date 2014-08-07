///
/// \class l1t::Stage2Layer2TauAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2TauAlgorithmFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"


l1t::Stage2Layer2TauAlgorithmFirmwareImp1::Stage2Layer2TauAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{

  loadCalibrationLuts();
}


l1t::Stage2Layer2TauAlgorithmFirmwareImp1::~Stage2Layer2TauAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster> & clusters,
							      std::vector<l1t::Tau> & taus) {

  merging(clusters, taus);

}


// FIXME: to be organized better
void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::merging(const std::vector<l1t::CaloCluster>& clusters, std::vector<l1t::Tau>& taus){
  //std::cout<<"---------------   NEW EVENT -----------------------------\n";
  //std::cout<<"---------------------------------------------------------\n";
  // navigator
  l1t::CaloStage2Nav caloNav; 

  // Temp copy of clusters (needed to set merging flags)
  std::vector<l1t::CaloCluster> tmpClusters(clusters);
  // First loop: setting merging flags
  for ( auto itr = tmpClusters.begin(); itr != tmpClusters.end(); ++itr ) {
    if( itr->isValid() ){
      l1t::CaloCluster& mainCluster = *itr;
      int iEta = mainCluster.hwEta();
      int iPhi = mainCluster.hwPhi();
      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);


      //std::cout<<"Merging: first pass. Cluster eta="<<mainCluster.hwEta()<<", phi="<<mainCluster.hwPhi()<<", E="<<mainCluster.hwPt()<<"\n";

      const l1t::CaloCluster& clusterN2  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiM2);
      const l1t::CaloCluster& clusterN3  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiM3);
      const l1t::CaloCluster& clusterN2W = l1t::CaloTools::getCluster(tmpClusters, iEtaM, iPhiM2);
      const l1t::CaloCluster& clusterN2E = l1t::CaloTools::getCluster(tmpClusters, iEtaP, iPhiM2);
      const l1t::CaloCluster& clusterS2  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiP2);
      const l1t::CaloCluster& clusterS3  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiP3);
      const l1t::CaloCluster& clusterS2W = l1t::CaloTools::getCluster(tmpClusters, iEtaM, iPhiP2);
      const l1t::CaloCluster& clusterS2E = l1t::CaloTools::getCluster(tmpClusters, iEtaP, iPhiP2);

      std::list<l1t::CaloCluster> satellites;
      if(clusterN2 .isValid()) satellites.push_back(clusterN2);
      if(clusterN3 .isValid()) satellites.push_back(clusterN3);
      if(clusterN2W.isValid()) satellites.push_back(clusterN2W);
      if(clusterN2E.isValid()) satellites.push_back(clusterN2E);
      if(clusterS2 .isValid()) satellites.push_back(clusterS2);
      if(clusterS3 .isValid()) satellites.push_back(clusterS3);
      if(clusterS2W.isValid()) satellites.push_back(clusterS2W);
      if(clusterS2E.isValid()) satellites.push_back(clusterS2E);

      if(satellites.size()>0) {
        satellites.sort();
        const l1t::CaloCluster& secondaryCluster = satellites.back();

        if(secondaryCluster>mainCluster) {
          // is secondary
          mainCluster.setClusterFlag(CaloCluster::IS_SECONDARY, true);
          // to be merged up or down?
          if(secondaryCluster.hwPhi()==iPhiP2 || secondaryCluster.hwPhi()==iPhiP3) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_UPDOWN, true); // 1 (down)
          }
          else if(secondaryCluster.hwPhi()==iPhiM2 || secondaryCluster.hwPhi()==iPhiM3) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_UPDOWN, false); // 0 (up)
          }
          // to be merged left or right?
          if(secondaryCluster.hwEta()==iEtaP) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_LEFTRIGHT, true); // 1 (right)
          }
          else if(secondaryCluster.hwEta()==iEta || secondaryCluster.hwEta()==iEtaM) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_LEFTRIGHT, false); // 0 (left)
          }
        }
        else {
          // is main cluster
          mainCluster.setClusterFlag(CaloCluster::IS_SECONDARY, false);
          // to be merged up or down?
          if(secondaryCluster.hwPhi()==iPhiP2 || secondaryCluster.hwPhi()==iPhiP3) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_UPDOWN, true); // 1 (down)
          }
          else if(secondaryCluster.hwPhi()==iPhiM2 || secondaryCluster.hwPhi()==iPhiM3) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_UPDOWN, false); // 0 (up)
          }
          // to be merged left or right?
          if(secondaryCluster.hwEta()==iEtaP) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_LEFTRIGHT, true); // 1 (right)
          }
          else if(secondaryCluster.hwEta()==iEta || secondaryCluster.hwEta()==iEtaM) {
            mainCluster.setClusterFlag(CaloCluster::MERGE_LEFTRIGHT, false); // 0 (left)
          }
        }
        //std::cout<<"   Max neighbor cluster eta="<<secondaryCluster.hwEta()<<", phi="<<secondaryCluster.hwPhi()<<", E="<<secondaryCluster.hwPt()<<"\n";
        //std::cout<<"   Flags: sec="<<mainCluster.checkClusterFlag(CaloCluster::IS_SECONDARY);
        //std::cout<<", ud="<<mainCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN);
        //std::cout<<", lr="<<mainCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)<<"\n";
      }
    }
  }
  // Second loop: do the actual merging based on merging flags
  for ( auto itr = tmpClusters.begin(); itr != tmpClusters.end(); ++itr ) {
    if( itr->isValid() ){
      l1t::CaloCluster& mainCluster = *itr;
      int iEta = mainCluster.hwEta();
      int iPhi = mainCluster.hwPhi();

      // physical eta/phi
      double eta = 0.;
      double phi = 0.;
      double seedEta     = CaloTools::towerEta(mainCluster.hwEta());
      double seedEtaSize = CaloTools::towerEtaSize(mainCluster.hwEta());
      double seedPhi     = CaloTools::towerPhi(mainCluster.hwEta(), mainCluster.hwPhi());
      double seedPhiSize = CaloTools::towerPhiSize(mainCluster.hwEta());
      if     (mainCluster.fgEta()==0) eta = seedEta; // center
      else if(mainCluster.fgEta()==2) eta = seedEta + seedEtaSize*0.25; // center + 1/4
      else if(mainCluster.fgEta()==1) eta = seedEta - seedEtaSize*0.25; // center - 1/4
      if     (mainCluster.fgPhi()==0) phi = seedPhi; // center
      else if(mainCluster.fgPhi()==2) phi = seedPhi + seedPhiSize*0.25; // center + 1/4
      else if(mainCluster.fgPhi()==1) phi = seedPhi - seedPhiSize*0.25; // center - 1/4


      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);

      //std::cout<<"Merging: second pass. Cluster eta="<<mainCluster.hwEta()<<", phi="<<mainCluster.hwPhi()<<", E="<<mainCluster.hwPt()<<"\n";

      const l1t::CaloCluster& clusterN2  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiM2);
      const l1t::CaloCluster& clusterN3  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiM3);
      const l1t::CaloCluster& clusterN2W = l1t::CaloTools::getCluster(tmpClusters, iEtaM, iPhiM2);
      const l1t::CaloCluster& clusterN2E = l1t::CaloTools::getCluster(tmpClusters, iEtaP, iPhiM2);
      const l1t::CaloCluster& clusterS2  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiP2);
      const l1t::CaloCluster& clusterS3  = l1t::CaloTools::getCluster(tmpClusters, iEta, iPhiP3);
      const l1t::CaloCluster& clusterS2W = l1t::CaloTools::getCluster(tmpClusters, iEtaM, iPhiP2);
      const l1t::CaloCluster& clusterS2E = l1t::CaloTools::getCluster(tmpClusters, iEtaP, iPhiP2);

      std::list<l1t::CaloCluster> satellites;
      if(clusterN2 .isValid()) satellites.push_back(clusterN2);
      if(clusterN3 .isValid()) satellites.push_back(clusterN3);
      if(clusterN2W.isValid()) satellites.push_back(clusterN2W);
      if(clusterN2E.isValid()) satellites.push_back(clusterN2E);
      if(clusterS2 .isValid()) satellites.push_back(clusterS2);
      if(clusterS3 .isValid()) satellites.push_back(clusterS3);
      if(clusterS2W.isValid()) satellites.push_back(clusterS2W);
      if(clusterS2E.isValid()) satellites.push_back(clusterS2E);

      // neighbour exists
      if(satellites.size()>0) {
        satellites.sort();
        const l1t::CaloCluster& secondaryCluster = satellites.back();
        // this is the most energetic cluster
        // merge with the secondary cluster if it is not merged to an other one
        if(mainCluster>secondaryCluster) {
          bool canBeMerged = true;
          bool mergeUp = (secondaryCluster.hwPhi()==iPhiM2 || secondaryCluster.hwPhi()==iPhiM3);
          bool mergeLeft = (secondaryCluster.hwEta()==iEtaM);
          bool mergeRight = (secondaryCluster.hwEta()==iEtaP);

          if(mergeUp && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)) canBeMerged = false;
          //std::cout<<"  "<<(mergeUp && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN));
          if(!mergeUp && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)) canBeMerged = false;
          //std::cout<<"  "<<(!mergeUp && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN));
          if(mergeLeft && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)) canBeMerged = false;
          //std::cout<<"  "<<(mergeLeft && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT));
          if(mergeRight && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)) canBeMerged = false;
          //std::cout<<"  "<<(mergeRight && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT))<<"\n";
          if(canBeMerged) {
            double calibPt = calibratedPt(mainCluster.hwPtEm()+secondaryCluster.hwPtEm(), mainCluster.hwPtHad()+secondaryCluster.hwPtHad(), mainCluster.hwEta());
            math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
            l1t::Tau tau( p4, mainCluster.hwPt()+secondaryCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
            taus.push_back(tau);
            //std::cout<<"   Make tau, Merging cluster eta="<<secondaryCluster.hwEta()<<", phi="<<secondaryCluster.hwPhi()<<", E="<<secondaryCluster.hwPt()<<"\n";
          }
          else {
            double calibPt = calibratedPt(mainCluster.hwPtEm(), mainCluster.hwPtHad(), mainCluster.hwEta());
            math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
            l1t::Tau tau( p4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
            taus.push_back(tau);
            //std::cout<<"   Make tau, No merging\n";
          }
        }
        else {
          bool canBeKept = false;
          bool mergeUp = (secondaryCluster.hwPhi()==iPhiM2 || secondaryCluster.hwPhi()==iPhiM3);
          bool mergeLeft = (secondaryCluster.hwEta()==iEtaM);
          bool mergeRight = (secondaryCluster.hwEta()==iEtaP);

          if(mergeUp && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)) canBeKept = true;
          //std::cout<<"  "<<(mergeUp && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN));
          if(!mergeUp && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN)) canBeKept = true;
          //std::cout<<"  "<<(!mergeUp && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_UPDOWN));
          if(mergeLeft && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)) canBeKept = true;
          //std::cout<<"  "<<(mergeLeft && !secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT));
          if(mergeRight && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT)) canBeKept = true;
          //std::cout<<"  "<<(mergeRight && secondaryCluster.checkClusterFlag(CaloCluster::MERGE_LEFTRIGHT))<<"\n";
          if(canBeKept) {
            double calibPt = calibratedPt(mainCluster.hwPtEm(), mainCluster.hwPtHad(), mainCluster.hwEta());
            math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
            l1t::Tau tau( p4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
            taus.push_back(tau);
            //std::cout<<"   Make tau, No merging\n";
          }
        }
      }
      else {
        double calibPt = calibratedPt(mainCluster.hwPtEm(), mainCluster.hwPtHad(), mainCluster.hwEta());
        math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
        l1t::Tau tau( p4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
        taus.push_back(tau);
        //std::cout<<"   Make tau, No merging\n";
      }
    }
  }
}

void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::loadCalibrationLuts()
{
  float minScale    = 0.;
  float maxScale    = 2.;
  float minScaleEta = 0.5;
  float maxScaleEta = 1.5;
  offsetBarrelEH_   = 0.5;
  offsetBarrelH_    = 1.5;
  offsetEndcapsEH_  = 0.;
  offsetEndcapsH_   = 1.5;

  std::vector<l1t::LUT*> luts;
  luts.push_back( params_->tauCalibrationLUTBarrelA()  );
  luts.push_back( params_->tauCalibrationLUTBarrelB()  );
  luts.push_back( params_->tauCalibrationLUTBarrelC()  );
  luts.push_back( params_->tauCalibrationLUTEndcapsA() );
  luts.push_back( params_->tauCalibrationLUTEndcapsB() );
  luts.push_back( params_->tauCalibrationLUTEndcapsC() );

  unsigned int size = (1 << luts.back()->nrBitsData());
  unsigned int nBins = (1 << luts.back()->nrBitsAddress());


  std::vector<float> emptyCoeff;
  emptyCoeff.resize(nBins,0.);
  float binSize = (maxScale-minScale)/(float)size;
  for(unsigned iLut=0;  iLut < luts.size();  ++iLut ) {
    coefficients_.push_back(emptyCoeff);
    for(unsigned addr=0;addr<nBins;addr++) {
      float y = (float)luts[iLut]->data(addr);
      coefficients_[iLut][addr] = minScale + binSize*y;
    }
  }

  l1t::LUT* lutEta = params_->tauCalibrationLUTEta();
  size = (1 << lutEta->nrBitsData());
  nBins = (1 << lutEta->nrBitsAddress());
  emptyCoeff.resize(nBins,0.);
  binSize = (maxScaleEta-minScaleEta)/(float)size;
  coefficients_.push_back(emptyCoeff);
  for(unsigned addr=0;addr<nBins;addr++) {
    float y = (float)lutEta->data(addr);
    coefficients_.back()[addr] = minScaleEta + binSize*y;
  }

}


double l1t::Stage2Layer2TauAlgorithmFirmwareImp1::calibratedPt(int hwPtEm, int hwPtHad, int ieta)
{
  // ET calibration
  bool barrel = (ieta<=17);
  unsigned int nBins = coefficients_[0].size();
  double e = (double)hwPtEm*params_->tauLsb();
  double h = (double)hwPtHad*params_->tauLsb();
  double calibPt = 0.;
  int ilutOffset = (barrel) ? 0: 3;
  unsigned int ibin=(unsigned int)(floor(e+h));
  if (ibin >= nBins -1) ibin = nBins-1;
  if(e>0.) {
    double offset = (barrel) ? offsetBarrelEH_ : offsetEndcapsEH_;
    calibPt = e*coefficients_[ilutOffset][ibin] + h*coefficients_[1+ilutOffset][ibin] + offset;
  }
  else {
    double offset = (barrel) ? offsetBarrelH_ : offsetEndcapsH_;
    calibPt = h*coefficients_[2+ilutOffset][ibin]+offset;
  } 

  // eta calibration
  if(ieta<-28) ieta=-28;
  if(ieta>28) ieta=28;
  ibin = (ieta>0 ? ieta+27 : ieta+28);
  calibPt *= coefficients_.back()[ibin];

  return calibPt;
}
