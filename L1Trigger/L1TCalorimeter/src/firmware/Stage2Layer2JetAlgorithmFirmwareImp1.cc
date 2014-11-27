///
/// \class l1t::Stage2Layer2JetAlgorithmFirmwareImp1
///
/// \author: Adam Elwood and Matthew Citron
///
/// Description: Implementation of Jad's asymmetric map overlap algorithm with donut subtraction

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2JetAlgorithmFirmware.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include <vector>
#include <algorithm>
#include <math.h>

bool operator > ( l1t::Jet& a, l1t::Jet& b )
{
  if ( a.hwPt() > b.hwPt() ){ 
    return true;
  } else {
    return false;
  }
}

l1t::Stage2Layer2JetAlgorithmFirmwareImp1::Stage2Layer2JetAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params){}


  l1t::Stage2Layer2JetAlgorithmFirmwareImp1::~Stage2Layer2JetAlgorithmFirmwareImp1() {}

  void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
      std::vector<l1t::Jet> & jets) {

    // find all possible jets

    edm::LogInfo("L1Emulator") << "Number of towers = " << towers.size();

    if(towers.size()>0){

      create(towers, jets, params_->jetPUSType());

      //Carry out jet energy corrections
      calibrate(jets, 10/params_->jetLsb()); //Pass the jet collection and the hw threshold above which to calibrate 

      // sort
      sort(jets);
    }
  }


void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::create(const std::vector<l1t::CaloTower> & towers,
    std::vector<l1t::Jet> & jets, std::string PUSubMethod) {

  //Declare the range to carry out the algorithm over
  int etaMax=40, etaMin=-40, phiMax=72, phiMin=1;

  // generate jet mask
  // needs to be configurable at some point
  // just a square for now
  // for 1 do greater than, for 2 do greater than equal to
  int mask[9][9] = {
    { 1,1,1,1,1,1,1,1,1 },
    { 1,1,1,1,1,1,1,1,2 },
    { 1,1,1,1,1,1,1,2,2 },
    { 1,1,1,1,1,1,2,2,2 },
    { 1,1,1,1,0,2,2,2,2 },
    { 1,1,1,2,2,2,2,2,2 },
    { 1,1,2,2,2,2,2,2,2 },
    { 1,2,2,2,2,2,2,2,2 },
    { 2,2,2,2,2,2,2,2,2 }
  };

  // loop over jet positions
  for ( int ieta = etaMin ; ieta <= etaMax ; ++ieta ) {
    if (ieta==0) continue;
    for ( int iphi = phiMin ; iphi <= phiMax ; ++iphi ) {

      const CaloTower& tow = CaloTools::getTower(towers, ieta, iphi); 

      int seedEt=tow.hwPt();
      int iEt(seedEt);
      bool vetoCandidate(false);

      //Check it passes the seed threshold
      if(iEt < floor(params_->jetSeedThreshold()/params_->towerLsbSum())) continue;

      // loop over towers in this jet
      for( int deta = -4; deta < 5; ++deta ) {
        for( int dphi = -4; dphi < 5; ++dphi ) {

          int towEt = 0;
          int ietaTest = ieta+deta;
          int iphiTest = iphi+dphi;

          //Wrap around phi
          while ( iphiTest > phiMax ) iphiTest -= phiMax;
          while ( iphiTest < phiMin ) iphiTest += phiMax;

          // Wrap over eta=0
          if (ieta > 0 && ietaTest <=0){
            ietaTest = ietaTest-1;
          }

          if (ieta < 0 && ietaTest >=0){
            ietaTest = ietaTest+1;
          }

          // check jet mask and sum tower et
          // re-use calo tools sum method, but for single tower
          // Separately for positive and negative eta as per firmware
          //          if (ieta > 0){
          if( mask[deta+4][dphi+4] == 1 ) { //Do greater than
            if(ietaTest <= etaMax && ietaTest >= etaMin){ //Only check if in the eta range
              const CaloTower& tow = CaloTools::getTower(towers, ietaTest, iphiTest); 
              towEt = tow.hwPt();
              iEt+=towEt;
            }
            vetoCandidate=(seedEt<towEt);
          }
          else if( mask[deta+4][dphi+4] == 2 ) { //Do greater than equal to
            if(ietaTest <= etaMax && ietaTest >= etaMin){ //Only check if in the eta range
              const CaloTower& tow = CaloTools::getTower(towers, ietaTest, iphiTest); 
              towEt = tow.hwPt();
              iEt+=towEt;
            }
            vetoCandidate=(seedEt<=towEt);
          }
          //          } else if (ieta<0){
          //            if( mask[8-(deta+4)][dphi+4] == 1 ) { //Do greater than                                                                                                                       
          //              if(ietaTest <= etaMax && ietaTest >= etaMin){ //Only check if in the eta range                                                                                          
          //                const CaloTower& tow = CaloTools::getTower(towers, ietaTest, iphiTest);
          //                towEt = tow.hwPt();
          //                iEt+=towEt;
          //              }
          //              vetoCandidate=(seedEt<towEt);
          //            }
          //            else if( mask[8-(deta+4)][dphi+4] == 2 ) { //Do greater than equal to                                                                                                         
          //              if(ietaTest <= etaMax && ietaTest >= etaMin){ //Only check if in the eta range                                                                                          
          //                const CaloTower& tow = CaloTools::getTower(towers, ietaTest, iphiTest);
          //                towEt = tow.hwPt();
          //                iEt+=towEt;
          //              }
          //              vetoCandidate=(seedEt<=towEt);
          //            } 
          //          }

          if(vetoCandidate) break; 
        }
        if(vetoCandidate) break; 
      }

      // add the jet to the list
      if (!vetoCandidate) {

        if (PUSubMethod == "Donut") iEt -= donutPUEstimate(ieta, iphi, 5, towers);

        if (PUSubMethod == "ChunkyDonut") iEt -= chunkyDonutPUEstimate(ieta, iphi, 5, towers);

        if(iEt>0){

          math::XYZTLorentzVector p4;
          l1t::Jet jet( p4, iEt, ieta, iphi, 0);
          jets.push_back( jet );
        }
      }

    }
  }
}

//A function to return the value for donut subtraction around an ieta and iphi position for donut subtraction
//Also pass it a vector to store the individual values of the strip for later testing
//The size is the number of ieta/iphi units out the ring is (ie for 9x9 jets, we want the 11x11 for PUS therefore we want to go 5 out, so size is 5)
int l1t::Stage2Layer2JetAlgorithmFirmwareImp1::donutPUEstimate(int jetEta, int jetPhi, int size, const std::vector<l1t::CaloTower> & towers){

  //Declare the range to carry out the algorithm over
  int etaMax=40, etaMin=-40, phiMax=72, phiMin=1;

  //ring is a vector with 4 ring strips, one for each side of the ring
  std::vector<int> ring(4,0);

  int iphiUp = jetPhi + size;
  while ( iphiUp > phiMax ) iphiUp -= phiMax;
  int iphiDown = jetPhi - size;
  while ( iphiDown < phiMin ) iphiDown += phiMax;

  int ietaUp = (jetEta + size > etaMax) ? 999 : jetEta+size;
  int ietaDown = (jetEta - size < etaMin) ? 999 : jetEta-size;

  for (int ieta = jetEta - size+1; ieta < jetEta + size; ++ieta)   
  {

    if (ieta > etaMax || ieta < etaMin) continue;
    int towerEta;

    if (jetEta > 0 && ieta <=0){
      towerEta = ieta-1;
    } else if (jetEta < 0 && ieta >=0){
      towerEta = ieta+1;
    } else {
      towerEta=ieta;
    }

    const CaloTower& tow = CaloTools::getTower(towers, towerEta, iphiUp);
    int towEt = tow.hwPt();
    ring[0]+=towEt;

    const CaloTower& tow2 = CaloTools::getTower(towers, towerEta, iphiDown);
    towEt = tow2.hwPt();
    ring[1]+=towEt;

  } 

  for (int iphi = jetPhi - size+1; iphi < jetPhi + size; ++iphi)   
  {

    int towerPhi = iphi;
    while ( towerPhi > phiMax ) towerPhi -= phiMax;
    while ( towerPhi < phiMin ) towerPhi += phiMax;

    const CaloTower& tow = CaloTools::getTower(towers, ietaUp, towerPhi);
    int towEt = tow.hwPt();
    ring[2]+=towEt;

    const CaloTower& tow2 = CaloTools::getTower(towers, ietaDown, towerPhi);
    towEt = tow2.hwPt();
    ring[3]+=towEt;
  } 

  //for the Donut Subtraction we only use the middle 2 (in energy) ring strips
  std::sort(ring.begin(), ring.end(), std::greater<int>());

  return 4*( ring[1]+ring[2] ); // This should really be multiplied by 4.5 not 4.
}

int l1t::Stage2Layer2JetAlgorithmFirmwareImp1::chunkyDonutPUEstimate(int jetEta, int jetPhi, int size, const std::vector<l1t::CaloTower> & towers){

  //Declare the range to carry out the algorithm over
  int etaMax=40, etaMin=-40, phiMax=72, phiMin=1;

  //ring is a vector with 4 ring strips, one for each side of the ring
  std::vector<int> ring(4,0);

  // Loop over number of strips

  int iphiUp = jetPhi + size;
  while ( iphiUp > phiMax ) iphiUp -= phiMax;
  int iphiDown = jetPhi - size;
  while ( iphiDown < phiMin ) iphiDown += phiMax;

  int iphiUp1 = jetPhi + size + 1;
  while ( iphiUp1 > phiMax ) iphiUp1 -= phiMax;
  int iphiDown1 = jetPhi - size - 1;
  while ( iphiDown1 < phiMin ) iphiDown1 += phiMax;

  int iphiUp2 = jetPhi + size + 2;
  while ( iphiUp2 > phiMax ) iphiUp2 -= phiMax;
  int iphiDown2 = jetPhi - size - 2;
  while ( iphiDown2 < phiMin ) iphiDown2 += phiMax;

  int ietaUp = (jetEta + size > etaMax) ? 999 : jetEta+size;
  int ietaDown = (jetEta - size < etaMin) ? 999 : jetEta-size;

  int ietaUp1 = (jetEta + size + 1 > etaMax) ? 999 : jetEta+size+1;
  int ietaDown1 = (jetEta - size - 1 < etaMin) ? 999 : jetEta-size-1;

  int ietaUp2 = (jetEta + size + 2 > etaMax) ? 999 : jetEta+size+2;
  int ietaDown2 = (jetEta - size -2 < etaMin) ? 999 : jetEta-size-2;

  for (int ieta = jetEta - size+1; ieta < jetEta + size; ++ieta)   
  {

    if (ieta > etaMax || ieta < etaMin) continue;
    int towerEta;

    if (jetEta > 0 && ieta <=0){
      towerEta = ieta-1;
    } else if (jetEta < 0 && ieta >=0){
      towerEta = ieta+1;
    } else {
      towerEta=ieta;
    }

    const CaloTower& tow = CaloTools::getTower(towers, towerEta, iphiUp);
    int towEt = tow.hwPt();
    ring[0]+=towEt;

    const CaloTower& tow1 = CaloTools::getTower(towers, towerEta, iphiUp1);
    towEt = tow1.hwPt();
    ring[0]+=towEt;

    const CaloTower& tow2 = CaloTools::getTower(towers, towerEta, iphiUp2);
    towEt = tow2.hwPt();
    ring[0]+=towEt;

    const CaloTower& tow3 = CaloTools::getTower(towers, towerEta, iphiDown);
    towEt = tow3.hwPt();
    ring[1]+=towEt;

    const CaloTower& tow4 = CaloTools::getTower(towers, towerEta, iphiDown1);
    towEt = tow4.hwPt();
    ring[1]+=towEt;

    const CaloTower& tow5 = CaloTools::getTower(towers, towerEta, iphiDown2);
    towEt = tow5.hwPt();
    ring[1]+=towEt;

  } 

  for (int iphi = jetPhi - size+1; iphi < jetPhi + size; ++iphi)   
  {

    int towerPhi = iphi;
    while ( towerPhi > phiMax ) towerPhi -= phiMax;
    while ( towerPhi < phiMin ) towerPhi += phiMax;

    const CaloTower& tow = CaloTools::getTower(towers, ietaUp, towerPhi);
    int towEt = tow.hwPt();
    ring[2]+=towEt;

    const CaloTower& tow1 = CaloTools::getTower(towers, ietaUp1, towerPhi);
    towEt = tow1.hwPt();
    ring[2]+=towEt;

    const CaloTower& tow2 = CaloTools::getTower(towers, ietaUp2, towerPhi);
    towEt = tow2.hwPt();
    ring[2]+=towEt;

    const CaloTower& tow3 = CaloTools::getTower(towers, ietaDown, towerPhi);
    towEt = tow3.hwPt();
    ring[3]+=towEt;

    const CaloTower& tow4 = CaloTools::getTower(towers, ietaDown1, towerPhi);
    towEt = tow4.hwPt();
    ring[3]+=towEt;

    const CaloTower& tow5 = CaloTools::getTower(towers, ietaDown2, towerPhi);
    towEt = tow5.hwPt();
    ring[3]+=towEt;

  } 

  //for the Donut Subtraction we only use the middle 2 (in energy) ring strips
  std::sort(ring.begin(), ring.end(), std::greater<int>());

  return ( ring[1]+ring[2] ); 
}

void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::sort(std::vector<l1t::Jet> & jets) {


  // Split jets into positive and negative eta
  std::vector<l1t::Jet> posEta, negEta;

  for(std::vector<l1t::Jet>::const_iterator lIt = jets.begin() ; lIt != jets.end() ; ++lIt )
  {
    if (lIt->hwEta()>0) {
      posEta.push_back(*lIt);
    } else if (lIt->hwEta()<0) {
      negEta.push_back(*lIt);
    }
  }

  // Sort by pT
  std::vector<l1t::Jet>::iterator start(posEta.begin());                                                                                                                                  
  std::vector<l1t::Jet>::iterator end(posEta.end());                                                                                                                                      
  BitonicSort< l1t::Jet >(down,start,end); 

  // Sort by pT
  start = negEta.begin();
  end = negEta.end();                                                                                                                                      
  BitonicSort< l1t::Jet >(down,start,end);

  if (posEta.size()>6) posEta.resize(6); // truncate to top 6 jets per eta half
  if (negEta.size()>6) negEta.resize(6); // truncate to top 6 jets per eta half

  // Merge into output collection
  jets.resize(0);
  jets.reserve(posEta.size()+negEta.size());

  jets.insert(jets.end(), posEta.begin(), posEta.end());
  jets.insert(jets.end(), negEta.begin(), negEta.end());

  // Sort the jets by pT - this should move to the demux class but I can't make the > operator in both for some reason                                                                    
  start = jets.begin();                                                                                                                           
  end = jets.end();
  BitonicSort< l1t::Jet >(down,start,end);

}

void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::calibrate(std::vector<l1t::Jet> & jets, int calibThreshold) {

  if( params_->jetCalibrationType() == "function6PtParams22EtaBins" ){ //One eta bin per region

    //Check the vector of calibration parameters is the correct size
    //Order the vector in terms of the parameters per eta bin, starting in -ve eta
    //So first 6 entries are very negative eta, next 6 are the next bin etc.

    if( params_->jetCalibrationParams().size() != 132){
      edm::LogError("l1t|stage 2") << "Invalid input vector to calo params. Input vector of size: " <<
           params_->jetCalibrationParams().size() << "  Require size: 132  Not calibrating Stage 2 Jets" << std::endl;
      return;
    }

    //Loop over jets and apply corrections
    for(std::vector<l1t::Jet>::iterator jet = jets.begin(); jet!=jets.end(); jet++){

      //Check jet is above the calibration threshold, if not do nothing
      if(jet->hwPt() < calibThreshold) continue;

      //Make a map for the trigger towers eta to the RCT region eta
      //80 TT to 22 RCT regions
      int ttToRct[80];
      int tt=0;
      for(int rct=0; rct<22; rct++){
        if(rct<4 || rct>17){
          for(int i=0; i<3; i++){
            ttToRct[tt] = rct;
            tt++;
          }
        }else{
          for(int i=0; i<4; i++){
            ttToRct[tt] = rct;
            tt++;
          }
        }
      }

      int etaBin;
      if(jet->hwEta() < 0) //Account for a lack of 0
        etaBin = ttToRct[ jet->hwEta() + 40 ];
      else 
        etaBin = ttToRct[ jet->hwEta() + 39 ];

      double params[6]; //These are the parameters of the fit
      double ptValue[1]; //This is the pt value to be corrected

      ptValue[0] = jet->hwPt()*params_->jetLsb(); //Corrections derived in terms of physical pt

      //Get the parameters from the vector
      //Each 6 values are the parameters for an eta bin
      for(int i=0; i<6; i++){
        params[i] = params_->jetCalibrationParams()[etaBin*6 + i];
      }

      //Perform the correction based on the calibration function defined
      //in calibFit
      //This is derived from the actual pt of the jets, not the hwEt
      //This needs to be addressed in the future
      double correction =  calibFit(ptValue,params);

      math::XYZTLorentzVector p4;
      *jet = l1t::Jet( p4, correction*jet->hwPt(), jet->hwEta(), jet->hwPhi(), 0);

    }


  } else if( params_->jetCalibrationType() == "function6PtParams80EtaBins" ) { // If we want TT level calibration

    if( params_->jetCalibrationParams().size() != 480){
      edm::LogError("l1t|stage 2") << "Invalid input vector to calo params. Input vector of size: " <<
           params_->jetCalibrationParams().size() << "  Require size: 480  Not calibrating Stage 2 Jets" << std::endl;
      return;
    }

    //Loop over jets and apply corrections
    for(std::vector<l1t::Jet>::iterator jet = jets.begin(); jet!=jets.end(); jet++){

      int etaBin;
      if(jet->hwEta() < 0) //Account for a lack of 0
        etaBin = jet->hwEta() + 40;
      else 
        etaBin = jet->hwEta() + 39;


      double params[6]; //These are the parameters of the fit
      double ptValue[1]; //This is the pt value to be corrected

      ptValue[0] = jet->hwPt()*params_->jetLsb(); //Corrections derived in terms of physical pt

      //Get the parameters from the vector
      //Each 6 values are the parameters for an eta bin
      for(int i=0; i<6; i++){
        params[i] = params_->jetCalibrationParams()[etaBin*6 + i];
      }

      //Perform the correction based on the calibration function defined
      //in calibFit
      double correction =  calibFit(ptValue,params);

      math::XYZTLorentzVector p4;
      *jet = l1t::Jet( p4, correction*jet->hwPt(), jet->hwEta(), jet->hwPhi(), 0);

    }

  } else if( params_->jetCalibrationType() == "lut6PtBins22EtaBins" ) { // If we want to use a LUT instead

    edm::LogError("l1t|stage 2") << "No LUT implementation for the calibration yet" << std::endl;
    return;

  } else {
    if(params_->jetCalibrationType() != "None" || params_->jetCalibrationType() != "none") 
      edm::LogError("l1t|stage 2") << "Invalid calibration type in calo params. Not calibrating Stage 2 Jets" << std::endl;
    return;
  }


}

//Function for the calibration, correct as a function of pT in bins of eta
double l1t::Stage2Layer2JetAlgorithmFirmwareImp1::calibFit( double *v, double *par ){

  // JETMET uses log10 rather than the ln used here...
  double logX = log(v[0]);

  double term1 = par[1] / ( logX * logX + par[2] );
  double term2 = par[3] * exp( -par[4]*((logX - par[5])*(logX - par[5])) );

  // Final fitting function 
  double f    = par[0] + term1 + term2; 

  return f;
}
