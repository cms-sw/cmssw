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

#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include <vector>
#include <algorithm>

l1t::Stage2Layer2JetAlgorithmFirmwareImp1::Stage2Layer2JetAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::Stage2Layer2JetAlgorithmFirmwareImp1::~Stage2Layer2JetAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
    std::vector<l1t::Jet> & jets) {


  // find all possible jets
  create(towers, jets, (params_->jetPUSType()=="Donut"));

  // remove overlaps
  filter(jets);

  // sort
  sort(jets);

}


void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::create(const std::vector<l1t::CaloTower> & towers,
      std::vector<l1t::Jet> & jets, bool doDonutSubtraction) {

  //Declare the range to carry out the algorithm over
  int etaMax=28, etaMin=-28, phiMax=72, phiMin=1;

   // generate jet mask
   // needs to be configurable at some point
   // just a square for now
   // for 1 do greater than, for 2 do greater than equal to
  int mask[9][9] = {
    { 1,1,1,1,1,1,1,1,1 },
    { 2,1,1,1,1,1,1,1,1 },
    { 2,2,1,1,1,1,1,1,1 },
    { 2,2,2,1,1,1,1,1,1 },
    { 2,2,2,2,0,1,1,1,1 },
    { 2,2,2,2,2,2,1,1,1 },
    { 2,2,2,2,2,2,2,1,1 },
    { 2,2,2,2,2,2,2,2,1 },
    { 2,2,2,2,2,2,2,2,2 }
  };


  // loop over jet positions
  for ( int ieta = etaMin ; ieta < etaMax+1 ; ++ieta ) {
    if (ieta==0) continue;
    for ( int iphi = phiMin ; iphi < phiMax+1 ; ++iphi ) {

      const CaloTower& tow = CaloTools::getTower(towers, ieta, iphi); 
      int seedEt = tow.hwEtEm();
      seedEt += tow.hwEtHad();

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
          if(iphiTest > phiMax){
            iphiTest = iphiTest -phiMax +phiMin -1;
          }
          else if(iphiTest < phiMin){
            iphiTest = iphiTest -phiMin +phiMax +1 ;
          }

          // check jet mask and sum tower et
          // re-use calo tools sum method, but for single tower
          if( mask[deta+4][dphi+4] == 1 ) { //Do greater than
            if(ietaTest <= etaMax && ietaTest >= etaMin){ //Only check if in the eta range
              const CaloTower& tow = CaloTools::getTower(towers, ietaTest, iphiTest); 
              towEt = tow.hwEtEm() + tow.hwEtHad();
              iEt+=towEt;
            }
            vetoCandidate=(seedEt<towEt);
          }
          else if( mask[deta+4][dphi+4] == 2 ) { //Do greater than equal to
            if(ietaTest <= etaMax && ietaTest >= etaMin){ //Only check if in the eta range
              const CaloTower& tow = CaloTools::getTower(towers, ietaTest, iphiTest); 
              int towEt = tow.hwEtEm() + tow.hwEtHad();
              iEt+=towEt;
            }
            vetoCandidate=(seedEt<=towEt);
          }
          if(vetoCandidate) break; 
        }
        if(vetoCandidate) break; 
      }

      // add the jet to the list
      if (!vetoCandidate) {
        math::XYZTLorentzVector p4;
        
        //If doing donut PUS find the outer ring around the jet
        if(doDonutSubtraction){
          std::vector<int> ring;
          //For 9x9 jets, subtract the 11x11 ring
          pusRing(ieta,iphi,5,ring,towers);

          //Using 2 strips with 9 towers for the subtraction
          //Need to scale it up to the jet size, ie 81/18 = 4.5
          int donutEt = 4.5*( ring[1]+ring[2] );

          iEt-=donutEt;
        }

        if(iEt>0){
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
void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::pusRing(int jetEta, int jetPhi, int size, std::vector<int>& ring, const std::vector<l1t::CaloTower> & towers) {

  //Declare the range to carry out the algorithm over
  int etaMax=28, etaMin=-28, phiMax=72, phiMin=1;

  //ring is a vector with 4 ring strips, one for each side of the ring
  for(int i=0; i<4; ++i) ring.push_back(0);

  int iphiUp = (jetPhi + size > phiMax) ? phiMin + size - (phiMax - jetPhi) - 1:jetPhi+size;
  int iphiDown = (jetPhi - size < phiMin) ? phiMax-(size - (jetPhi - phiMin)) + 1:jetPhi-size;
  int ietaUp = (jetEta + size > etaMax) ? 999 : jetEta+size;
  int ietaDown = (jetEta - size < etaMin) ? 999 : jetEta-size;

  for (int ieta = jetEta - size+1; ieta != jetEta + size; ++ieta)   
  {
    if (ieta > etaMax || ieta < etaMin) continue;
    const CaloTower& tow = CaloTools::getTower(towers, ieta, iphiUp);
    int towEt = tow.hwEtEm() + tow.hwEtHad();
    ring[0]+=towEt;
    const CaloTower& tow2 = CaloTools::getTower(towers, ieta, iphiDown);
    towEt = tow2.hwEtEm() + tow2.hwEtHad();
    ring[1]+=towEt;
  } 
  for (int iphi = jetPhi - size+1; iphi != jetPhi + size; ++iphi)   
  {
    int towerPhi;
    if (iphi < phiMin)
    {
      towerPhi = phiMax-(size - (jetPhi - phiMin)) + 1;
    }
    else if (iphi > phiMax)
    {
      towerPhi = phiMin + size - (phiMax - jetPhi) - 1;
    }
    else 
    {
      towerPhi = iphi;
    }
    //if (ieta > etaMax || ieta < etaMin) continue;
    const CaloTower& tow = CaloTools::getTower(towers, ietaUp, towerPhi);
    int towEt = tow.hwEtEm() + tow.hwEtHad();
    ring[2]+=towEt;
    const CaloTower& tow2 = CaloTools::getTower(towers, ietaDown, towerPhi);
    towEt = tow2.hwEtEm() + tow2.hwEtHad();
    ring[3]+=towEt;
  } 

  //for the Donut Subtraction we only use the middle 2 (in energy) ring strips
  //Sort the vector in order and then return it 
  std::sort(ring.begin(), ring.end(), std::greater<int>());
}

void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::filter(std::vector<l1t::Jet> & jets) {

  //  jets.erase(std::remove_if(jets.begin(), jets.end(), jetIsZero) );

}


void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::sort(std::vector<l1t::Jet> & jets) {

  // do nothing for now!

}

// remove jets with zero et
bool l1t::Stage2Layer2JetAlgorithmFirmwareImp1::jetIsZero(l1t::Jet jet) { return jet.hwPt()==0; }

