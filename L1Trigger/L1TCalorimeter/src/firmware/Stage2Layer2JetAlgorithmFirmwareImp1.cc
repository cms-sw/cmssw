// 
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

namespace l1t {
  bool operator > ( l1t::Jet& a, l1t::Jet& b )
  {
    if ( a.hwPt() > b.hwPt() ) {
      return true;
    } else {
      return false;
    }
  }
}

// jet mask, needs to be configurable at some point
// just a square for now
// for 1 do greater than, for 2 do greater than equal to

int mask_[9][9] = {
  { 1,2,2,2,2,2,2,2,2 },
  { 1,1,2,2,2,2,2,2,2 },
  { 1,1,1,2,2,2,2,2,2 },
  { 1,1,1,1,2,2,2,2,2 },
  { 1,1,1,1,0,2,2,2,2 },
  { 1,1,1,1,1,2,2,2,2 },
  { 1,1,1,1,1,1,2,2,2 },
  { 1,1,1,1,1,1,1,2,2 },
  { 1,1,1,1,1,1,1,1,2 },
};


std::vector<l1t::Jet>::iterator start_, end_;

l1t::Stage2Layer2JetAlgorithmFirmwareImp1::Stage2Layer2JetAlgorithmFirmwareImp1(CaloParamsHelper* params) :
  params_(params){}


l1t::Stage2Layer2JetAlgorithmFirmwareImp1::~Stage2Layer2JetAlgorithmFirmwareImp1() {}

void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
							     std::vector<l1t::Jet> & jets,
							     std::vector<l1t::Jet> & alljets) {
  
  // find jets
  create(towers, jets, alljets, params_->jetPUSType());

  // jet energy corrections
  calibrate(jets, 0); // pass the jet collection and the hw threshold above which to calibrate
  calibrate(alljets, 0); // pass all jets and the hw threshold above which to calibrate
  
}


void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::create(const std::vector<l1t::CaloTower> & towers,
						       std::vector<l1t::Jet> & jets, 
						       std::vector<l1t::Jet> & alljets, 
						       std::string PUSubMethod) {
  
  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {
    
    // the 4 groups of rings
    std::vector<int> ringGroup1, ringGroup2, ringGroup3, ringGroup4;
    for (int i=1; i<=CaloTools::kHFEnd-5; i++) {
      if      ( ! ((i-1)%4) ) ringGroup1.push_back( i * etaSide );
      else if ( ! ((i-2)%4) ) ringGroup2.push_back( i * etaSide );
      else if ( ! ((i-3)%4) ) ringGroup3.push_back( i * etaSide );
      else if ( ! ((i-4)%4) ) ringGroup4.push_back( i * etaSide );
    }
    std::vector< std::vector<int> > theRings = { ringGroup1, ringGroup2, ringGroup3, ringGroup4 };
    
    // the 24 jets in this eta side
    std::vector<l1t::Jet> jetsHalf;
       
    // loop over the 4 groups of rings
    for ( unsigned ringGroupIt=1; ringGroupIt<=theRings.size(); ringGroupIt++ ) {
      
      // the 6 accumulated jets
      std::vector<l1t::Jet> jetsAccu;
     
      // loop over the 10 rings in this group
      for ( unsigned ringIt=0; ringIt<theRings.at(ringGroupIt-1).size(); ringIt++ ) {
	
	int ieta = theRings.at(ringGroupIt-1).at(ringIt);
       
	// the jets in this ring
	std::vector<l1t::Jet> jetsRing;
	
	// loop over phi in the ring
	for ( int iphi=1; iphi<=CaloTools::kHBHENrPhi; ++iphi ) {
	  
	  // no more than 18 jets per ring
	  if (jetsRing.size()==18) break;
	  
	  // seed tower
	  const CaloTower& tow = CaloTools::getTower(towers, ieta, iphi); 
	  
	  int seedEt = tow.hwPt();
	  int iEt = seedEt;
	  bool vetoCandidate = false;
	  
	  // check it passes the seed threshold
	  if(iEt < floor(params_->jetSeedThreshold()/params_->towerLsbSum())) continue;
	  
	  // loop over towers in this jet
	  for( int deta = -4; deta < 5; ++deta ) {
	    for( int dphi = -4; dphi < 5; ++dphi ) {
	      
	      int towEt = 0;
	      int ietaTest = ieta+deta;
	      int iphiTest = iphi+dphi;
	      
	      // wrap around phi
	      while ( iphiTest > CaloTools::kHBHENrPhi ) iphiTest -= CaloTools::kHBHENrPhi;
	      while ( iphiTest < 1 ) iphiTest += CaloTools::kHBHENrPhi;
	      
	      // wrap over eta=0
	      if (ieta > 0 && ietaTest <=0) ietaTest -= 1;
	      if (ieta < 0 && ietaTest >=0) ietaTest += 1;
	   
	      // check jet mask and sum tower et
	      const CaloTower& towTest = CaloTools::getTower(towers, ietaTest, iphiTest);
	      towEt = towTest.hwPt();
	      
              if      (mask_[8-(dphi+4)][deta+4] == 0) continue;
	      else if (mask_[8-(dphi+4)][deta+4] == 1) vetoCandidate = (seedEt < towEt);
	      else if (mask_[8-(dphi+4)][deta+4] == 2) vetoCandidate = (seedEt <= towEt);
	      
	      if (vetoCandidate) break;
	      else iEt += towEt;
	   
	    }
	    if(vetoCandidate) break; 
	  }
	  
	  // add the jet to the list
	  if (!vetoCandidate) {
	
	    if (PUSubMethod == "Donut")       iEt -= donutPUEstimate(ieta, iphi, 5, towers);	    
	    if (PUSubMethod == "ChunkyDonut") iEt -= chunkyDonutPUEstimate(ieta, iphi, 5, towers);
	    	   
            if (iEt<=0) continue;
 
	    math::XYZTLorentzVector p4;
	    int caloEta = CaloTools::caloEta(ieta);
	    l1t::Jet jet( p4, iEt, caloEta, iphi, 0);
	    
	    jetsRing.push_back(jet);
	    alljets.push_back(jet);
	    
	  }
	  
	}
	
	// sort these jets and keep top 6
	start_ = jetsRing.begin();  
	end_   = jetsRing.end();
	BitonicSort<l1t::Jet>(down, start_, end_);
	if (jetsRing.size()>6) jetsRing.resize(6);
	  
	// merge with the accumulated jets
	std::vector<l1t::Jet> jetsSort;
	jetsSort.insert(jetsSort.end(), jetsAccu.begin(), jetsAccu.end());
	jetsSort.insert(jetsSort.end(), jetsRing.begin(), jetsRing.end());
	
	// sort and truncate
	start_ = jetsSort.begin();
	end_   = jetsSort.end();
	BitonicSort<l1t::Jet>(down, start_, end_); // or just use BitonicMerge
	if (jetsSort.size()>6) jetsSort.resize(6);
	
	// update accumulated jets
	jetsAccu = jetsSort;
	
      }
      
      // add to final jets in this eta side
      jetsHalf.insert(jetsHalf.end(), jetsAccu.begin(), jetsAccu.end());
      
    }
    
    // sort the 24 jets and keep top 6
    start_ = jetsHalf.begin();  
    end_   = jetsHalf.end();
    BitonicSort<l1t::Jet>(down, start_, end_);
    if (jetsHalf.size()>6) jetsHalf.resize(6);

    // add to final jets
    jets.insert(jets.end(), jetsHalf.begin(), jetsHalf.end());
    
  }

}

//A function to return the value for donut subtraction around an ieta and iphi position for donut subtraction
//Also pass it a vector to store the individual values of the strip for later testing
//The size is the number of ieta/iphi units out the ring is (ie for 9x9 jets, we want the 11x11 for PUS therefore we want to go 5 out, so size is 5)
int l1t::Stage2Layer2JetAlgorithmFirmwareImp1::donutPUEstimate(int jetEta, 
							       int jetPhi, 
							       int size, 
							       const std::vector<l1t::CaloTower> & towers){

  //ring is a vector with 4 ring strips, one for each side of the ring
  std::vector<int> ring(4,0);

  int iphiUp = jetPhi + size;
  while ( iphiUp > CaloTools::kHBHENrPhi ) iphiUp -= CaloTools::kHBHENrPhi;
  int iphiDown = jetPhi - size;
  while ( iphiDown < 1 ) iphiDown += CaloTools::kHBHENrPhi;

  int ietaUp = (jetEta + size > CaloTools::kHFEnd) ? 999 : jetEta+size;
  int ietaDown = (abs(jetEta - size) > CaloTools::kHFEnd) ? 999 : jetEta-size;

  for (int ieta = jetEta - size+1; ieta < jetEta + size; ++ieta)   
  {
    
    if (abs(ieta) > CaloTools::kHFEnd || abs(ieta) < 1) continue;
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
    while ( towerPhi > CaloTools::kHBHENrPhi ) towerPhi -= CaloTools::kHBHENrPhi;
    while ( towerPhi < 1 ) towerPhi += CaloTools::kHBHENrPhi;
    
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

int l1t::Stage2Layer2JetAlgorithmFirmwareImp1::chunkyDonutPUEstimate(int jetEta, 
								     int jetPhi, 
								     int size, 
								     const std::vector<l1t::CaloTower> & towers){
 
   // ring is a vector with 4 ring strips, one for each side of the ring
  // order is PhiUp, PhiDown, EtaUp, EtaDown
  std::vector<int> ring(4,0);

  // number of strips in donut - should make this configurable
  int nStrips = 3;

  // loop over strips
  for (int stripIt=0; stripIt<nStrips; stripIt++) {

    int iphiUp   = jetPhi + size + stripIt;
    int iphiDown = jetPhi - size - stripIt;
    while ( iphiUp > CaloTools::kHBHENrPhi )   iphiUp   -= CaloTools::kHBHENrPhi;
    while ( iphiDown < 1 ) iphiDown += CaloTools::kHBHENrPhi;

    int ietaUp   = jetEta + size + stripIt;
    int ietaDown = jetEta - size - stripIt;
    if ( jetEta<0 && ietaUp>=0 )   ietaUp   += 1;
    if ( jetEta>0 && ietaDown<=0 ) ietaDown -= 1;
    
    // do PhiUp and PhiDown
    for (int ieta=jetEta-size+1; ieta<jetEta+size; ++ieta) {
      
      if (abs(ieta) > CaloTools::kHFEnd) continue;
      
      int towEta = ieta;
      if (jetEta>0 && towEta<=0) towEta-=1;
      if (jetEta<0 && towEta>=0) towEta+=1;
            
      const CaloTower& towPhiUp = CaloTools::getTower(towers, towEta, iphiUp);
      int towEt = towPhiUp.hwPt();
      ring[0] += towEt;
            
      const CaloTower& towPhiDown = CaloTools::getTower(towers, towEta, iphiDown);
      towEt = towPhiDown.hwPt();
      ring[1] += towEt;
            
    } 
    
    // do EtaUp
    for (int iphi=jetPhi-size+1; iphi<jetPhi+size; ++iphi) {
      
      if (abs(ietaUp) <= CaloTools::kHFEnd-1) {    
        int towPhi = iphi;
        while ( towPhi > CaloTools::kHBHENrPhi ) towPhi -= CaloTools::kHBHENrPhi;
        while ( towPhi < 1 ) towPhi += CaloTools::kHBHENrPhi;

        const CaloTower& towEtaUp = CaloTools::getTower(towers, ietaUp, towPhi);
        int towEt = towEtaUp.hwPt();
        ring[2] += towEt;
      }else{
        ring[2] = 0;
        break;
      }
      
    } 
    
    // do EtaDown
    for (int iphi=jetPhi-size+1; iphi<jetPhi+size; ++iphi) {
      
      if (abs(ietaDown) <= CaloTools::kHFEnd-1) {
        int towPhi = iphi;
        while ( towPhi > CaloTools::kHBHENrPhi ) towPhi -= CaloTools::kHBHENrPhi;
        while ( towPhi < 1 ) towPhi += CaloTools::kHBHENrPhi;
	
        const CaloTower& towEtaDown = CaloTools::getTower(towers, ietaDown, towPhi);
        int towEt = towEtaDown.hwPt();
        ring[3] += towEt;
      }else{
        ring[3] = 0;
        break;
      }
      
    }     
    
    
  }
  
  // for donut subtraction we only use the middle 2 (in energy) ring strips
  // std::sort(ring.begin(), ring.end(), std::greater<int>());
  // return ( ring[1]+ring[2] ); 

  // use lowest 3 strips as PU estimate
  std::sort( ring.begin(), ring.end() );
  return ( ring[0] + ring[1] + ring[2] );

}



void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::calibrate(std::vector<l1t::Jet> & jets, int calibThreshold) {

  if( params_->jetCalibrationType() == "function6PtParams22EtaBins" ){ //One eta bin per region

    //Check the vector of calibration parameters is the correct size
    //Order the vector in terms of the parameters per eta bin, starting in -ve eta
    //So first 6 entries are very negative eta, next 6 are the next bin etc.

    if( params_->jetCalibrationParams().size() != 6*22){
      edm::LogError("l1t|stage 2") << "Invalid input vector to calo params. Input vector of size: " <<
	params_->jetCalibrationParams().size() << "  Require size: 132  Not calibrating Stage 2 Jets" << std::endl;
      return;
    }

    //Loop over jets and apply corrections
    for(std::vector<l1t::Jet>::iterator jet = jets.begin(); jet!=jets.end(); jet++){

      //Check jet is above the calibration threshold, if not do nothing
      if(jet->hwPt() < calibThreshold) continue;

      int etaBin = CaloTools::regionEta( jet->hwEta() );

      //Get the parameters from the vector
      //Each 6 values are the parameters for an eta bin
      double params[6];
      for(int i=0; i<6; i++){
        params[i] = params_->jetCalibrationParams()[etaBin*6 + i];
      }

      //Perform the correction based on the calibration function defined
      //in calibFit
      //This is derived from the actual physical pt of the jets, not the hwEt
      //This needs to be addressed in the future
      double ptPhys = jet->hwPt() * params_->jetLsb();
      double correction = calibFit(ptPhys, params);

      math::XYZTLorentzVector p4;
      *jet = l1t::Jet( p4, correction*jet->hwPt(), jet->hwEta(), jet->hwPhi(), 0);

    }

  }
  else if( params_->jetCalibrationType() == "function8PtParams22EtaBins" ){
    // as above but with cap on max correction at low pT

    if( params_->jetCalibrationParams().size() != 8*22){
      edm::LogError("l1t|stage 2") << "Invalid input vector to calo params. Input vector of size: " <<
           params_->jetCalibrationParams().size() << "  Require size: 176  Not calibrating Stage 2 Jets" << std::endl;
      return;
    }

    for(std::vector<l1t::Jet>::iterator jet = jets.begin(); jet!=jets.end(); jet++){

      if(jet->hwPt() < calibThreshold) continue;

      int etaBin = CaloTools::regionEta( jet->hwEta() );

      double params[8];
      for(int i=0; i<8; i++){
        params[i] = params_->jetCalibrationParams()[etaBin*8 + i];
      }

      double ptPhys = jet->hwPt() * params_->jetLsb();
      double correction = params[6];
      if (ptPhys>params[7]) correction = calibFit(ptPhys, params);

      math::XYZTLorentzVector p4;
      *jet = l1t::Jet( p4, correction*jet->hwPt(), jet->hwEta(), jet->hwPhi(), 0);

    }

  } else {
    if(params_->jetCalibrationType() != "None" && params_->jetCalibrationType() != "none") 
      edm::LogError("l1t|stage 2") << "Invalid calibration type in calo params. Not calibrating Stage 2 Jets" << std::endl;
    return;
  }


}

//Function for the calibration, correct as a function of pT in bins of eta
double l1t::Stage2Layer2JetAlgorithmFirmwareImp1::calibFit( double pt, double *par ){

  double logX = log10(pt);

  double term1 = par[1] / ( logX * logX + par[2] );
  double term2 = par[3] * exp( -par[4]*((logX - par[5])*(logX - par[5])) );

  // Final fitting function, with sanity check
  double f = par[0] + term1 + term2;
  if (f < 0)
    f = 0;
  if (f != f) // stop NaN
    f = 1;
  return f;
}
