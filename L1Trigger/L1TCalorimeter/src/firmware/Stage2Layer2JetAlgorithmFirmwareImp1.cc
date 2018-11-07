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
#include "L1Trigger/L1TCalorimeter/interface/AccumulatingSort.h"
#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "TMath.h"

#include <vector>
#include <algorithm>
#include <cmath>

namespace l1t {
  bool operator > ( const l1t::Jet& a, const l1t::Jet& b ) {
    return  a.hwPt() > b.hwPt();
  }
}

// jet mask, needs to be configurable at some point
// just a square for now
// for 1 do greater than, for 2 do greater than equal to

namespace {
constexpr int mask_[9][9] = {
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

}


l1t::Stage2Layer2JetAlgorithmFirmwareImp1::Stage2Layer2JetAlgorithmFirmwareImp1(CaloParamsHelper const* params) :
  params_(params){}


void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & towers,
							     std::vector<l1t::Jet> & jets,
							     std::vector<l1t::Jet> & alljets) {
  
  // find jets
  create(towers, jets, alljets, params_->jetPUSType());

  // calibrate all jets
  calibrate(alljets, 0, true); // pass all jets and the hw threshold above which to calibrate

  // jets accumulated sort
  accuSort(jets);

}


void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::create(const std::vector<l1t::CaloTower> & towers,
						       std::vector<l1t::Jet> & jets, 
						       std::vector<l1t::Jet> & alljets, 
						       std::string PUSubMethod) {

  
  // etaSide=1 is positive eta, etaSide=-1 is negative eta
  for (int etaSide=1; etaSide>=-1; etaSide-=2) {
    
    // the 4 groups of rings
    std::vector<int> ringGroup1, ringGroup2, ringGroup3, ringGroup4;
    for (int i=1; i<=CaloTools::mpEta(CaloTools::kHFEnd); i++) {
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
	  const CaloTower& tow = CaloTools::getTower(towers, CaloTools::caloEta(ieta), iphi); 
	  
	  int seedEt = tow.hwPt();
	  int iEt = seedEt;
	  bool satSeed = false;
	  bool vetoCandidate = false;
	  
	  // check it passes the seed threshold
	  if(iEt < floor(params_->jetSeedThreshold()/params_->towerLsbSum())) continue;

	  if (seedEt == CaloTools::kSatHcal ||
	      seedEt == CaloTools::kSatEcal ||
	      seedEt == CaloTools::kSatTower)
	    satSeed=true;
	  
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
	      const CaloTower& towTest = CaloTools::getTower(towers, CaloTools::caloEta(ietaTest), iphiTest);
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

	    int rawEt = iEt;
	    int puEt(0);
	
	    math::XYZTLorentzVector p4;
	    int caloEta = CaloTools::caloEta(ieta);
	    l1t::Jet jet( p4, -999, caloEta, iphi, 0);

	    if(!params_->jetBypassPUS()
	       && !satSeed){

	      // comment out for emulator studies
	      if(params_->jetPUSUsePhiRing())
		PUSubMethod = "PhiRing1";
	      else
		PUSubMethod = "ChunkyDonut";

	      if (PUSubMethod == "Donut") {
		puEt = donutPUEstimate(ieta, iphi, 5, towers);	    
		iEt -= puEt;
	      }
	      
	      if (PUSubMethod == "ChunkyDonut"){
		puEt = chunkyDonutPUEstimate(jet, 5, towers);
		iEt -= puEt;
	      }

	      if(PUSubMethod.find("ChunkySandwich") != std::string::npos){
		puEt = chunkySandwichPUEstimate(jet, 5, towers, PUSubMethod);
		iEt -= puEt;
	      }

	      
	      if(PUSubMethod == "PhiRing1"){
		int size=5;
		std::map<int,int> SumEtEtaMap=getSumEtEtaMap(towers);
		// int jetPhi=jet.hwPhi();
		int jetEta=CaloTools::mpEta(jet.hwEta());
		int jetEtaLow=jetEta-size+1;
		int jetEtaHigh=jetEta+size-1;
		if(jetEta>0 && jetEtaLow<=0) jetEtaLow-=1;
		if(jetEta<0 && jetEtaHigh>=0) jetEtaHigh+=1;
		puEt = 0;
		for(int ieta=jetEtaLow; ieta<=jetEtaHigh; ++ieta){
		  if(ieta==0) continue; // eta start from +- 1
		  auto iter=SumEtEtaMap.find(ieta);
		  puEt+=iter->second;
		}
		puEt=(puEt*(size*2-1))/CaloTools::kHBHENrPhi;
		iEt -= puEt;
	      }
	      
	      if(PUSubMethod == "PhiRing2"){
		int size=5;
		std::map<int,int> SumEtEtaMap=getSumEtEtaMap(towers);
		// int jetPhi=jet.hwPhi();
		int jetEta=CaloTools::mpEta(jet.hwEta());
		int jetEtaLow=jetEta-size+1;
		int jetEtaHigh=jetEta+size-1;

		if(jetEta>0 && jetEtaLow<=0) jetEtaLow-=1;
		if(jetEta<0 && jetEtaHigh>=0) jetEtaHigh+=1;
		puEt = 0;

		for(int ieta=jetEtaLow; ieta<=jetEtaHigh; ++ieta){
		  if(ieta==0) continue;
		  auto iter=SumEtEtaMap.find(ieta);
		  puEt+=iter->second;
		}
		puEt-=iEt;
		puEt=(puEt*(size*2-1))/(CaloTools::kHBHENrPhi - (size*2-1));
		iEt -= puEt;
	      }
	      
	

	    }
	    
	    if (iEt<=0) continue;

	    // if tower Et is saturated, saturate jet Et
	    if (seedEt == CaloTools::kSatHcal || seedEt == CaloTools::kSatEcal || seedEt == CaloTools::kSatTower) iEt = CaloTools::kSatJet;

	    jet.setHwPt(iEt);
	    jet.setRawEt( (short int) rawEt);
	    jet.setSeedEt((short int) seedEt);
	    jet.setTowerIEta((short int) caloEta);
	    jet.setTowerIPhi((short int) iphi);
	    jet.setPUEt((short int) puEt);
	    

	    jetsRing.push_back(jet);
	    alljets.push_back(jet);
	    
	  }
	  
	}

	 // jet energy corrections
	calibrate(jetsRing, 0, false); // pass the jet collection and the hw threshold above which to calibrate

	// sort these jets and keep top 6
	auto start_ = jetsRing.begin();  
	auto end_   = jetsRing.end();
	BitonicSort<l1t::Jet>(down, start_, end_);
	if (jetsRing.size()>6) jetsRing.resize(6);
	
	// update jets
	jets.insert(jets.end(),jetsRing.begin(),jetsRing.end());
	  
      }
    }
  } 
}


//Accumulating sort
void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::accuSort(std::vector<l1t::Jet> & jets){

  math::PtEtaPhiMLorentzVector emptyP4;
  l1t::Jet tempJet (emptyP4, 0, 0, 0, 0);
  std::vector< std::vector<l1t::Jet> > jetEtaPos( 41 , std::vector<l1t::Jet>(18, tempJet));
  std::vector< std::vector<l1t::Jet> > jetEtaNeg( 41 , std::vector<l1t::Jet>(18, tempJet));
  
  for (unsigned int iJet = 0; iJet < jets.size(); iJet++)
    {
      if (jets.at(iJet).hwEta() > 0) jetEtaPos.at(jets.at(iJet).hwEta()-1).at((72-jets.at(iJet).hwPhi())/4) = jets.at(iJet);
      else  jetEtaNeg.at(-(jets.at(iJet).hwEta()+1)).at((72-jets.at(iJet).hwPhi())/4) = jets.at(iJet);
    }
  
  AccumulatingSort <l1t::Jet> etaPosSorter(7);
  AccumulatingSort <l1t::Jet> etaNegSorter(7);
  std::vector<l1t::Jet> accumEtaPos;
  std::vector<l1t::Jet> accumEtaNeg;
    
  for( int ieta = 0 ; ieta < 41 ; ++ieta)
    {
      // eta +
      std::vector<l1t::Jet>::iterator start_, end_;
      start_ = jetEtaPos.at(ieta).begin();  
      end_   = jetEtaPos.at(ieta).end();
      BitonicSort<l1t::Jet>(down, start_, end_);
      etaPosSorter.Merge( jetEtaPos.at(ieta) , accumEtaPos );
      
      // eta -
      start_ = jetEtaNeg.at(ieta).begin();  
      end_   = jetEtaNeg.at(ieta).end();
      BitonicSort<l1t::Jet>(down, start_, end_);
      etaNegSorter.Merge( jetEtaNeg.at(ieta) , accumEtaNeg );
      
    }
 
  //check for 6 & 7th jets with same et and eta. Keep jet with larger phi
  if(accumEtaPos.at(6).hwPt()==accumEtaPos.at(5).hwPt() && accumEtaPos.at(6).hwEta()==accumEtaPos.at(5).hwEta()
     && accumEtaPos.at(6).hwPhi() > accumEtaPos.at(5).hwPhi()){
    accumEtaPos.at(5)=accumEtaPos.at(6);
  }
  if(accumEtaNeg.at(6).hwPt()==accumEtaNeg.at(5).hwPt() && accumEtaNeg.at(6).hwEta()==accumEtaNeg.at(5).hwEta()
     && accumEtaNeg.at(6).hwPhi() > accumEtaNeg.at(5).hwPhi()){
    accumEtaNeg.at(5)=accumEtaNeg.at(6);
  }

  //truncate
  accumEtaPos.resize(6);
  accumEtaNeg.resize(6);
 
  // put all 12 candidates in the original jet vector, removing zero energy ones
  jets.clear();
  for (l1t::Jet accjet : accumEtaPos)
    {
      if (accjet.hwPt() > 0) jets.push_back(accjet);
    }
  for (l1t::Jet accjet : accumEtaNeg)
    {
      if (accjet.hwPt() > 0) jets.push_back(accjet);
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

  int ietaUp = jetEta+size;   //(jetEta + size > CaloTools::mpEta(CaloTools::kHFEnd)) ? 999 : jetEta+size;
  int ietaDown = jetEta-size; //(abs(jetEta - size) > CaloTools::mpEta(CaloTools::kHFEnd)) ? 999 : jetEta-size;

  for (int ieta = jetEta - size+1; ieta < jetEta + size; ++ieta)   
  {
    
    if (abs(ieta) > CaloTools::mpEta(CaloTools::kHFEnd) || abs(ieta) < 1) continue;
    int towerEta;
    
    if (jetEta > 0 && ieta <=0){
      towerEta = ieta-1;
    } else if (jetEta < 0 && ieta >=0){
      towerEta = ieta+1;
    } else {
      towerEta=ieta;
    }
    
    const CaloTower& tow = CaloTools::getTower(towers, CaloTools::caloEta(towerEta), iphiUp);
    int towEt = tow.hwPt();
    ring[0]+=towEt;
    
    const CaloTower& tow2 = CaloTools::getTower(towers, CaloTools::caloEta(towerEta), iphiDown);
    towEt = tow2.hwPt();
    ring[1]+=towEt;
    
  } 
  
  for (int iphi = jetPhi - size+1; iphi < jetPhi + size; ++iphi)   
  {
      
    int towerPhi = iphi;
    while ( towerPhi > CaloTools::kHBHENrPhi ) towerPhi -= CaloTools::kHBHENrPhi;
    while ( towerPhi < 1 ) towerPhi += CaloTools::kHBHENrPhi;
    
    const CaloTower& tow = CaloTools::getTower(towers, CaloTools::caloEta(ietaUp), towerPhi);
    int towEt = tow.hwPt();
    ring[2]+=towEt;
    
    const CaloTower& tow2 = CaloTools::getTower(towers, CaloTools::caloEta(ietaDown), towerPhi);
    towEt = tow2.hwPt();
    ring[3]+=towEt;
  } 
  
  //for the Donut Subtraction we only use the middle 2 (in energy) ring strips
  std::sort(ring.begin(), ring.end(), std::greater<int>());
  
  return 4*( ring[1]+ring[2] ); // This should really be multiplied by 4.5 not 4.
}

std::map<int,int> l1t::Stage2Layer2JetAlgorithmFirmwareImp1::getSumEtEtaMap(const std::vector<l1t::CaloTower> & towers){


	std::map<int,int> SumEtEtaMap;
	int EtaMin=-41;
	int EtaMax=41;
	int phiMin=1;
	int phiMax=72;

	for(int iEta=EtaMin ; iEta<=EtaMax; iEta++ )
	{
		int SumEt=0;
		for(int iPhi=phiMin; iPhi<=phiMax; iPhi++){
			const CaloTower& tow = CaloTools::getTower(towers, CaloTools::caloEta(iEta), iPhi);
			int towEt=tow.hwPt();
			SumEt+=towEt;

		}// end for iPih<=phiMax
		SumEtEtaMap[iEta]=SumEt;

	}// end for iEta<=EtaMax

	return SumEtEtaMap;

}


std::vector<int> l1t::Stage2Layer2JetAlgorithmFirmwareImp1::getChunkyRing(l1t::Jet & jet, int size, 
									  const std::vector<l1t::CaloTower> & towers,
									  const std::string chunkyString)
{
  int jetPhi = jet.hwPhi();
  int jetEta = CaloTools::mpEta(jet.hwEta());

  // ring is a vector with 4 or 8 eta or strips, one for each side of the ring in ChunkyDonut, else pure phi
  int ringSize = 4;
  if(chunkyString == "ChunkySandwich8") ringSize = 8;

  std::vector<int> ring(ringSize,0);
  
  // number of strips in donut - should make this configurable
  int nStrips = 3;

  // loop over strips
  for (int stripIt=0; stripIt<nStrips; stripIt++) {

    int iphiUp   = jetPhi + size + stripIt;
    int iphiDown = jetPhi - size - stripIt;
    while ( iphiUp > CaloTools::kHBHENrPhi )   iphiUp   -= CaloTools::kHBHENrPhi;
    while ( iphiDown < 1 ) iphiDown += CaloTools::kHBHENrPhi;

    // we will always do phiUp and PhiDown
    // do PhiUp and PhiDown
    for (int ieta=jetEta-size+1; ieta<jetEta+size; ++ieta) {
      
      if (abs(ieta) > CaloTools::mpEta(CaloTools::kHFEnd)) continue;
      
      int towEta = ieta;
      if (jetEta>0 && towEta<=0) towEta-=1;
      if (jetEta<0 && towEta>=0) towEta+=1;
            
      const CaloTower& towPhiUp = CaloTools::getTower(towers, CaloTools::caloEta(towEta), iphiUp);
      int towEt = towPhiUp.hwPt();
      ring[0] += towEt;
            
      const CaloTower& towPhiDown = CaloTools::getTower(towers, CaloTools::caloEta(towEta), iphiDown);
      towEt = towPhiDown.hwPt();
      ring[1] += towEt;

    } 

    //Only consider eta for chunkydonut
    if(chunkyString == "ChunkyDonut"){
      int ietaUp   = jetEta + size + stripIt;
      int ietaDown = jetEta - size - stripIt;
      if ( jetEta<0 && ietaUp>=0 )   ietaUp   += 1;
      if ( jetEta>0 && ietaDown<=0 ) ietaDown -= 1;
      
      // do EtaUp
      for (int iphi=jetPhi-size+1; iphi<jetPhi+size; ++iphi) {
	
	if (abs(ietaUp) <= CaloTools::mpEta(CaloTools::kHFEnd)) {    
	  int towPhi = iphi;
	  while ( towPhi > CaloTools::kHBHENrPhi ) towPhi -= CaloTools::kHBHENrPhi;
	  while ( towPhi < 1 ) towPhi += CaloTools::kHBHENrPhi;
	  
	  const CaloTower& towEtaUp = CaloTools::getTower(towers, CaloTools::caloEta(ietaUp), towPhi);
	  int towEt = towEtaUp.hwPt();
	  ring[2] += towEt;
	}
      }

      // do EtaDown
      for (int iphi=jetPhi-size+1; iphi<jetPhi+size; ++iphi) {
	
	if (abs(ietaDown) <= CaloTools::mpEta(CaloTools::kHFEnd)) {
	  int towPhi = iphi;
	  while ( towPhi > CaloTools::kHBHENrPhi ) towPhi -= CaloTools::kHBHENrPhi;
	  while ( towPhi < 1 ) towPhi += CaloTools::kHBHENrPhi;
	  
	  const CaloTower& towEtaDown = CaloTools::getTower(towers, CaloTools::caloEta(ietaDown), towPhi);
	  int towEt = towEtaDown.hwPt();
	  ring[3] += towEt;
	}	
      }     
    }
    else if(chunkyString == "ChunkySandwich2"){//simplest of the sandwiches - we will just reuse phi flaps
      for(int i = 0; i < 2; ++i){ring[i+2] = ring[i];}
    }
    else if(chunkyString == "ChunkySandwich2Ave" || chunkyString == "ChunkySandwich"){//simplest of the sandwiches - we will just reuse phi flaps
      ring[2] = ((ring[0] + ring[1]) >> 1); //bitwise, effective div by 2
      ring[3] = (ring[0] + ring[1]); // just make sure it is >= max val
    }
    else if(chunkyString == "ChunkySandwich4" || chunkyString == "ChunkySandwich8"){//Need to calculate two or siz additional phiflaps. Note that ChunkySandwich defaults to ChunkySandwich4 variant

      for(int rI = 0; rI < (ringSize-2)/2; ++rI){
	int iphiUp2   = jetPhi + size + (stripIt + nStrips*(rI+1));
	int iphiDown2 = jetPhi - size - (stripIt + nStrips*(rI+1));
	while ( iphiUp2 > CaloTools::kHBHENrPhi )   iphiUp2   -= CaloTools::kHBHENrPhi;
	while ( iphiDown2 < 1 ) iphiDown2 += CaloTools::kHBHENrPhi;
	
	for (int ieta=jetEta-size+1; ieta<jetEta+size; ++ieta) {	
	  if (abs(ieta) > CaloTools::mpEta(CaloTools::kHFEnd)) continue;
	  
	  int towEta = ieta;
	  if (jetEta>0 && towEta<=0) towEta-=1;
	  if (jetEta<0 && towEta>=0) towEta+=1;
	  
	  const CaloTower& towPhiUp = CaloTools::getTower(towers, CaloTools::caloEta(towEta), iphiUp2);
	  int towEt = towPhiUp.hwPt();
	  ring[2 + rI*2] += towEt;
	  
	  const CaloTower& towPhiDown = CaloTools::getTower(towers, CaloTools::caloEta(towEta), iphiDown2);
	  towEt = towPhiDown.hwPt();
	  ring[3 + rI*2] += towEt;	
	}
      }
    }    
  }

  //sort our final ring and return;
  std::sort( ring.begin(), ring.end() );
  
  return ring;  
}


int l1t::Stage2Layer2JetAlgorithmFirmwareImp1::chunkyDonutPUEstimate(l1t::Jet & jet, int size, 
								     const std::vector<l1t::CaloTower> & towers){
 
   // ring is a vector with 4 ring strips, one for each side of the ring
  // PhiUp, PhiDown, EtaUp, EtaDown, sorted w/ call to chunkyring
  std::vector<int> ring = getChunkyRing(jet, size, towers, "ChunkyDonut");
  for(unsigned int i=0; i<4; ++i) jet.setPUDonutEt(i, (short int) ring[i]);
  return ( ring[0] + ring[1] + ring[2] );  
}

//Following example of chunky donut, but considering only phi flaps
//Useful for simple handling of zeroed ieta strips and for high PU environments like PbPb
int l1t::Stage2Layer2JetAlgorithmFirmwareImp1::chunkySandwichPUEstimate(l1t::Jet & jet, int size, 
									const std::vector<l1t::CaloTower> & towers,
									const std::string chunkySandwichStr){
  //ring will be handled identically by all variants for now - could consider a different picking w/ ChunkySandwich8, say to exclude a downward fluctuation w/ 1,2,3 from ring, etc.
  std::vector<int> ring = getChunkyRing(jet, size, towers, chunkySandwichStr);
  for(unsigned int i=0; i<4; ++i) jet.setPUDonutEt(i, (short int)ring[i]);
  return (ring[0] + ring[1] + ring[2]);  
}


void l1t::Stage2Layer2JetAlgorithmFirmwareImp1::calibrate(std::vector<l1t::Jet> & jets, int calibThreshold, bool isAllJets) {

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
      if(jet->hwPt() >= 0xFFFF) continue;

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

      jet->setHwPt(correction*jet->hwPt());

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
      if(jet->hwPt() >= 0xFFFF) continue;

      int etaBin = CaloTools::regionEta( jet->hwEta() );

      double params[8];
      for(int i=0; i<8; i++){
        params[i] = params_->jetCalibrationParams()[etaBin*8 + i];
      }

      double ptPhys = jet->hwPt() * params_->jetLsb();
      double correction = params[6];
      if (ptPhys>params[7]) correction = calibFit(ptPhys, params);

      jet->setHwPt(correction*jet->hwPt());

    }

  }
  else if( params_->jetCalibrationType() == "functionErf11PtParams16EtaBins" ){
    // as above but with cap on max correction at low pT

    if( params_->jetCalibrationParams().size() != 11*16){
      edm::LogError("l1t|stage 2") << "Invalid input vector to calo params. Input vector of size: " <<
      params_->jetCalibrationParams().size() << "  Require size: 176  Not calibrating Stage 2 Jets" << std::endl;
      return;
    }

    for(std::vector<l1t::Jet>::iterator jet = jets.begin(); jet!=jets.end(); jet++){

      if(jet->hwPt() < calibThreshold) continue;
      if(jet->hwPt() >= 0xFFFF) continue;

      int etaBin = CaloTools::bin16Eta( jet->hwEta() );

      double params[11];
      for(int i=0; i<11; i++){
        params[i] = params_->jetCalibrationParams()[etaBin*11 + i];
      }

      double ptPhys = jet->hwPt() * params_->jetLsb();
      double correction = params[7];

      if (ptPhys<params[8]) correction = params[7];
      else if (ptPhys>params[10]) correction = params[9];
      else correction = calibFitErr(ptPhys, params); 

      jet->setHwPt(correction*jet->hwPt());

    }

  } 
  else if( params_->jetCalibrationType() == "LUT" ){
    // Calibrate using 3 LUTs: pt and eta compression LUTs, and a multiplicand/addend LUT.
    // The pt and eta are each converted to a compressed scale using individual LUTs
    // pt : 8 -> 4 bits, eta 6 -> 4 bits
    // This then forms an address. Using the third LUT, we get a
    // multiplicand & addend, so we can do y = m*x + c on the original
    // (i.e. non-compressed) jet pt.
    // The multiplicand is 10-bit unsigned, addend is 8-bit signed.

    //Loop over jets and apply corrections
    for(std::vector<l1t::Jet>::iterator jet = jets.begin(); jet!=jets.end(); jet++){

      //Check jet is above the calibration threshold, if not do nothing
      if(jet->hwPt() < calibThreshold) continue;
      
      //don't calibrate saturated jets for HT
      if( isAllJets && (jet->hwPt() == CaloTools::kSatJet) ) continue;

      // In the firmware, we take bits 1 to 8 of the hwPt.
      // To avoid getting nonsense by only taking smaller bits,
      // any values larger than 511 are automatically set to 511.
      int jetHwPt = jet->hwPt();
      if (jetHwPt >= 0x200) {
	jetHwPt = 0x1FF;
      }
      unsigned int ptBin = params_->jetCompressPtLUT()->data(jetHwPt>>1);
      unsigned int etaBin = params_->jetCompressEtaLUT()->data(abs(CaloTools::mpEta(jet->hwEta())));
      unsigned int compBin =  (etaBin<<4) | ptBin;
      
      unsigned int addPlusMult = params_->jetCalibrationLUT()->data(compBin);
      unsigned int multiplier = addPlusMult & 0x3ff;
      // handles -ve numbers correctly
      int8_t addend = (addPlusMult>>10);
      unsigned int jetPtCorr = ((jet->hwPt()*multiplier)>>9) + addend;

      if(jetPtCorr < 0xFFFF) {
	jet->setHwPt(jetPtCorr);
      }
      else {
	jet->setHwPt(0xFFFF);
      }
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

//NEW Function for the calibration, correct as a function of pT in bins of eta
double l1t::Stage2Layer2JetAlgorithmFirmwareImp1::calibFitErr( double pt, double *par ){

  double f = par[0]+par[1]*TMath::Erf(par[2]*(log10(pt)-par[3])+par[4]*exp(par[5]*(log10(pt)-par[6])*(log10(pt)-par[6])));
  // sanity check
  if (f < 0)
    f = 0;
  if (f != f) // stop NaN
    f = 1;
  return f;
}
