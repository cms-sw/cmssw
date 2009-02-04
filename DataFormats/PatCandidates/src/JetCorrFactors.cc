//
// $Id: JetCorrFactors.cc,v 1.1 2008/03/07 18:52:55 lowette Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include <iostream>

using namespace pat;

JetCorrFactors::JetCorrFactors() {
  flavourIndepCorrections_.push_back(-1); 			//L1 default
  flavourIndepCorrections_.push_back(-1); 			//L2 default
  flavourIndepCorrections_.push_back(-1); 			//L3 default
  flavourIndepCorrections_.push_back(-1); 			//L4 default
  flavourDepCorrections_.push_back( FlavourCorrections() ); 	//L5 default
  flavourIndepCorrections_.push_back(-1); 			//L6 default
  flavourDepCorrections_.push_back( FlavourCorrections() ); 	//L7 default
}

JetCorrFactors::JetCorrFactors(float l1, float l2, float l3, float l4, FlavourCorrections l5, float l6, FlavourCorrections l7){
  flavourIndepCorrections_.push_back( l1 ); 		
  flavourIndepCorrections_.push_back( l2 ); 		
  flavourIndepCorrections_.push_back( l3 ); 		
  flavourIndepCorrections_.push_back( l4 ); 		
  flavourDepCorrections_.push_back(   l5 ); 
  flavourIndepCorrections_.push_back( l6 ); 		
  flavourDepCorrections_.push_back(   l7 ); 
}

bool
JetCorrFactors::hasCorrection(CorrStep step) const {
    size_t i = istep(step);
    if (i == 0) return true;
    if ((i <= 4) && 
        (i <= flavourIndepCorrections_.size()) && 
        (flavourIndepCorrections_[i-1] != -1.0)) return true;
    switch (i) {
        case 5: 
            return (flavourIndepCorrections_.size() >= 4) && (flavourDepCorrections_.size() > 0);
        case 6: 
            return (flavourIndepCorrections_.size() >= 5) && (flavourDepCorrections_.size() > 0);
        case 7: 
            return (flavourIndepCorrections_.size() >= 5) && (flavourDepCorrections_.size() > 1);
        default: 
            return false;
    }
}

float 
JetCorrFactors::correction(CorrStep laststep, CorrStep begin) const {
    // First make sure that begin < end
    if (laststep == begin) return 1.0;
    else if (begin > laststep) return 1.0/correction(begin,laststep);

    size_t step = istep(begin);
    size_t myflav = iflav(laststep), oldflav = iflav(begin);
    if (myflav != oldflav && step==6) myflav = oldflav; //begin==5 and flavor is not 0 as default
    
    double accum = 1.0;
    bool lostService=false;
    for (; step <= istep(laststep); ++step){
      if (!hasCorrection(step, myflav)){
	lostService=true;
	continue;
      }
      if (step!=0 && step!=5 && step!=6 && step!=7 )
        accum *= flavourIndepCorrections_[step-1];
      else if (step==6)
	accum *= flavourIndepCorrections_[4];
      else if (step==5){
	if (istep(begin)<5 || myflav==oldflav) 
	  // everything ok if step<5 or if the flav
	  // of start and target is anyhow the same 
	  accum *= getFlavCorr(flavourDepCorrections_[0], myflav);
	else if (myflav!=oldflav)
	  // we are already in the regime of flavour 
	  // dependency and the flavour is different 
	  // we have to correct for that 
	  accum *= getFlavCorr(flavourDepCorrections_[0], myflav)/getFlavCorr(flavourDepCorrections_[0], oldflav);
	else if (myflav!=oldflav && istep(begin)==7)
	  // if we were unlucky and even had passed 
	  // step 7 we also have to take back this
	  accum /= getFlavCorr(flavourDepCorrections_[1], oldflav);
      }
      else if (step==7){
	if (istep(begin)<5 || myflav==oldflav)
	  // everything ok if istep(begin)<5; we
	  // have already everything in according 
	  // to myflav; the same is true in case
	  // of oldflav==myflav
	  accum *= getFlavCorr(flavourDepCorrections_[1], myflav);
	else if (myflav!=oldflav)
	  // else we have to replace the old L7
	  // correction by the new flavour
	  accum *= getFlavCorr(flavourDepCorrections_[1], myflav)/getFlavCorr(flavourDepCorrections_[1], oldflav);
      }    
    }
    // turn accum negative to communicated that 
    // at least one service was missing in the 
    // correction chain. 
    if(lostService) accum*=-1.0;
    return float(accum);
}

float
JetCorrFactors::getFlavCorr(const JetCorrFactors::FlavourCorrections &item, const size_t & flavcode ) {
    switch (flavcode) {
        case 0: return item.g; 
        case 1: return item.uds; 
        case 4: return item.c; 
        case 5: return item.b; 
        default:
            return -1.0;
    } 
}

JetCorrFactors::CorrStep const
JetCorrFactors::corrStep(const std::string &step, const std::string &flavour) {
  CorrStep result = Raw;
  bool invalidFlavour=false;
  bool invalidCorrect=false;
  if(step=="RAW" || step=="raw" ) result = Raw; else 
  if(step=="OFF" || step=="off" ) result = L1;  else 
  if(step=="REL" || step=="rel" ) result = L2;  else 
  if(step=="ABS" || step=="abs" ) result = L3;  else 
  if(step=="EMF" || step=="emf" ) result = L4;  else 
  if(step=="HAD" || step=="had"){
    if(flavour=="GLU" || flavour=="glu") result = L5g;   else 
    if(flavour=="UDS" || flavour=="uds") result = L5uds; else 
    if(flavour=="C"   || flavour=="c"  ) result = L5c;   else 
    if(flavour=="B"   || flavour=="b"  ) result = L5b;
    else invalidFlavour=true;
  } else 
  if(step=="UE"  || step=="ue"    ){
    if(flavour=="GLU" || flavour=="glu") result = L6g;   else 
    if(flavour=="UDS" || flavour=="uds") result = L6uds; else 
    if(flavour=="C"   || flavour=="c"  ) result = L6c;   else 
    if(flavour=="B"   || flavour=="b"  ) result = L6b;
    else invalidFlavour=true;
  }
  else if(step=="PART"|| step=="part"){
    if(flavour=="GLU" || flavour=="glu") result = L7g;   else 
    if(flavour=="UDS" || flavour=="uds") result = L7uds; else 
    if(flavour=="C"   || flavour=="c"  ) result = L7c;   else 
    if(flavour=="B"   || flavour=="b"  ) result = L7b;
    else invalidFlavour=true;
  }
  else{
    invalidCorrect=true;
  }

  if(invalidFlavour)
    throw cms::Exception("InvalidRequest") 
      << "invalid flavour " << flavour << " requested for jet correction: " << step << std::endl;
  if(invalidCorrect)
    throw cms::Exception("InvalidRequest") 
      << "invalid jet correction level " << flavour << " requested" << std::endl;
  return result;
}

std::string 
JetCorrFactors::corrStep(CorrStep step) const {
  switch (istep(step)){
  case 0: 
    return "RAW";
  case 1:
    return "OFF";
  case 2:
    return "REL";
  case 3:
    return "ABS";
  case 4:
    return "EMF";
  case 5:
    return "HAD";
  case 6:
    return "UE";
  case 7:
    return "PART";
  default:
    return "ERROR";
  }
}

std::string 
JetCorrFactors::flavour(CorrStep step) const {
  switch ((int)iflav){
  case 0:
    return "GLU";
  case 1:
    return "UDS";
  case 4:
    return "C";
  case 5:
    return "B";
  default:
    return "NONE";
  }
}
