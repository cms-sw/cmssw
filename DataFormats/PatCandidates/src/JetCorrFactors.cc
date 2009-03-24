//
// $Id: JetCorrFactors.cc,v 1.4 2009/02/19 15:39:29 rwolf Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include <iostream>

using namespace pat;

JetCorrFactors::JetCorrFactors() 
{
  flavourIndepCorrections_.push_back( -1 ); 			//L1 default
  flavourIndepCorrections_.push_back( -1 ); 			//L2 default
  flavourIndepCorrections_.push_back( -1 ); 			//L3 default
  flavourIndepCorrections_.push_back( -1 ); 			//L4 default
  flavourDepCorrections_.  push_back( FlavourCorrections() ); 	//L5 default
  flavourIndepCorrections_.push_back( -1 ); 			//L6 default
  flavourDepCorrections_.  push_back( FlavourCorrections() ); 	//L7 default
}

JetCorrFactors::JetCorrFactors(std::string &label, float l1, float l2, float l3, float l4, FlavourCorrections l5, float l6, FlavourCorrections l7) : 
  label_(label) 
{
  flavourIndepCorrections_.push_back( l1 ); 		
  flavourIndepCorrections_.push_back( l2 ); 		
  flavourIndepCorrections_.push_back( l3 ); 		
  flavourIndepCorrections_.push_back( l4 ); 		
  flavourDepCorrections_  .push_back( l5 ); 
  flavourIndepCorrections_.push_back( l6 ); 		
  flavourDepCorrections_  .push_back( l7 ); 
}

bool
JetCorrFactors::hasCorrection(CorrStep step) const 
{
  size_t i = istep(step);
  // raw is always true
  if (i == 0) 
    return true;
  // for level is <=4
  if ((i <= 4)                              && // level is <=4  
      (i <= flavourIndepCorrections_.size())&& // vector is large enough 
      (flavourIndepCorrections_[i-1] != -1.0)) // correction factor is !=-1 
    return true;
  bool has_flavour = false;        
  // starting from L5 
  switch (i) {
  case 5: 
    // flavor corrections are available
    if (flavourDepCorrections_.size() > 0) has_flavour = (flavourDepCorrections_[0]!=-1.0);
    return (flavourIndepCorrections_.size() >= 4) && has_flavour;
  case 6: 
    // flavor corrections are available
    return (flavourIndepCorrections_.size() >= 5) && (flavourDepCorrections_.size() > 0);
  case 7: 
    // flavor & parton corrections are available
    if (flavourDepCorrections_.size() > 1) has_flavour = (flavourDepCorrections_[1]!=-1.0);
    return (flavourIndepCorrections_.size() >= 5) && has_flavour;
  default: 
    return false;
  }
}

float 
JetCorrFactors::correction(CorrStep laststep, CorrStep begin) const 
{
  // first make sure that begin < end
  if (laststep == begin){
    return 1.0;
  }
  else if (begin > laststep){
    return 1.0/correction(begin,laststep);
  }
  size_t step = istep(begin)+1;
  size_t myflav = iflav(laststep), oldflav = iflav(begin);
  // begin==5 and flavor is not 0 as default
  if (myflav != oldflav && step==6) myflav = oldflav;
  
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
  // report if at least one service is
  // missing in the correction chain 
  if(lostService ){
    edm::LogInfo( "JetCorrFactors" )
      << " Chain of JetCorrectors is not consecutive; at least     \n"
      << " one JetCorrector is missing in the chain of JetCorrectors.";
  }
  return fabs(float(accum));
}

float
JetCorrFactors::getFlavCorr(const JetCorrFactors::FlavourCorrections &item, const size_t & flavcode ) 
{
  switch (flavcode) {
  case 0: 
    return item.g; 
  case 1: 
    return item.uds; 
  case 4: 
    return item.c; 
  case 5: 
    return item.b; 
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
      << "invalid jet correction level " << step << " requested" << std::endl;
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

void
JetCorrFactors::print() const
{
  edm::LogInfo( "JetCorrFactors" )
    << " JetCorrFactors: " << label_ << "\n"
    << "  * L1         : " << correction(L1,    Raw  ) << "\n"
    << "  * L2         : " << correction(L2,     L1  ) << "\n"
    << "  * L3         : " << correction(L3,     L2  ) << "\n"
    << "  * L4         : " << correction(L4,     L3  ) << "\n"
    << "  * L5(glu   ) : " << correction(L5g,    L4  ) << "\n"
    << "  *   (uds   ) : " << correction(L5uds,  L4  ) << "\n"
    << "  *   (charm ) : " << correction(L5c,    L4  ) << "\n"
    << "  *   (beauty) : " << correction(L5b,    L4  ) << "\n"
    << "  * L6(glu   ) : " << correction(L6g,   L5g  ) << "\n"
    << "  * L6(uds   ) : " << correction(L6uds, L5uds) << "\n"
    << "  * L6(charm ) : " << correction(L6c,   L5c  ) << "\n"
    << "  * L6(beauty) : " << correction(L6b,   L5b  ) << "\n"
    << "  * L7(glu   ) : " << correction(L7g,   L6g  ) << "\n"
    << "  *   (uds   ) : " << correction(L7uds, L6uds) << "\n"
    << "  *   (charm ) : " << correction(L7c,   L6c  ) << "\n"
    << "  *   (beauty) : " << correction(L7b,   L6b  );
}
