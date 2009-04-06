//
// $Id: JetCorrFactors.cc,v 1.6 2009/03/26 20:04:10 rwolf Exp $
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include <iomanip>
#include <iostream>

using namespace pat;

JetCorrFactors::JetCorrFactors() 
{
  flavourIndepCorrections_.push_back( -1 ); 			//L1 default
  flavourIndepCorrections_.push_back( -1 ); 			//L2 default
  flavourIndepCorrections_.push_back( -1 ); 			//L3 default
  flavourIndepCorrections_.push_back( -1 ); 			//L4 default
  flavourDepCorrections_.  push_back( FlavourCorrections() ); 	//L5 default
  flavourDepCorrections_  .push_back( FlavourCorrections() );   //L6 default
  flavourDepCorrections_.  push_back( FlavourCorrections() ); 	//L7 default
}

JetCorrFactors::JetCorrFactors(std::string &label, float l1, float l2, float l3, float l4, FlavourCorrections l5, FlavourCorrections l6, FlavourCorrections l7) : 
  label_(label) 
{
  flavourIndepCorrections_.push_back( l1 ); 		
  flavourIndepCorrections_.push_back( l2 ); 		
  flavourIndepCorrections_.push_back( l3 ); 		
  flavourIndepCorrections_.push_back( l4 ); 		
  flavourDepCorrections_  .push_back( l5 );
  flavourDepCorrections_  .push_back( l6 );
  flavourDepCorrections_  .push_back( l7 ); 
}

float 
JetCorrFactors::correction(CorrStep target, CorrStep begin) const 
{
  float corr=1.0;
  size_t newFlav = iflav(target), oldFlav = iflav(begin);
  size_t newStep = istep(target), oldStep = istep(begin);
  // check for flavor consistency
  if(oldStep>4 && newStep>4){
    // we deal with changes from flavor dependent to flavor dependent
    if( oldStep!=newStep && oldFlav!=newFlav ){
      throw cms::Exception("InvalidRequest") << "you try to change flavour and correction step at the same time, which is currently not allowed \n";
    }
  }
  if(oldStep>0){
    // un-correct from begin to raw (nothing has to be done for raw -> oldStep==0)
    (oldStep<=4) ? corr/=flavourIndepCorrections_[oldStep-1] : corr/=getFlavorCorrection(flavourDepCorrections_[oldStep-5], oldFlav);
  }
  if(newStep>0){
    // correct from raw to target   (nothing has to be done for raw -> newStep==0)
    (newStep<=4) ? corr*=flavourIndepCorrections_[newStep-1] : corr*=getFlavorCorrection(flavourDepCorrections_[newStep-5], newFlav);
  }
  return corr;
}

float
JetCorrFactors::getFlavorCorrection(const JetCorrFactors::FlavourCorrections& item, const size_t& flav) 
{
  switch( flav ){
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
JetCorrFactors::corrStep(const std::string& step, const std::string& flavour) 
{
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
JetCorrFactors::corrStep(CorrStep step) const 
{
  switch( istep(step) ){
  case 0: 
    return "raw";
  case 1:
    return "off";
  case 2:
    return "rel";
  case 3:
    return "abs";
  case 4:
    return "emf";
  case 5:
    return "had";
  case 6:
    return "ue";
  case 7:
    return "part";
  default:
    return "error";
  }
}

std::string 
JetCorrFactors::flavour(CorrStep step) const 
{
  switch( iflav ){
  case 0:
    return "glu";
  case 1:
    return "uds";
  case 4:
    return "c";
  case 5:
    return "b";
  default:
    return "none";
  }
}

void
JetCorrFactors::print() const
{
  edm::LogInfo( "JetCorrFactors" )
    << " JetCorrFactors: \n"
    << "  * L1         : " << std::setw(10) << correction(L1,    Raw  ) << " (" << std::setw(10) << correction(L1,    Raw  ) << ")" << "\n"
    << "  * L2         : " << std::setw(10) << correction(L2,    Raw  ) << " (" << std::setw(10) << correction(L2,     L1  ) << ")" << "\n"
    << "  * L3         : " << std::setw(10) << correction(L3,    Raw  ) << " (" << std::setw(10) << correction(L3,     L2  ) << ")" << "\n"
    << "  * L4         : " << std::setw(10) << correction(L4,    Raw  ) << " (" << std::setw(10) << correction(L4,     L3  ) << ")" << "\n"
    << "  * L5(glu   ) : " << std::setw(10) << correction(L5g,   Raw  ) << " (" << std::setw(10) << correction(L5g,    L4  ) << ")" << "\n"
    << "  * L5(uds   ) : " << std::setw(10) << correction(L5uds, Raw  ) << " (" << std::setw(10) << correction(L5uds,  L4  ) << ")" << "\n"
    << "  * L5(charm ) : " << std::setw(10) << correction(L5c,   Raw  ) << " (" << std::setw(10) << correction(L5c,    L4  ) << ")" << "\n"
    << "  * L5(beauty) : " << std::setw(10) << correction(L5g,   Raw  ) << " (" << std::setw(10) << correction(L5b,    L4  ) << ")" << "\n"
    << "  * L6(glu   ) : " << std::setw(10) << correction(L6g,   Raw  ) << " (" << std::setw(10) << correction(L6g,   L5g  ) << ")" << "\n"
    << "  * L6(uds   ) : " << std::setw(10) << correction(L6uds, Raw  ) << " (" << std::setw(10) << correction(L6uds, L5uds) << ")" << "\n"
    << "  * L6(charm ) : " << std::setw(10) << correction(L6c,   Raw  ) << " (" << std::setw(10) << correction(L6c,   L5c  ) << ")" << "\n"
    << "  * L6(beauty) : " << std::setw(10) << correction(L6b,   Raw  ) << " (" << std::setw(10) << correction(L6b,   L5b  ) << ")" << "\n"
    << "  * L7(glu   ) : " << std::setw(10) << correction(L7g,   Raw  ) << " (" << std::setw(10) << correction(L7g,   L6g  ) << ")" << "\n"
    << "  * L7(uds   ) : " << std::setw(10) << correction(L7uds, Raw  ) << " (" << std::setw(10) << correction(L7uds, L6uds) << ")" << "\n"
    << "  * L7(charm ) : " << std::setw(10) << correction(L7c,   Raw  ) << " (" << std::setw(10) << correction(L7c,   L6c  ) << ")" << "\n"
    << "  * L7(beauty) : " << std::setw(10) << correction(L7b,   Raw  ) << " (" << std::setw(10) << correction(L7b,   L6b  ) << ")";
}
