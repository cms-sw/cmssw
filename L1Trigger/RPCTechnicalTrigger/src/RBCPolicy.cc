/*
 *  See header file for a description of this class.
 *
 *
 *  $Date: 2007/07/06 11:59:46 $
 *  $Revision: 1.1 $
 *  \author M. Maggi, C. Viviani, D. Pagano - University of Pavia & INFN Pavia
 *
 */


#include "L1Trigger/RPCTechnicalTrigger/interface/RBCPolicy.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCPatternLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCChamberORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCEmulator.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace std;
using namespace edm;

RBCPolicy::RBCPolicy(RBCPolicy::Policy p, const edm::ParameterSet& pset) : pol(p), l(0)
{
  
  BX = pset.getUntrackedParameter<int>("BX");
  majority = pset.getUntrackedParameter<int>("majority");
  neighbours = pset.getUntrackedParameter<bool>("neighbours");

  //ChamberOR trigger the single sector
  //Neighbours trigger the first neighbours sectors
  
  if (neighbours == true){
    poly = 0;
    pol == "Neighbours";
  }
  
  else {
    poly = 1;
    pol == "ChamberOR";
  }
}

RBCPolicy::~RBCPolicy()
{ 
  std::cout<<"bye policy"<<std::endl;
  if (l)
    delete l;
  l=0;
}



std::string 
RBCPolicy::message()
{
  if (pol == Patterns){ 
    return "Patterns";
  }
  
  else if ( pol == ChamberOR ){
    return "ChamberOR";
  }
  
  else
    return "No idea";
}



RBCLogic* 
RBCPolicy::instance()
{
  if (l)
    delete l;
  l=0;
  if (pol == Patterns)
    l = new RBCPatternLogic();
  else if ( pol == ChamberOR )
    l =  new RBCChamberORLogic();
  return l;
}
