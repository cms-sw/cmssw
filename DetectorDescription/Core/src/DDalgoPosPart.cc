

//#include <cstdio>
//#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDalgoPosPart.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDAlgo.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoPos.h"

// Message logger.

void DDalgoPosPart(const DDName & self,
                   const DDName & parent,
		   DDAlgo & algo
		   )
{
   DDalgoPosPart(DDLogicalPart(self), 
                 DDLogicalPart(parent),
		 algo);
}		   		   


void DDalgoPosPart(const DDLogicalPart & self,
                   const DDLogicalPart & parent,
		   DDAlgo & algo
		   )
{
  static int cnt_=0;
  ++cnt_;
  if (algo.rep().numRegistered() == 0) {
    std::string e;
    e = "DDalgoPosPart: algorithmic positioning\n";
    e += "\t[" + algo.name().ns() 
               + ":" 
	       + algo.name().name() 
	       + "] is not defined!\n";
    throw DDException(e);

//    if (!algo) {
//      std::string e;
//      e  = "DDalgoPosPart: algorithmic positioning\n";
//      e += "\t[" + algo.ns() + ":" + algo.name() + "] is not defined!\n";
//      throw DDException(e);
  }
  
  //LogDebug ("AlgoPos")  << "AlgoPos, algo=" << std::endl << algo << std::endl;
  int inner=0;
  do { 
    //FIXME: DDalgoPosPart: better solution for rotation matrices!!!
    //FIXME:                find out about unit-rotations!
    /*
    ++inner;
    char buf_ns [20];
    char buf_nm [20];
    sprintf(buf_ns,"%d",cnt_) ; sprintf(buf_nm,"%d",inner);
    std::string ns(buf_ns)         ; std::string nm(buf_nm);
    */
    ++inner;
    DDRotationMatrix * rmp = new DDRotationMatrix(algo.rotation());
    DDRotation anonymRot = DDanonymousRot(rmp);
    DDpos(self.ddname(), parent.ddname(), 
          algo.label(), algo.translation(), anonymRot); 
    algo.next();
  } 
  while(algo.go());
}				   		   
		   

