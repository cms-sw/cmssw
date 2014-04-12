//   COCOA class implementation file
//Id: DeviationSensor2D.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/DeviationSensor2D.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

DeviationSensor2D::DeviationSensor2D( ALIdouble posDimFactor, ALIdouble angDimFactor )
{ 
  thePosDimFactor = posDimFactor;
  theAngDimFactor = angDimFactor;
}

void DeviationSensor2D::fillData( const std::vector<ALIstring>& wl )
{
  if( wl.size() != 8 ) {
    ALIUtils::dumpVS( wl,  "!!!! EXITING DeviationsSensor2D::fillData. Number of words <> 8 ", std::cerr );
  }

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  thePosDimFactor = gomgr->GlobalOptions()[ ALIstring("deviffValDimf") ];
  theAngDimFactor = gomgr->GlobalOptions()[ ALIstring("deviffAngDimf") ];
  thePosX = ALIUtils::getFloat( wl[0] )*thePosDimFactor;
  thePosErrX = ALIUtils::getFloat( wl[1] );
  thePosY = ALIUtils::getFloat( wl[2] )*thePosDimFactor;
  thePosErrY = ALIUtils::getFloat( wl[3] );
  theDevX = ALIUtils::getFloat( wl[4] )*theAngDimFactor;
  theDevErrX = ALIUtils::getFloat( wl[5] );
  theDevY = ALIUtils::getFloat( wl[6] )*theAngDimFactor;
  theDevErrY = ALIUtils::getFloat( wl[7] );

}




