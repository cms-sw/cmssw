//   COCOA class implementation file
//Id:  EntryLengthAffAngles.C
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/EntryAngleAffAngles.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
EntryAngleAffAngles::EntryAngleAffAngles( const ALIstring& type )
  : EntryAngle( type )
{
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void EntryAngleAffAngles::FillName( const ALIstring& name )
{
  ALIstring nn = "Angles ";
  nn += name;
  setName( nn );
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void EntryAngleAffAngles::displace( ALIdouble disp )
{
  XYZcoor coor = XCoor;
  ALIint namelength = name().length()-1;
  //-  std::cout << this << "ENtryAnglesAffAngle" << name_ << namelength <<std::endl;
  if ( name_[namelength] == 'X' ) {
    coor = XCoor;
  } else if ( name_[namelength] == 'Y' ) {
    coor = YCoor;
  } else if ( name_[namelength] == 'Z' ) {
    coor = ZCoor;
  }

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if(gomgr->GlobalOptions()["rotateAroundLocal"] == 0) {
    OptOCurrent()->displaceRmGlobAroundGlobal( OptOCurrent(), coor, disp );
  }else {
    OptOCurrent()->displaceRmGlobAroundLocal( OptOCurrent(), coor, disp );
  }
 
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void EntryAngleAffAngles::displaceOriginal( ALIdouble disp )
{
  ALIint namelength = name().length()-1;
  //-  std::cout << this << "ENtryAnglesAffAngle" << name_ << namelength <<std::endl;
  if ( name_[namelength] == 'X' ) {
    //-    std::cout << "displaX";
      OptOCurrent()->displaceRmGlobOriginal( OptOCurrent(), XCoor, disp );
  } else if ( name_[namelength] == 'Y' ) {
    //-    std::cout << "displaY";
      OptOCurrent()->displaceRmGlobOriginal( OptOCurrent(), YCoor, disp );
  } else if ( name_[namelength] == 'Z' ) {
      OptOCurrent()->displaceRmGlobOriginal( OptOCurrent(), ZCoor, disp );
  }
 
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void EntryAngleAffAngles::displaceOriginalOriginal( ALIdouble disp )
{
  ALIint namelength = name().length()-1;
  if(ALIUtils::debug >= 5) std::cout << this << "ENtryAnglesAffAngle displaceOriginalOriginal" << name_ <<std::endl;
  if ( name_[namelength] == 'X' ) {
    //-    std::cout << "displaX";
      OptOCurrent()->displaceRmGlobOriginalOriginal( OptOCurrent(), XCoor, disp );
  } else if ( name_[namelength] == 'Y' ) {
    //-    std::cout << "displaY";
      OptOCurrent()->displaceRmGlobOriginalOriginal( OptOCurrent(), YCoor, disp );
  } else if ( name_[namelength] == 'Z' ) {
      OptOCurrent()->displaceRmGlobOriginalOriginal( OptOCurrent(), ZCoor, disp );
  }
 
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble EntryAngleAffAngles::valueDisplaced() const
{

  //  std::cout << "  EntryAngleAffAngles::valueDisplaced() parent " << OptOCurrent()->parent()->name() << std::endl;
  //  ALIUtils::dumprm( OptOCurrent()->parent()->rmGlob() , " parent RmGlob ");
  if(ALIUtils::debug >= 5) {
    std::cout << "  EntryAngleAffAngles::valueDisplaced() parent opto  " << OptOCurrent()->parent()->name() << std::endl;
    ALIUtils::dumprm( OptOCurrent()->rmGlob() , " RmGlob ");
    ALIUtils::dumprm( OptOCurrent()->rmGlobOriginal() , " RmGlobOriginal ");
    ALIUtils::dumprm( OptOCurrent()->parent()->rmGlobOriginal() , " parent RmGlobOriginal ");
  }

  CLHEP::HepRotation diffRm =  OptOCurrent()->rmGlob() * OptOCurrent()->rmGlobOriginal().inverse();
  CLHEP::HepRotation rmLocal = diffRm * OptOCurrent()->parent()->rmGlobOriginal().inverse();
  std::vector<double> localrot = OptOCurrent()->getRotationAnglesFromMatrix( rmLocal, OptOCurrent()->CoordinateEntryList() );
  if(ALIUtils::debug >= 5) {
    ALIUtils::dumprm( diffRm , " diffRm ");
    ALIUtils::dumprm( rmLocal , " rmLocal ");
    std::cout << " localrot " << localrot[0] << " " << localrot[1] << " " << localrot[2] << std::endl;
  }

  if( name() == "angles_X" ) {
    return localrot[0];
  }else if( name() == "angles_Y" ) {
    return localrot[1];
  }else if( name() == "angles_Z" ) {
    return localrot[2];
  }

  /*
  CLHEP::HepRotation rmLocalOrig = OptOCurrent()->parent()->rmGlobOriginal().inverse() *  OptOCurrent()->rmGlobOriginal();
  
  CLHEP::HepRotation rmLocal = OptOCurrent()->parent()->rmGlob().inverse() *  OptOCurrent()->rmGlob();
  std::vector<double> localrot = OptOCurrent()->getRotationAnglesFromMatrix(  rmLocal, OptOCurrent()->CoordinateEntryList() );

  std::cout << " localrot " << localrot[0] << " " << localrot[1] << " " << localrot[2] << std::endl;
  std::cout << " localrotorig " << localrotorig[0] << " " << localrotorig[1] << " " << localrotorig[2] << std::endl;
  ALIdouble diff;
  CLHEP::Hep3Vector Xaxis(0.,0.,1.);
  Xaxis = OptOCurrent()->parent()->rmGlob() * Xaxis;
  CLHEP::Hep3Vector XaxisOrig(0.,0.,1.);
  XaxisOrig = OptOCurrent()->parent()->rmGlobOriginal() * XaxisOrig;

  diff = fabs( checkDiff( Xaxis, XaxisOrig, localrot, localrotorig ) );

  //maybe X is not a good axis because the rotation is done precisely around X
  if( diff <= 1.E-9 ){
    CLHEP::Hep3Vector Yaxis(0.,1.,0.);
    Yaxis = OptOCurrent()->parent()->rmGlob() * Yaxis;
    CLHEP::Hep3Vector YaxisOrig(0.,1.,0.);
    YaxisOrig = OptOCurrent()->parent()->rmGlobOriginal() * YaxisOrig;
    
    diff = fabs( checkDiff( Yaxis, YaxisOrig, localrot, localrotorig ) );
  }

  return diff;
  */

  return 0.; // to avoid warning
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble EntryAngleAffAngles::checkDiff( CLHEP::Hep3Vector axis, CLHEP::Hep3Vector axisOrig, std::vector<double> localrot, std::vector<double> localrotorig ) const
{
  int inam = 0;
  if( name() == "angles_X" ) {
    inam = 1;
  }else if( name() == "angles_Y" ) {
    inam = 2;
  }else if( name() == "angles_Z" ) {
    inam = 3;
  }
  switch( inam ){
  case 1:
    axis.rotateX( localrot[0] );
    axisOrig.rotateX( localrotorig[0] );
  case 2:
    axis.rotateY( localrot[1] );
    axisOrig.rotateY( localrotorig[1] );
  case 3:
    axis.rotateZ( localrot[2] );
    axisOrig.rotateZ( localrotorig[2] );
    break;
  }

  ALIdouble ang = axis.angle( axisOrig );
 
    /*  }else if( name() == "angles_Y" ) {
    return localrot[1] - localrotorig[1];
  }else if( name() == "angles_Z" ) {
    return localrot[2] - localrotorig[2];
    }*/
  
  return ang;
}

 
