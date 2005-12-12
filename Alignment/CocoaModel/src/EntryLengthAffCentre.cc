//   COCOA class implementation file
//Id:  EntryLengthAffCentre.C
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "OpticalAlignment/CocoaModel/interface/EntryLengthAffCentre.h"
#include "OpticalAlignment/CocoaModel/interface/OpticalObject.h"
#include "OpticalAlignment/CocoaUtilities/interface/ALIUtils.h"


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
EntryLengthAffCentre::EntryLengthAffCentre( const ALIstring& type )
  : EntryLength( type )
{
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void EntryLengthAffCentre::FillName( const ALIstring& name )
{
  ALIstring nn = "Centre ";
  nn += name;
  setName( nn );
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void EntryLengthAffCentre::displace( ALIdouble disp )
{
  if(ALIUtils::debug>=9) std::cout << "EntryLengthAffCentre::Displace" <<  disp <<std::endl;
  ALIint namelength = name().length()-1;
  XYZcoor axisNo = XCoor;
  if ( _name[namelength] == 'X' ) {
     axisNo = XCoor;
  } else if ( _name[namelength] == 'Y' ) {
     axisNo = YCoor;
  } else if ( _name[namelength] == 'Z' ) {
     axisNo = ZCoor;
  }
  OptOCurrent()->displaceCentreGlob( axisNo, disp );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void EntryLengthAffCentre::displaceOriginal( ALIdouble disp )
{
  if(ALIUtils::debug>=9) std::cout << "EntryLengthAffCentre::DisplaceOriginal" <<  disp <<std::endl;
  ALIint namelength = name().length()-1;
  if ( _name[namelength] == 'X' ) {
      OptOCurrent()->displaceCentreGlobOriginal( XCoor, disp );
  } else if ( _name[namelength] == 'Y' ) {
      OptOCurrent()->displaceCentreGlobOriginal( YCoor, disp );
  } else if ( _name[namelength] == 'Z' ) {
      OptOCurrent()->displaceCentreGlobOriginal( ZCoor, disp );
  }

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void EntryLengthAffCentre::displaceOriginalOriginal( ALIdouble disp )
{
  if(ALIUtils::debug>=9) std::cout << "EntryLengthAffCentre::DisplaceOriginalOriginal" <<  disp <<std::endl;
  ALIint namelength = name().length()-1;
  if ( _name[namelength] == 'X' ) {
      OptOCurrent()->displaceCentreGlobOriginalOriginal( XCoor, disp );
  } else if ( _name[namelength] == 'Y' ) {
      OptOCurrent()->displaceCentreGlobOriginalOriginal( YCoor, disp );
  } else if ( _name[namelength] == 'Z' ) {
      OptOCurrent()->displaceCentreGlobOriginalOriginal( ZCoor, disp );
  }

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble EntryLengthAffCentre::valueInGlobalReferenceFrame() const
{
  ALIdouble a=2.;

  return a;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble EntryLengthAffCentre::valueDisplaced() const
{
  ALIdouble vdisp = 0.;

  CLHEP::Hep3Vector cdisp = OptOCurrent()->centreGlob() - OptOCurrent()->centreGlobOriginal();
  CLHEP::HepRotation rmParentInv = inverseOf(  OptOCurrent()->parent()->rmGlob() );
  cdisp = rmParentInv * cdisp;

  if( name() == "centre_X" ) {
    return cdisp.x();
  }else if( name() == "centre_Y" ) {
    return cdisp.y();
   //-   return OptOCurrent()->centreLocal().y() - value();
  }else if( name() == "centre_Z" ) {
    return cdisp.z();
  }

  if ( ALIUtils::debug >= 5 ) std::cout << name() << " in OptO " << OptOCurrent()->name() << " valueDisplaced: " << vdisp << std::endl;

}
