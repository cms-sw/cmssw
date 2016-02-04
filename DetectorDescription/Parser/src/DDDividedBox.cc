//
// ********************************************************************
// 25.04.04 - M. Case ddd-ize G4ParameterisationBox*
// ********************************************************************

#include "DetectorDescription/Parser/src/DDDividedBox.h"

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

#include "DetectorDescription/Base/interface/DDdebug.h"

#include <iomanip>

DDDividedBoxX::DDDividedBoxX( const DDDivision& div, DDCompactView* cpv)
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  setType( "DivisionBoxX" );
  DDBox mbox = (DDBox)(div_.parent().solid());

  if( divisionType_ == DivWIDTH ) {
    compNDiv_ = calculateNDiv( 2*mbox.halfX(), div_.width(), div_.offset() );
  } else if( divisionType_ == DivNDIV ) {
    compWidth_ = calculateWidth( 2*mbox.halfX(), div_.nReplicas(), div_.offset() );
  }
  
  // somehow here, I want to iterate over the nDivs or nReplicas 
  // and make the solid, make the logical part, then position it.
  // since DDBox is special, I will not make the solid and logical
  // part but once.  We'll see how bad the others go.
  // ihatethisihatethisihatethis
  //    for (int i = 0; i < compNDiv_; ++i)
  //      {
  //         DDpos(makeDDLogicalPart(i)  // child logical part
  //  	    , div_.parent()        // parent logical part
  //  	    , i                    // copy number
  //  	    , makeDDTranslation(i) // translation
  //  	    , makeDDRotation(i)    // rotation.  box is default (i.e. identity).
  //  	    );
  //      }

  DCOUT_V ('P', " DDDividedBoxX:DDDividedBoxX"  << std::endl);
}

DDDividedBoxX::~DDDividedBoxX( void )
{}

double
DDDividedBoxX::getMaxParameter( void ) const
{
  DDBox msol = (DDBox)(div_.parent().solid());
  return 2*msol.halfX();
}

DDRotation
DDDividedBoxX::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

DDTranslation
DDDividedBoxX::makeDDTranslation( const int copyNo ) const
{
  DDBox msol = (DDBox)(div_.parent().solid());
  double mdx = msol.halfX();

  //----- translation 
  DDTranslation translation;
  double posi = -mdx + div_.offset() + (copyNo+0.5) * compWidth_;
  translation.SetX( posi );

  DCOUT_V ('P', " DDDividedBoxX: " << copyNo << "\n Position " << translation << " Axis " << DDAxesNames::name(div_.axis()) << "\n");
  return translation;
}

DDLogicalPart
DDDividedBoxX::makeDDLogicalPart( const int copyNo ) const
{
  // in other cases, this solid will have 1, 2, 3, etc. after it.
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD", div_.parent().ddname().ns());
  DDSolid ddbox(solname);
  DDMaterial usemat(div_.parent().material());
  DDBox msol = (DDBox) (div_.parent().solid());
  DDLogicalPart ddlp(solname);
  if (!ddbox.isDefined().second) //This solid has NOT been defined. 
  {
    double pDx = msol.halfX();
    double pDy = compWidth_/2.;
    double pDz = msol.halfZ();
    
    ddbox = DDSolidFactory::box(solname, pDx, pDy, pDz);
    ddlp =  DDLogicalPart(solname, usemat, ddbox);
  }
  DCOUT_V ('P', " DDDividedBoxX::computeDimensions() lp:\n" << ddlp); 
  return ddlp;
}

DDDividedBoxY::DDDividedBoxY( const DDDivision& div, DDCompactView* cpv)
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  setType( "DivisionBoxY" );
  DDBox mbox = (DDBox)(div_.parent().solid());

  if( divisionType_ == DivWIDTH ) {
    compNDiv_ = calculateNDiv( 2*mbox.halfY(), div_.width(), div_.offset() );
  } else if( divisionType_ == DivNDIV ) {
    compWidth_ = calculateWidth( 2*mbox.halfY(), div_.nReplicas(), div_.offset() );
  } 

  
  // somehow here, I want to iterate over the nDivs or nReplicas 
  // and make the solid, make the logical part, then position it.
  // since DDBox is special, I will not make the solid and logical
  // part but once.  We'll see how bad the others go.
  // ihatethisihatethisihatethis
  //    for (int i = 0; i < compNDiv_; ++i)
  //      {
  //         DDpos(makeDDLogicalPart(i)  // child logical part
  //  	    , div_.parent()        // parent logical part
  //  	    , i                    // copy number
  //  	    , makeDDTranslation(i) // translation
  //  	    , makeDDRotation(i)    // rotation.  box is default (i.e. identity).
  //  	    );
  //      }

  DCOUT_V ('P', " DDDividedBoxY:DDDividedBoxY"  << std::endl);
}

DDDividedBoxY::~DDDividedBoxY( void )
{}

double
DDDividedBoxY::getMaxParameter( void ) const
{
  DDBox msol = (DDBox)(div_.parent().solid());
  return 2*msol.halfY();
}

DDRotation
DDDividedBoxY::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

DDTranslation
DDDividedBoxY::makeDDTranslation( const int copyNo ) const
{
  DDBox msol = (DDBox)(div_.parent().solid());
  double mdx = msol.halfY();

  //----- translation 
  DDTranslation translation;

  double posi = -mdx + div_.offset() + (copyNo+0.5) * compWidth_;
  translation.SetY( posi );

  DCOUT_V ('P', " DDDividedBoxY: " << copyNo << "\n Position " << translation << " Axis " << DDAxesNames::name(div_.axis()) << "\n");
  return translation;
}

DDLogicalPart
DDDividedBoxY::makeDDLogicalPart( const int copyNo ) const
{
  // in other cases, this solid will have 1, 2, 3, etc. after it.
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD", div_.parent().ddname().ns());
  DDSolid ddbox(solname);
  DDMaterial usemat(div_.parent().material());
  DDBox msol = (DDBox) (div_.parent().solid());
  DDLogicalPart ddlp(solname);
  if (!ddbox.isDefined().second) //This solid has NOT been defined. 
  {
    double pDx = msol.halfX();
    double pDy = compWidth_/2.;
    double pDz = msol.halfZ();
    ddbox = DDSolidFactory::box(solname, pDx, pDy, pDz);
    ddlp =  DDLogicalPart(solname, usemat, ddbox);
  }
  DCOUT_V ('P', " DDDividedBoxY::computeDimensions() lp:\n" << ddlp); 
  return ddlp;
}

DDDividedBoxZ::DDDividedBoxZ( const DDDivision& div, DDCompactView* cpv)
  : DDDividedGeometryObject::DDDividedGeometryObject( div, cpv )
{
  checkParametersValidity();
  setType( "DivisionBoxZ" );
  DDBox mbox = (DDBox)(div_.parent().solid());

  if( divisionType_ == DivWIDTH ) {
    compNDiv_ = calculateNDiv( 2*mbox.halfZ(), div_.width(), div_.offset() );
  } else if( divisionType_ == DivNDIV ) {
    compWidth_ = calculateWidth( 2*mbox.halfY(), div_.nReplicas(), div_.offset() );
  }
  
  // somehow here, I want to iterate over the nDivs or nReplicas 
  // and make the solid, make the logical part, then position it.
  // since DDBox is special, I will not make the solid and logical
  // part but once.  We'll see how bad the others go.
  // ihatethisihatethisihatethis
  //    for (int i = 0; i < compNDiv_; ++i)
  //      {
  //         DDpos(makeDDLogicalPart(i)  // child logical part
  //  	    , div_.parent()        // parent logical part
  //  	    , i                    // copy number
  //  	    , makeDDTranslation(i) // translation
  //  	    , makeDDRotation(i)    // rotation.  box is default (i.e. identity).
  //  	    );
  //      }

  DCOUT_V ('P', " DDDividedBoxZ:DDDividedBoxZ"  << std::endl);
}

DDDividedBoxZ::~DDDividedBoxZ( void )
{}

double
DDDividedBoxZ::getMaxParameter( void ) const
{
  DDBox msol = (DDBox)(div_.parent().solid());
  return 2*msol.halfZ();
}

DDRotation
DDDividedBoxZ::makeDDRotation( const int copyNo ) const
{
  return DDRotation();
}

DDTranslation
DDDividedBoxZ::makeDDTranslation( const int copyNo ) const
{
  DDBox msol = (DDBox)(div_.parent().solid());
  double mdx = msol.halfZ();

  //----- translation 
  DDTranslation translation;

  double posi = -mdx + div_.offset() + (copyNo+0.5) * compWidth_;
  translation.SetZ( posi );

  DCOUT_V ('P', " DDDividedBoxZ: " << copyNo << "\n Position " << translation << " Axis " << DDAxesNames::name(div_.axis()) << "\n");
  return translation;
}

DDLogicalPart
DDDividedBoxZ::makeDDLogicalPart( const int copyNo ) const
{
  // in other cases, this solid will have 1, 2, 3, etc. after it.
  DDName solname(div_.parent().ddname().name() + "_DIVCHILD", div_.parent().ddname().ns());
  DDSolid ddbox(solname);
  DDMaterial usemat(div_.parent().material());
  DDBox msol = (DDBox) (div_.parent().solid());
  DDLogicalPart ddlp(solname);
  if (!ddbox.isDefined().second) //This solid has NOT been defined. 
  {
    double pDx = msol.halfX();
    double pDy = msol.halfY();
    double pDz = compWidth_/2.;
    ddbox = DDSolidFactory::box(solname, pDx, pDy, pDz);
    ddlp =  DDLogicalPart(solname, usemat, ddbox);
  }
  DCOUT_V ('P', " DDDividedBoxZ::computeDimensions() lp:\n" << ddlp); 
  return ddlp;
}



