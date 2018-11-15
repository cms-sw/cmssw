#include "DetectorDescription/Parser/src/DDLRotationByAxis.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

#include <map>
#include <utility>
#include <vector>

class DDCompactView;

DDLRotationByAxis::DDLRotationByAxis( DDLElementRegistry* myreg )
  : DDXMLElement( myreg ) 
{}

void
DDLRotationByAxis::preProcessElement( const std::string& name, const std::string& nmspace,
				      DDCompactView& cpv )
{
  pNameSpace = nmspace;
  pName = name;
}

void
DDLRotationByAxis::processElement( const std::string& name, const std::string& nmspace,
				   DDCompactView& cpv )
{
  DDXMLAttribute atts = getAttributeSet();
  if (parent() != "RotationSequence")
  {
    std::string axis = atts.find("axis")->second;
    std::string angle = atts.find("angle")->second;
      
    DDRotationMatrix R;
    R = processOne(R, axis, angle);

    if( atts.find( "name" ) == atts.end())
    {
      auto myRealParent = myRegistry_->getElement( parent());
      DDName pName = myRealParent->getDDName( nmspace );
      std::string tn = pName.name() + std::string( "Rotation" );
      std::vector<std::string> names;
      names.emplace_back( "name" );

      std::vector<std::string> values;
      values.emplace_back( tn );

      clear();
      loadAttributes( name, names, values, nmspace, cpv );
    }
    DDRotation rot = DDrot( getDDName( nmspace ), std::make_unique<DDRotationMatrix>( R ));
    
    clear();
  }
}

DDRotationMatrix
DDLRotationByAxis::processOne( DDRotationMatrix R, std::string& axis, std::string& angle )
{
  /** Get the name, axis and angle of the RotationByAxis and do it. 
   */
  
  ClhepEvaluator & ev = myRegistry_->evaluator();
  double dAngle = ev.eval(pNameSpace, angle);

  if ( axis == "x") {
    R = ROOT::Math::RotationX(dAngle);
  }
  else if ( axis == "y" ) {
    R = ROOT::Math::RotationY(dAngle);      
  }
  else if ( axis =="z" ) {
    R = ROOT::Math::RotationZ(dAngle);
  }
  else {
    std::string msg = "\nDDLRotationByAxis invalid axis... you must not have validated XML sources!  Element is ";
    msg += pName;
    throwError(msg);
  }
  
  return R;
}
