#include "DetectorDescription/Parser/src/DDLEllipsoid.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLEllipsoid::DDLEllipsoid( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

void
DDLEllipsoid::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{ 
  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();
  double zbot(0.), ztop(0.);
  if( atts.find( "zBottomCut" ) != atts.end() )
  {
    zbot = ev.eval( nmspace, atts.find( "zBottomCut" )->second );
  }
  if( atts.find( "zTopCut" ) != atts.end() )
  {
    ztop = ev.eval( nmspace, atts.find( "zTopCut" )->second );
  }
  DDSolid ddel = DDSolidFactory::ellipsoid( getDDName( nmspace ),
					    ev.eval(nmspace, atts.find("xSemiAxis")->second),
					    ev.eval(nmspace, atts.find("ySemiAxis")->second),
					    ev.eval(nmspace, atts.find("zSemiAxis")->second),
					    zbot,
					    ztop );
  DDLSolid::setReference( nmspace, cpv );
}
