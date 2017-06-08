#include "DetectorDescription/Parser/src/DDLMultiUnionSolid.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "FWCore/Utilities/interface/Exception.h"

class DDCompactView;

DDLMultiUnionSolid::DDLMultiUnionSolid( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

// Clear out rSolids.
void
DDLMultiUnionSolid::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  myRegistry_->getElement( "rSolid" )->clear();
}

// To process a MultiUnionSolid we should have in the meantime
// hit two rSolid calls and possibly one rRotation and one Translation.
// So, retrieve them and make the call to DDCore.
void
DDLMultiUnionSolid::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  // new DDLMultiUnion will handle:
   // <MultiUnionSolid> <rSolid...> <Translation...> <rRotation...> <rSolid...> <Translation...> <rRotation...> ... </MultiUnionSolid>

  auto myrSolids = myRegistry_->getElement( "rSolid" ); // get rSolid children
  auto myTranslations = myRegistry_->getElement( "Translation" ); // get Translation child
  auto myrRotations  = myRegistry_->getElement( "rRotation" ); // get rRotation child

  ClhepEvaluator & ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  std::vector<DDSolid> solids;
  std::vector<DDTranslation> translations;
  std::vector<DDRotation> rotations;
  for( unsigned int i = 0; i < myrSolids->size(); ++i )
  {
    DDName ddname = myrSolids->getDDName( nmspace, "name", i );
    solids.emplace_back( DDSolid( ddname ));
  }
  for( unsigned int i = 0; i < myTranslations->size(); ++i )
  {
    double x = 0.0, y = 0.0, z = 0.0;
    atts = myTranslations->getAttributeSet(i);
    x = ev.eval(nmspace, atts.find("x")->second);
    y = ev.eval(nmspace, atts.find("y")->second);
    z = ev.eval(nmspace, atts.find("z")->second);
    translations.emplace_back( DDTranslation( x, y, z ));
  }
  for( unsigned int i = 0; i < myrRotations->size(); ++i )
  {
    DDRotation ddrot = myrRotations->getDDName( nmspace, "name", i );
    rotations.emplace_back( ddrot );
  }
  
  // Basically check if there are rSolids or Translation or rRotation then we have
  // should NOT have any of the attributes shown above.
  if( myrSolids->size() == 0 ) 
  {
    std::string s( "DDLMultiUnionSolid did not find any solids with which to form a multi union solid." );
    s += dumpMultiUnionSolid(name, nmspace);
    throwError( s );
  }

  DDSolid theSolid;

  if (name == "MultiUnionSolid") {
    theSolid = DDSolidFactory::multiUnionSolid( getDDName( nmspace ),
						solids,
						translations,
						rotations );
  }
  else {
    throw cms::Exception("DDException") << "DDLMultiUnionSolid was asked to do something other than MultiUnion-?";
  }
  
  DDLSolid::setReference( nmspace, cpv );

  // clear all "children" and attributes
  myTranslations->clear();
  myrRotations->clear();
  myrSolids->clear();
  clear();
}

std::string
DDLMultiUnionSolid::dumpMultiUnionSolid( const std::string& name, const std::string& nmspace )
{
  std::string s;
  DDXMLAttribute atts = getAttributeSet();

  s = std::string ("\n<") + name + " name=\"" + atts.find("name")->second + "\"";
  s +=  ">\n";

  auto myrSolids = myRegistry_->getElement("rSolid"); // get rSolid children
  auto myTranslations = myRegistry_->getElement("Translation"); // get Translation child
  auto myrRotations  = myRegistry_->getElement("rRotation"); // get rRotation child
  if( myrSolids->size() > 0 )
  {
    for( size_t i = 0; i < myrSolids->size(); ++i )
    {
      atts = myrSolids->getAttributeSet(i);
      s+="<rSolid name=\"" + atts.find("name")->second + "\"/>\n";
    }
  }

  for( size_t i = 0; i < myTranslations->size(); ++i )
  {
    atts = myTranslations->getAttributeSet(i);
    s+= "<Translation";
    if (atts.find("x") != atts.end()) 
      s+=" x=\"" + atts.find("x")->second + "\"";
    if (atts.find("y") != atts.end()) 
      s+= " y=\"" + atts.find("y")->second + "\"";
    if (atts.find("z") != atts.end()) 
      s+= " z=\"" + atts.find("z")->second + "\"";
    s+="/>\n";
  }
  
  for( size_t i = 0; i < myrRotations->size(); ++i )
  {
    atts = myrRotations->getAttributeSet(i);
    if (atts.find("name") != atts.end())
    {
      s+= "<rRotation name=\"" + atts.find("name")->second + "\"/>\n";
    }
    s+= "</" + name + ">\n\n";
  }

  return s;
}
