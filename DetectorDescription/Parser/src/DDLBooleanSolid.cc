/***************************************************************************
                          DDLBooleanSolid.cc  -  description
                             -------------------
    begin                : Wed Dec 12, 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLBooleanSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

DDLBooleanSolid::DDLBooleanSolid( DDLElementRegistry* myreg )
  : DDLSolid( myreg )
{}

DDLBooleanSolid::~DDLBooleanSolid( void )
{}

// Clear out rSolids.
void
DDLBooleanSolid::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  myRegistry_->getElement( "rSolid" )->clear();
}

// To process a BooleanSolid we should have in the meantime
// hit two rSolid calls and possibly one rRotation and one Translation.
// So, retrieve them and make the call to DDCore.
void
DDLBooleanSolid::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V( 'P', "DDLBooleanSolid::processElement started" );

  // new DDLBoolean will handle:
  // <UnionSolid name="bs" firstSolid="blah" secondSolid="argh"> <Translation...> <rRotation .../> </UnionSolid
  // AND <UnionSolid> <rSolid...> <rSolid...> <Translation...> <rRotation...> </UnionSolid>

  DDXMLElement* myrSolid = myRegistry_->getElement( "rSolid" ); // get rSolid children
  DDXMLElement* myTranslation = myRegistry_->getElement( "Translation" ); // get Translation child
  DDXMLElement* myrRotation  = myRegistry_->getElement( "rRotation" ); // get rRotation child

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  DDXMLAttribute atts = getAttributeSet();

  DDName ddn1, ddn2;
  double x=0.0, y=0.0, z=0.0;
  DDRotation ddrot;

  // Basically check if there are rSolids or Translation or rRotation then we have
  // should NOT have any of the attributes shown above.
  if( myrSolid->size() == 0 ) 
  {
    // do the solids using the attributes only.
    if ( atts.find("firstSolid") != atts.end() && atts.find("secondSolid") != atts.end() ) {
      ddn1 = getDDName(nmspace, "firstSolid");
      ddn2 = getDDName(nmspace, "secondSolid");
    } else {
      std::string s ("DDLBooleanSolid did not find any solids with which to form a boolean solid.");
      s += dumpBooleanSolid(name, nmspace);
      throwError( s );
    }
  }
  else if (myrSolid->size() == 2)
  {
    ddn1 = myrSolid->getDDName(nmspace, "name", 0);
    ddn2 = myrSolid->getDDName(nmspace, "name", 1);
  } else {
    std::string s("DDLBooleanSolid did not find any solids with which to form a boolean solid.");
    s += dumpBooleanSolid(name, nmspace);
    throwError( s );
  }

  if (myTranslation->size() > 0)
  {
    atts.clear();
    atts = myTranslation->getAttributeSet();
    x = ev.eval(nmspace, atts.find("x")->second);
    y = ev.eval(nmspace, atts.find("y")->second);
    z = ev.eval(nmspace, atts.find("z")->second);
  }

  if (myrRotation->size() > 0) 
  {
    ddrot = DDRotation( myrRotation->getDDName (nmspace) );
  }

  DDSolid theSolid;

  if (name == "UnionSolid") {
    theSolid = DDSolidFactory::unionSolid (getDDName(nmspace)
					   , DDSolid(ddn1)
					   , DDSolid(ddn2)
					   , DDTranslation(x, y, z)
					   , ddrot
      );	       
  }
  else if (name == "SubtractionSolid") {
    theSolid = DDSolidFactory::subtraction (getDDName(nmspace)
					    , DDSolid(ddn1)
					    , DDSolid(ddn2)
					    , DDTranslation(x, y, z)
					    , ddrot
      );	       
  }
  else if (name == "IntersectionSolid") {
    theSolid = DDSolidFactory::intersection (getDDName(nmspace)
					     , DDSolid(ddn1)
					     , DDSolid(ddn2)
					     , DDTranslation(x, y, z)
					     , ddrot
      );	       
  }
  else {
    throw cms::Exception("DDException") << "DDLBooleanSolid was asked to do something other than Union-, Subtraction- or IntersectionSolid?";
  }
  
  DDLSolid::setReference(nmspace, cpv);

  DCOUT_V('p', theSolid);

  // clear all "children" and attributes
  myTranslation->clear();
  myrRotation->clear();
  myrSolid->clear();
  clear();
  DCOUT_V('P', "DDLBooleanSolid::processElement completed");

}

// This only happens on error, so I don't care how "slow" it is :-)
std::string
DDLBooleanSolid::dumpBooleanSolid( const std::string& name, const std::string& nmspace )
{
  std::string s;
  DDXMLAttribute atts = getAttributeSet();

  s = std::string ("\n<") + name + " name=\"" + atts.find("name")->second + "\"";

  if (atts.find("firstSolid") != atts.end()) s+= " firstSolid=\"" + atts.find("firstSolid")->second + "\"";
  if (atts.find("secondSolid") != atts.end()) s+= " secondSolid=\"" + atts.find("secondSolid")->second + "\"";
  s +=  ">\n";

  DDXMLElement* myrSolid = myRegistry_->getElement("rSolid"); // get rSolid children
  DDXMLElement* myTranslation = myRegistry_->getElement("Translation"); // get Translation child
  DDXMLElement* myrRotation  = myRegistry_->getElement("rRotation"); // get rRotation child
  if (myrSolid->size() > 0)
  {
    for (size_t i = 0; i < myrSolid->size(); ++i)
    {
      atts = myrSolid->getAttributeSet(i);
      s+="<rSolid name=\"" + atts.find("name")->second + "\"/>\n";
    }
  }

  atts = myTranslation->getAttributeSet();
  s+= "<Translation";
  if (atts.find("x") != atts.end()) 
    s+=" x=\"" + atts.find("x")->second + "\"";
  if (atts.find("y") != atts.end()) 
    s+= " y=\"" + atts.find("y")->second + "\"";
  if (atts.find("z") != atts.end()) 
    s+= " z=\"" + atts.find("z")->second + "\"";
  s+="/>\n";

  atts = myrRotation->getAttributeSet();
  if (atts.find("name") != atts.end())
  {
    s+= "<rRotation name=\"" + atts.find("name")->second + "\"/>\n";
  }
  s+= "</" + name + ">\n\n";
  return s;
}
