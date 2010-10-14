/***************************************************************************
                          DDLLogicalPart.cc  -  description
                             -------------------
    begin                : Tue Oct 31 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLLogicalPart.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

DDLLogicalPart::DDLLogicalPart(  DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{
  // initialize category map
  catMap_["sensitive"]   = DDEnums::sensitive;
  catMap_["cable"]       = DDEnums::cable;
  catMap_["cooling"]     = DDEnums::cooling;
  catMap_["support"]     = DDEnums::support;
  catMap_["envelope"]    = DDEnums::envelope;
  catMap_["unspecified"] = DDEnums::unspecified;
}

DDLLogicalPart::~DDLLogicalPart( void )
{}

// upon initialization, we want to clear rMaterial and rSolid.
void
DDLLogicalPart::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  myRegistry_->getElement( "rMaterial" )->clear();
  myRegistry_->getElement( "rSolid" )->clear();
}

// Upon encountering the end of the LogicalPart element, retrieve all 
// relevant information from its sub-elements and put it all together to 
// call the DDCore appropriately.
// 
// History:  Initially only rSolid and rMaterial elements worked.  Now,
// a Material or a Solid element inside the LogicalPart also works but
// this is handled outside this class.  Each Solid inherits from DDLSolid
// and in each Solid, the processElement method must call the setReference
// of the DDLSolid.  The Material element also works similarly.  Thus,
// by retrieving the rMaterial and the rSolid it actually will be handling
// Material and Solid subelements as well.

void
DDLLogicalPart::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P', "DDLLogicalPart::processElement started");

  // rMaterial and rSolid  
  DDXMLElement* myrMaterial = myRegistry_->getElement("rMaterial"); // get Material reference child
  DDXMLElement* myrSolid = myRegistry_->getElement("rSolid"); // get Solid reference child

  DDXMLAttribute atts = getAttributeSet();

  // this check really is overkill so I'm commenting it out for now.
  // validation of the XML should PREVENT this.
//    if (myrSolid->size() > 1)
//      {
//        std::string s = "DDLLogicalPart::processElement:  When looking at rSolid, found more than one. ";
//        s += " Logical part name was: ";
//        s += atts.find("name")->second;
//        throw DDException(s);
//      }

  DDSolid mySolid = DDSolid(myrSolid->getDDName(nmspace));
  DDMaterial myMaterial = DDMaterial(myrMaterial->getDDName(nmspace));

  DDEnums::Category cat;

  if (atts.find("category") != atts.end())
    cat = catMap_[atts.find("category")->second];
  else
    cat = catMap_["unspecified"];

  DDLogicalPart lp( getDDName( nmspace ), myMaterial, mySolid, cat );

  // clear all "children" and attributes
  myrMaterial->clear();
  myrSolid->clear();

  // after each logical part is made, we can clear it
  clear();

  DCOUT_V('P', "DDLLogicalPart::processElement completed");
}
