/***************************************************************************
                          DDLRotationByAxis.cc  -  description
                             -------------------
    begin                : Wed Nov 19, 2003
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLRotationByAxis.h"

#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <Math/RotationX.h>
#include <Math/RotationY.h>
#include <Math/RotationZ.h>

DDLRotationByAxis::DDLRotationByAxis( DDLElementRegistry* myreg )
  : DDXMLElement( myreg ) 
{}

DDLRotationByAxis::~DDLRotationByAxis( void )
{}

void
DDLRotationByAxis::preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  pNameSpace = nmspace;
  pName = name;
}

void
DDLRotationByAxis::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P', "DDLRotationByAxis::processElement started " << name);

  //  std::cout << "my parent is : " << parent() << std::endl;
  DDXMLAttribute atts = getAttributeSet();
  if (parent() != "RotationSequence")
  {
    std::string axis = atts.find("axis")->second;
    std::string angle = atts.find("angle")->second;
      
    DDRotationMatrix R;
    R = processOne(R, axis, angle);

    DDRotationMatrix* ddr = new DDRotationMatrix(R);
    if (atts.find("name") == atts.end())
    {
      //how do we make up a ddname! damn_it!
      //          DDXMLElement * myRealParent = DDLElementRegistry::instance()->getElement(parent());
      DDXMLElement * myRealParent = myRegistry_->getElement(parent());
      DDName pName = myRealParent->getDDName(nmspace);
      std::string tn = pName.name() + std::string("Rotation");
      std::vector<std::string> names;
      names.push_back("name");
      //no need, used already names.push_back("axis");
      //no need, used already names.push_back("angle");

      std::vector<std::string> values;
      values.push_back(tn);
      //no need, used already values.push_back(atts.find("axis")->second);
      //no need, used already values.push_back(atts.find("angle")->second);
      clear();
      loadAttributes(name, names, values, nmspace, cpv);
    }
    DDRotation rot = DDrot(getDDName(nmspace), ddr);
      
    clear();
  }
  else { } //let the parent handle the clearing, etc.

  DCOUT_V('P', "DDLRotationByAxis::processElement completed");
}

DDRotationMatrix
DDLRotationByAxis::processOne( DDRotationMatrix R, std::string& axis, std::string& angle )
{
  /** Get the name, axis and angle of the RotationByAxis and do it. 
   */
  
  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  double dAngle = ev.eval(pNameSpace, angle);
  //  CLHEP::HepRotation R;

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
