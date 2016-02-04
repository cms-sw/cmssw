/***************************************************************************
                          DDLAlgoPosPart.cc  -  description
                             -------------------
    begin                : Wed Apr 17 2002
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLAlgoPosPart.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDAlgo.h"
#include "DetectorDescription/Base/interface/DDAlgoPar.h"
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <iostream>

DDLAlgoPosPart::DDLAlgoPosPart( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

DDLAlgoPosPart::~DDLAlgoPosPart( void )
{}

// Upon encountering the end tag of the AlgoPosPart we should have in the meantime
// hit rParent, rChild, ParS and ParE.
void
DDLAlgoPosPart::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P', "DDLAlgoPosPart::processElement started");
  
  // get all internal elements.
  DDXMLElement* myParent  = myRegistry_->getElement("rParent");
  DDXMLElement* myChild   = myRegistry_->getElement("rChild");
  DDXMLElement* myParS    = myRegistry_->getElement("ParS");
  DDXMLElement* myParE    = myRegistry_->getElement("ParE");

  ExprEvalInterface & ev = ExprEvalSingleton::instance();
  
  DDXMLAttribute atts = getAttributeSet();

  // these were doubles
  int st = static_cast<int> ((atts.find("start") == atts.end() ? 0.0 : ev.eval(nmspace, atts.find("start")->second)));
  int ic = static_cast<int> ((atts.find("incr") == atts.end() ? 0.0 : ev.eval(nmspace, atts.find("incr")->second)));
  int ed = static_cast<int> ((atts.find("end") == atts.end() ? 0.0 : ev.eval(nmspace, atts.find("end")->second)));
  
  // get actual DDLogicalPart objects.
  DDLogicalPart parent(DDName(myParent->getDDName(nmspace)));
  DDLogicalPart self(DDName(myChild->getDDName(nmspace)));

  // get the algorithm
  DDAlgo algo( getDDName(nmspace, "algo" ));
  if (!(algo.isDefined().second)) 
  {
    std::string  msg = std::string("\n\tDDLParser, algo requested is not defined.  Either AlgoInit() or check algo spelling.\n ")
		       + "\n\t\talgo=" + std::string(getDDName(nmspace, "algo" ))
		       + "\n\t\tparent=" + std::string(myParent->getDDName(nmspace))
		       + "\n\t\tself=" + std::string(myChild->getDDName(nmspace));
    throwError(msg);
  }

  // set the parameters for the algorithm

  // First for ParE type
  parE_type parE;
  for (size_t i = 0; i < myParE->size(); ++i)
  {
    atts = myParE->getAttributeSet(i);
    // find vname in ParE.
    parE_type::iterator existingName=parE.find(atts.find("name")->second);
      
    // if found, get std::vector, then add this value to it.
    // if not found, add this var, then add a value to it.
    if (existingName != parE.end())
      existingName->second.push_back(ev.eval(nmspace,atts.find("value")->second));
    //	tvect = existingName->second;
    else
    {
      std::vector<double> tvect;
      tvect.push_back(ev.eval(nmspace,atts.find("value")->second));
      parE[atts.find("name")->second] = tvect;
    }
  }

  // Now for ParS type
  parS_type parS;

  for (size_t i = 0; i < myParS->size(); ++i)
  {
    atts = myParS->getAttributeSet(i);

    // find vname in ParS.
    parS_type::iterator existingName=parS.find(atts.find("name")->second);
      
    // if found, get std::vector, then add this value to it.
    // if not found, add this var, then add a value to it.

    if (existingName != parS.end())
      existingName->second.push_back(atts.find("value")->second);
    else
    {
      std::vector<std::string> tvect;
      tvect.push_back(atts.find("value")->second);
      parS[atts.find("name")->second] = tvect;
    }
  }
  
  algo.setParameters(st,ed,ic,parS,parE);
  // for efficiency, I do not want to make a DDAlgoPositioner every time this is called.
  // so DDCompactView must have a way to position inside itself as well.
  cpv.algoPosPart(self, parent, algo);
  //  ap_(self, parent, algo);
  // clear all "children" and attributes
  myChild->clear();
  myParent->clear();
  myParS->clear();
  myParE->clear();
  // after an AlgoPosPart, we are sure it can be cleared.
  clear();
  
  DCOUT_V('P', "DDLAlgoPosPart::processElement completed");
}

