/***************************************************************************
                          DDLSpecPar.cc  -  description
                             -------------------
    begin                : Tue Nov 21 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/src/DDLSpecPar.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDValuePair.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <sstream>

DDLSpecPar::DDLSpecPar( DDLElementRegistry* myreg )
  : DDXMLElement( myreg )
{}

DDLSpecPar::~DDLSpecPar( void )
{}

// Process a SpecPar element.  We have to assume that 
// certain things have happened.
void
DDLSpecPar::processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv )
{
  DCOUT_V('P',"DDLSpecPar::processElement started");

  // sends the call to the DDD Core OR does nothing if it is a sub-element

  // What I want to do here is the following:
  // 1.  output PartSelector information.
  // 2.  pass the Path and parameters to DDSpecifics
  // for each of the above, use the name of the SpecPar, since DDL does not
  // provide a name for a PartSelector.

  DDXMLElement* myParameter      = myRegistry_->getElement("Parameter");
  DDXMLElement* myNumeric        = myRegistry_->getElement("Numeric");
  DDXMLElement* myString         = myRegistry_->getElement("String");
  DDXMLElement* myPartSelector   = myRegistry_->getElement("PartSelector");
  DDXMLElement* mySpecParSection = myRegistry_->getElement("SpecParSection");

  // Because of namespace magic "!" means namespaces should be provided
  // in the names of the XML elements for the DDD.  So if this is
  // the state/case then we need to force the expression evaluator to 
  // use the namespace of the SpecPar element being processed.
  // --  Michael Case 2008-11-06
  std::string ns(nmspace);
  DDXMLAttribute spatts = getAttributeSet();
  std::string rn = spatts.find("name")->second;
  if ( ns == "!" ) {
    size_t foundColon= rn.find(':');
    if (foundColon != std::string::npos) {
      ns = rn.substr(0,foundColon);
      //       rn = rn.substr(foundColon+1);
    }
  }

  // DDPartSelector name comes from DDLSpecPar (this class, there is no analogue to 
  // DDLSpecPar in DDCore)
  std::vector <std::string> partsels;
  size_t i;

  //    if (getName("name") == "")
  //      {
  //        std::cout << "ERROR: no name for SpecPar" << std::endl;
  //        partsels = myPartSelector->getVectorAttribute("path");
  //        snames = myParameter->getVectorAttribute("name");
  //        std::cout << "\tParameter Names" << std::endl;
  //        size_t i;
  //        for (i = 0; i < snames.size(); ++i)
  //  	{
  //  	  std::cout << "\t\t" << snames[i] << std::endl;
  //  	}
  //        std::cout << "\tPart Selectors:" << std::endl;
  //        for (i = 0; i < partsels.size(); ++i)
  //  	{
  //  	  std::cout << "\t\t" << partsels[i] << std::endl;
  //  	}
  //      }
  //    else 
  //      {

  //should i keep this? partsels = myPartSelector->getVectorAttribute("path");
  //otherise I have to do this block...
  for (i = 0; i < myPartSelector->size(); ++i)
    partsels.push_back((myPartSelector->getAttributeSet(i).find("path"))->second);
  DDsvalues_type svt;

  // boolean flag to indicate whether the std::vector<DDValuePair> has been evaluated 
  // using the Evaluator
  typedef std::map<std::string, std::pair<bool,std::vector<DDValuePair> > > vvvpType;

  vvvpType vvvp;

  /** 08/13/03 doNotEval for Parameter is based on the value of the eval flag.
      For String it is always false and for Numeric it is always true.
      But for "legacy" Parameter, remember, we need to check eval.
      Default is NOT to evaluate.
  **/
  bool doNotEval = true;
  bool doRegex = true;
  {
    // check parent level  
    const DDXMLAttribute & atts = mySpecParSection->getAttributeSet();
    
    if (atts.find("eval") != atts.end() && atts.find("eval")->second == "true")
      doNotEval = false;
    
    if (atts.find("regex") != atts.end() && atts.find("regex")->second == "false")
      doRegex = false;
  }
  {
    // check this level
    const DDXMLAttribute & atts = getAttributeSet();
    
    if (atts.find("eval") != atts.end() && atts.find("eval")->second == "true")
      doNotEval = false;
    else if (atts.find("eval") != atts.end())
      doNotEval = true;
    
    if (atts.find("regex") != atts.end() && atts.find("regex")->second == "false")
      doRegex = false;
    else if (atts.find("regex") != atts.end())
      doRegex = true;
  }
  for (i = 0; i < myParameter->size(); ++i)
  {
    const DDXMLAttribute & atts = myParameter->getAttributeSet(i);
    std::vector <DDValuePair> vvp;
    vvvpType::iterator itv = vvvp.find((atts.find("name")->second));
    if (itv != vvvp.end())
      vvp = itv->second.second;
    double tval = 0.0;
    bool isEvaluated = false;

    /** 
	1.  Check eval flag of each level (SpecParSection, SpecPar
	and Parameter).
	2.  Default is the closest specified eval attribute
	with any value other than "false".
    */
      
    // bool notThis =  doNotEval  myParameter->get(std::string("eval"), i) != "true";

    // since eval is an optional attribute, we need to check if it exists for
    // the debug statement.  Unfortunately, I see no way to do this internal
    // to the DCOUT_V statement... ts is a "wasted" variable.
    std::string ts = "** no eval attribute **";
    if (atts.find("eval") != atts.end()) 
      ts = atts.find("eval")->second;
    DCOUT_V('P', std::string("about to process ") << atts.find("value")->second << std::string(" eval = ") << ts);
    if ((atts.find("eval") != atts.end() && atts.find("eval")->second !="false")
	|| (atts.find("eval") == atts.end() && !doNotEval))
    { 
      tval = ExprEvalSingleton::instance().eval(ns, atts.find("value")->second);
      isEvaluated=true;
      DCOUT_V('P', std::string("EVALUATED"));
    }
    else
    {
      DCOUT_V('P', std::string("NOT Evaluated"));
    }
      
    DDValuePair vp(atts.find("value")->second, tval);
    vvp.push_back(vp);
    vvvp[atts.find("name")->second] = make_pair(isEvaluated,vvp);
  }

  // Process the String names and values.
  for (i = 0; i < myString->size(); ++i)
  {
    const DDXMLAttribute & atts = myString->getAttributeSet(i);
    std::vector <DDValuePair> vvp;
    vvvpType::iterator itv = vvvp.find(atts.find("name")->second);
    if (itv != vvvp.end())
      vvp = itv->second.second;
    DCOUT_V('P', std::string("about to process String ") << (atts.find("name")->second) << " = " << (atts.find("value")->second));
    DDValuePair vp(atts.find("value")->second, 0.0);
    vvp.push_back(vp);
    vvvp[atts.find("name")->second] = make_pair(false,vvp);
  }
  
  // Process the Numeric names and values.
  for (i = 0; i < myNumeric->size(); ++i)
  {
    const DDXMLAttribute & atts = myNumeric->getAttributeSet(i);
    std::vector <DDValuePair> vvp;
    vvvpType::iterator itv = vvvp.find(atts.find("name")->second);
    if (itv != vvvp.end())
      vvp = itv->second.second;
    DCOUT_V('P', std::string("about to process String ") << (atts.find("name")->second) << " = " << (atts.find("value")->second));
    double tval = ExprEvalSingleton::instance().eval(ns, atts.find("value")->second);
    DCOUT_V('P', std::string("EVALUATED"));
    DDValuePair vp(atts.find("value")->second, tval);
    vvp.push_back(vp);
    vvvp[atts.find("name")->second] = make_pair(true,vvp);
  }
  
  svt.reserve(vvvp.size());
  for (vvvpType::const_iterator it = vvvp.begin(); it != vvvp.end(); ++it)
  {
    DDValue val(it->first, it->second.second);
    bool isEvaluated = it->second.first;
    val.setEvalState(isEvaluated);
    svt.push_back(DDsvalues_Content_type(val,val));      
  }
  std::sort(svt.begin(),svt.end());


  DCOUT_V('p', "DDLSpecPar::processElement\n\tname " << getDDName(nmspace) << "\n\tpartsels.size() = " << myPartSelector->size() << "\n\tsvt " << svt);

  DDSpecifics ds(getDDName(nmspace), 
		 partsels,
		 svt,
		 doRegex);

  myParameter->clear();
  myPartSelector->clear();
  
  // after a SpecPar is done, we can clear
  clear();
  
  DCOUT_V('P',"DDLSpecPar::processElement(...)");
}

