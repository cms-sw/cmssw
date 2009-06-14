#ifndef FWCore_MessageService_MessageServicePSetValidation_h
#define FWCore_MessageService_MessageServicePSetValidation_h

// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageServicePSetValidation_h
//
/**\class MessageLogger MessageServicePSetValidation.h FWCore/MessageService/interface/MessageServicePSetValidation.h

 Description: Checking MessageService PSet in _cfg.py file for subtle problems

 Usage:
   MessageServicePSetValidation v;
   std::string valresults = v(iPS);
     where iPS is (passed by const reference) the MessageService PSet
   If valresults is not empty, then problems were found.
     (This is later checked in preSourceConstructor or preModuleConstructor)

*/
//
// Original Author:  M. Fischler
//         Created:  Tue May 19  2009
// $Id:  $
//

// system include files

#include <string>
#include <sstream>
#include <vector>

// user include files

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations

namespace edm  {
namespace service  {


class MessageServicePSetValidation
{
public:
  std::string operator() ( ParameterSet const & pset ); 
private:
  typedef std::string          String;
  typedef std::vector<String>  vString;

  void messageLoggerPSet ( ParameterSet const & pset ); 
  void psetLists ( ParameterSet const & pset ); 
  void suppressionLists ( ParameterSet const & pset ); 
  bool validateThreshold (std::string const & thresh, 
  			  std::string const & psetName);
  bool checkThreshold (std::string const & thresh);
  void noDuplicates(vString const & v, std::string const & psetName,
                    std::string  const & parameterLabel );
  void noDuplicates(vString const & v1, vString const & v2, 
             	    std::string const & psetName,
      		    std::string  const & p1, std::string  const & p2 );
  void noCoutCerrClash(vString const & v, std::string const & psetName,
                       std::string  const & parameterLabel );
  void noKeywords(vString const & v, std::string const & psetName,
                  std::string  const & parameterLabel );
  bool keywordCheck(std::string const & word);
  void noNonPSetUsage( ParameterSet const & pset, 
  		      vString const & v, std::string const & psetName,
                      std::string const & parameterLabel );
  void noBadParams(vString const & v, vString const & params, 
  		   std::string const & psetName, 
		   std::string const & parameterLabel, 
		   std::string const & type);
  void vStringsCheck(ParameterSet const & pset,std::string const & psetName); 
  bool wildcard(vString const & v);
  bool allowedVstring (std::string const & s);
  void noOtherPsets(ParameterSet const & pset); 
  void noNoncategoryPsets(ParameterSet const & pset,
  			  std::string const & psetName);  
  bool lookForMatch(vString const & v, std::string const & s);
  void destinationPSets(ParameterSet const & pset);
  void destinationPSet(ParameterSet const & pset, std::string const & psetName);
  void statisticsPSets(ParameterSet const & pset);
  void statisticsPSet(ParameterSet const & pset, std::string const & psetName);
  void fwkJobReportPSets(ParameterSet const & pset);
  void fwkJobReportPSet(ParameterSet const & pset,std::string const & psetName);
  void categoryPSets(ParameterSet const & pset, std::string const & psetName); 
  void categoryPSet (ParameterSet const & pset, 
  		     std::string const & OuterPsetName,
		     std::string const & categoryName); 
  void catInts (ParameterSet const & pset, 
  		std::string const & psetName,
		std::string const & categoryName); 
  void catNoPSets (ParameterSet const & pset, 
  		   std::string const & psetName,
		   std::string const & categoryName); 

  template <typename T>
  T check 
      ( ParameterSet const & pset, 
  	std::string  const & psetName,
	std::string  const & parameterLabel ) 
  {
    T val = T();
    try {
      if (!pset.exists(parameterLabel)) return val;
      if ( pset.existsAs<T> (parameterLabel, false) ) {
	val = pset.getUntrackedParameter<T>(parameterLabel, val);
	return val;
      }
      if ( pset.existsAs<T> (parameterLabel, true) )   {
	flaws << psetName << " PSet: \n"
	      << parameterLabel
	      << " is declared as tracked - needs to be untracked \n";
        val = pset.getParameter<T>(parameterLabel);
      } else {
        flaws << psetName << " PSet: \n"
	      << parameterLabel 
	      << " is declared with incorrect type \n";
      }
      return val;
    } catch (cms::Exception& e) {
      flaws << psetName << " PSet: \n"
	    << parameterLabel 
	    << " is declared but causes an exception when processed: \n"
	    << e.what() << "\n";
      return val;
    }
  }  // check()

  template <typename T>
  void disallowedParam 
      ( ParameterSet const & pset,
        vString      const & v,
  	std::string  const & psetName,
	std::string  const & parameterLabel, 	
	std::string  const & type ) 
  {
    vString params = pset.getParameterNamesForType <T> (true);
    noBadParams(v, params, psetName, parameterLabel, type);
    params = pset.getParameterNamesForType <T> (false);
    noBadParams(v, params, psetName, parameterLabel, type);
  }  // disallowedParam()

  template <typename T>
  void noneExcept (ParameterSet const & pset, 
  		std::string  const & psetName, std::string const & type)
  {
    vString x = pset.template getParameterNamesForType <T> (false);
    vString::const_iterator end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      flaws << psetName << " PSet: \n"
	    << (*i) << " is used as a " << type << "\n"
	    << "Usage of " << type << " is not recognized here\n"; 
    }
    x = pset.template getParameterNamesForType <T> (true);
    end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      if  ( (*i) == "@service_type" ) continue;    
      flaws << psetName << " PSet: \n"
	    << (*i) << " is used as a tracked " << type << "\n"
	    << "Tracked parameters not allowed here, "
	    <<" and even untracked it would not be recognized\n";
    }
  } // noneExcept()

  template <typename T>
  void noneExcept (ParameterSet const & pset, 
  		std::string  const & psetName, std::string const & type,
		std::string  const & ok)
  {
    vString x = pset.template getParameterNamesForType <T> (false);
    vString::const_iterator end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      std::string val = (*i);
      if ( val != ok ) {
	flaws << psetName << " PSet: \n"
	      << val << " is used as a " << type << "\n"
	      << "This usage is not recognized in this type of PSet\n"; 
      }
    }
    x = pset.template getParameterNamesForType <T> (true);
    end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      if  ( (*i) == "@service_type" ) continue;    
      flaws << psetName << " PSet: \n"
	    << (*i) << " is used as a tracked " << type << "\n"
	    << "Tracked parameters not allowed here\n";
    }
  } // noneExcept(okValue)

  template <typename T>
  void noneExcept (ParameterSet const & pset, 
  		std::string  const & psetName, std::string const & type,
		T const & ok1, T const & ok2 )
  {
    vString x = pset.template getParameterNamesForType <T> (false);
    vString::const_iterator end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      std::string val = (*i);
      if ( (val != ok1) && (val != ok2) ) {
	flaws << psetName << " PSet: \n"
	      << val << " is used as a " << type << "\n"
	      << "This usage is not recognized in this type of PSet\n"; 
      }
    }
    x = pset.template getParameterNamesForType <T> (true);
    end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      if  ( (*i) == "@service_type" ) continue;    
      flaws << psetName << " PSet: \n"
	    << (*i) << " is used as a tracked " << type << "\n"
	    << "Tracked parameters not allowed here\n";
    }
  } // noneExcept(okValue1, okValue2)

  template <typename T>
  void noneExcept (ParameterSet const & pset, 
  		std::string  const & psetName, std::string const & type,
		vString const & vok )
  {
    vString x = pset.template getParameterNamesForType <T> (false);
    vString::const_iterator end = x.end();
    vString::const_iterator vend = vok.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      bool found = false;    
      for (vString::const_iterator vit = vok.begin(); vit!=vend; ++vit) {
	if ( *i == *vit ) found = true;
      }
      if ( !found ) {
	  flaws << psetName << " PSet: \n"
		<< *i << " is used as a " << type << "\n"
		<< "This usage is not recognized in this type of PSet\n"; 
      }
    }
    x = pset.template getParameterNamesForType <T> (true);
    end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      if  ( (*i) == "@service_type" ) continue;    
      flaws << psetName << " PSet: \n"
	    << (*i) << " is used as a tracked " << type << "\n"
	    << "Tracked parameters not allowed here\n";
    }
  } // noneExcept(vok)

  template <typename T>
  void catNone (ParameterSet const & pset, 
  		std::string  const & psetName, 
  		std::string  const & categoryName, 
		std::string const & type)
  {
    vString x = pset.template getParameterNamesForType <T> (false);
    vString::const_iterator end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      flaws << categoryName << " category PSet nested in "
      	    << psetName << " PSet: \n"
	    << (*i) << " is used as a " << type << "\n"
	    << "Usage of " << type << " is not recognized here\n"; 
    }
    x = pset.template getParameterNamesForType <T> (true);
    end = x.end();
    for ( vString::const_iterator i = x.begin(); i != end; ++i ) {
      flaws << categoryName << " category PSet nested in "
      	    << psetName << " PSet: \n"
	    << (*i) << " is used as a tracked " << type << "\n"
	    << "Tracked parameters not allowed here, "
	    <<" and even untracked it would not be recognized\n";
    }
  } // catNone()



  // private data  
  std::ostringstream flaws; 
  std::vector<std::string> destinations;
  std::vector<std::string> statistics;
  std::vector<std::string> fwkJobReports;
  std::vector<std::string> categories;
  std::vector<std::string> messageIDs;
  std::vector<std::string> debugModules;
  std::vector<std::string> suppressInfo;
  std::vector<std::string> suppressDebug;
  std::vector<std::string> suppressWarning;

  
};  // MessageServicePSetValidation


}  // namespace service

}  // namespace edm



#endif  // FWCore_MessageService_MessageServicePSetValidation_h

