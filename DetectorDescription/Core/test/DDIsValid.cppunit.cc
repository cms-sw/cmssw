#include <cppunit/extensions/HelperMacros.h>
#include <ext/alloc_traits.h>
#include <regex.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/Singleton.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

namespace {

  std::pair<bool,std::string> oldDDIsValid(const std::string & ns, const std::string & nm, std::vector<DDLogicalPart> & result, bool doRegex)
  {
    //return std::make_pair(true,"");      
    std::string status, aName, aNs;
    bool emptyNs = false;
    if (ns=="") emptyNs=true;
    
    aName = "^" + nm + "$";
    aNs  =  "^" + ns + "$";
    //bool flag(true);
    regex_t aRegex, aNsRegex;
    const char * aRegexCStr = aName.c_str();
    const char * aNsRegexCStr = aNs.c_str();
    if (doRegex) {
      //    std:: cout << "Regex " << nm << " " << ns << std::endl; 
      if (regcomp(&aRegex,aRegexCStr,0)) {
	regfree(&aRegex);
	return std::make_pair(false,"Error in the regexp for the name: " + nm);
      }
      
      if (regcomp(&aNsRegex,aNsRegexCStr,0)) {
	regfree(&aRegex);
	regfree(&aNsRegex);
	return std::make_pair(false,"Error in the regexp for the namespace: " + ns);
      }
    }  
    else {
      DDName ddnm(nm,ns);
      result.emplace_back(DDLogicalPart(ddnm));
      return std::make_pair(true,"");
    }
    //edm::LogInfo("DDLogicalPart") << " . emptyNs=" << emptyNs << std::endl;
    //edm::LogInfo("DDLogicalPart") << " . qname=[" << ns << ":" << nm << "]" << std::endl;
    
    // THIS IS THE SLOW PART: I have to compare every namespace & name of every
    // logical part with a regex-comparison .... a linear search always through the
    // full range of logical parts!!!!
    /*
      Algorithm description:
      x. empty nm and ns argument of method means: use all matching regex ^.*$
      a. iterate over all logical part names, match against regex for names
      b. iterate over all namespaces of names found in a & match against regex for namespaces   
    */
    LPNAMES::value_type::const_iterator it(LPNAMES::instance().begin()),
      ed(LPNAMES::instance().end());
    for (; it != ed; ++it) {
      bool doit = false;
      doit = !regexec(&aRegex, it->first.c_str(), 0,nullptr,0);
      //if (doit)  edm::LogInfo("DDLogicalPart") << "rgx: " << aName << ' ' << it->first << ' ' << doit << std::endl;
      if ( doit  ) {
	std::vector<DDName>::size_type sz = it->second.size(); // no of 'compatible' namespaces
	if ( emptyNs && (sz==1) ) { // accept all logical parts in all the namespaces
	  result.emplace_back(it->second[0]);
	  //std::vector<DDName>::const_iterator nsIt(it->second.begin()), nsEd(it->second.end());
	  //for(; nsIt != nsEd; ++nsIt) {
	  //   result.emplace_back(DDLogicalPart(*nsIt));
	  //   edm::LogInfo("DDLogicalPart") << "DDD-WARNING: multiple namespaces match (in SpecPars PartSelector): " << *nsIt << std::endl;
	  //}
	}
	else if ( !emptyNs ) { // only accept matching namespaces
	  std::vector<DDName>::const_iterator nsit(it->second.begin()), nsed(it->second.end());
	  for (; nsit !=nsed; ++nsit) {
	    //edm::LogInfo("DDLogicalPart") << "comparing " << aNs << " with " << *nsit << std::endl;
	    bool another_doit = !regexec(&aNsRegex, nsit->ns().c_str(), 0,nullptr,0);
	    if ( another_doit ) {
	      //temp.emplace_back(std::make_pair(it->first,*nsit));
	      result.emplace_back(DDLogicalPart(*nsit));
	    }
	  }
	}
	else { // emtpyNs and sz>1 -> error, too ambigous
	  std::string message = "DDLogicalPart-name \"" + it->first +"\" matching regex \""
	    + nm + "\" has been found at least in following namespaces:\n";
	  std::vector<DDName>::const_iterator vit = it->second.begin();
	  for(; vit != it->second.end(); ++vit) {
	    message += vit->ns();
	    message += " "; 
	  } 
	  message += "\nQualify the name with a regexp for the namespace, i.e \".*:name-regexp\" !";
	  
	  regfree(&aRegex);
	  regfree(&aNsRegex);	
	  
	  return std::make_pair(false,message);        
	}
      }
    }
    bool flag=true;    
    std::string message;
    
    // check whether the found logical-parts are also defined (i.e. have material, solid ...)
    // std::cout << "IsValid-Result " << nm << " :";
    if (!result.empty()) {
      std::vector<DDLogicalPart>::const_iterator lpit(result.begin()), lped(result.end());
      for (; lpit != lped; ++lpit) {
	// std::cout << " " << std::string(lpit->name());
	if (!lpit->isDefined().second) {
	  message = message + "LogicalPart " + lpit->name().fullname() + " not (yet) defined!\n";
	  flag = false;
	}
      }
    }
    else {
      flag = false;
      message = "No regex-match for namespace=" + ns + "  name=" + nm + "\n";
    }
    regfree(&aRegex);
    regfree(&aNsRegex);
    // std::cout << std::endl;
    
    return std::make_pair(flag,message);
  }
}

namespace {
  // the static dumper 
  // to be used in the real application
  struct BHA {
    ~BHA() {
      LPNAMES::value_type::const_iterator it(LPNAMES::instance().begin()),
	ed(LPNAMES::instance().end());
      std::cout <<  "LPNAMES begin" << std::endl;
      for (; it != ed; ++it) {
	std::cout << it->first;
	std::vector<DDName>::const_iterator nsit(it->second.begin()), nsed(it->second.end());
	for (; nsit !=nsed; ++nsit) std::cout << " " << nsit->ns();
	std::cout << std::endl;
      }
      std::cout <<  "LPNAMES end" << std::endl;
    }
  };
  //  BHA bah;
}

class testDDIsValid : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDIsValid);
  CPPUNIT_TEST(checkAgaistOld);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() override{}
  void tearDown() override {}
  void buildIt();
  void testloading();
  void checkAgaistOld();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDIsValid);

void testDDIsValid::testloading() {
  std::cerr << "test Loading" << std::endl;
  // get name using FileInPath.
  edm::FileInPath fp("DetectorDescription/Core/test/lpnames.out");
  // test that we load LPNAMES correclty
  //  std::ifstream in("lpnames.out");
  std::ifstream in(fp.fullPath().c_str());
  //  std::cout << "FILE : " << fp << " absolute path " << fp.fullPath().c_str() << std::endl;
  in.unsetf( std::ios::skipws );
  std::istream_iterator<char> sbegin(in),send;
  std::string str;
  std::copy(sbegin,send,std::inserter(str,str.begin()));
  
  std::ostringstream  os;
  
  LPNAMES::value_type::const_iterator it(LPNAMES::instance().begin()),
    ed(LPNAMES::instance().end());
  for (; it != ed; ++it) {
    os << it->first;
    std::vector<DDName>::const_iterator nsit(it->second.begin()), nsed(it->second.end());
    for (; nsit !=nsed; ++nsit) os << " " << nsit->ns();
    os << std::endl;
  }

  if (os.str()!=str) std::cerr << "not the same!" << std::endl;
  CPPUNIT_ASSERT (os.str()==str);

}


void testDDIsValid::buildIt() {
  // get name using FileInPath.
  edm::FileInPath fp("DetectorDescription/Core/test/lpnames.out");
  // fill LPNAMES
  //  std::ifstream in("lpnames.out");
  std::ifstream in(fp.fullPath().c_str());
  std::string line;
  while (std::getline(in,line) ) {
    std::string::size_type p;
    p = line.find(" ");
    std::string nm(line,0,p);
    std::vector<DDName> & v = LPNAMES::instance()[nm];
    while (p!=std::string::npos) {
      ++p;
      std::string::size_type e=line.find(" ",p);
      std::string::size_type s = e-p;
      if (e==std::string::npos) s=e;
      v.emplace_back(DDName(nm,line.substr(p,s))); 
      p=e;
    }
  }
  std::cerr << "read " << LPNAMES::instance().size() << std::endl;
  CPPUNIT_ASSERT (LPNAMES::instance().size()==3819);
}


void testDDIsValid::checkAgaistOld() {

  buildIt();
  testloading();
  // get name using FileInPath.
  edm::FileInPath fp("DetectorDescription/Core/test/regex.queries");

  //  std::ifstream in("regex.queries");
  std::ifstream in(fp.fullPath().c_str());

  std::string line;
  const std::string ns;
  int bad=0;
  while (std::getline(in,line) ) {
    std::string::size_type p = line.find(" ");
    ++p;
    std::string::size_type e = line.find(" ",p);
    // ns, we know, is always ""
    std::vector<DDLogicalPart>  oldResult;
    std::vector<DDLogicalPart>  result;
    std::pair<bool,std::string> oldRes = 
      oldDDIsValid(ns, line.substr(p,e-p), oldResult, true);
    std::pair<bool,std::string> res = 
      DDIsValid(ns, line.substr(p,e-p), result, true);
    if (oldRes.first!=res.first || 
	oldRes.second!=res.second ||
	oldResult.size()!=result.size() ) {
      ++bad;
      continue;
    }
    for (int i=0; i!=int(result.size()); ++i) {
      if (result[i].name()==oldResult[i].name()) continue;
      ++bad; 
      break;
    }
  }

  if (bad) std::cerr << bad << " queries not the same!" << std::endl;
  CPPUNIT_ASSERT (bad==0);

}

