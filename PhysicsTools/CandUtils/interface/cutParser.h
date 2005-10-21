#if !defined(PACKAGE_CUTPARSER_H)
#define PACKAGE_CUTPARSER_H
// -*- C++ -*-
//
// Package:     <package>
// Module:      cutParser
// 
/**\class cutParser cutParser.h package/cutParser.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris D Jones
// Created:     Sun Aug  7 20:45:51 EDT 2005
// $Id$
//
// Revision history
//
// $Log$

// system include files
#include <string>
#include <boost/shared_ptr.hpp>
#include <map>

// user include files

// forward declarations
namespace aod {
  class Candidate;
  class Selector;
}
namespace aod {
  typedef double ( aod::Candidate::* PCandMethod )() const;
  typedef std::map<std::string, PCandMethod> CandidateMethods;
  
  bool cutParser( const std::string & iString,
		  const CandidateMethods & iMethods,
		  boost::shared_ptr<aod::Selector> & oCut );
}

#endif /* PACKAGE_CUTPARSER_H */
