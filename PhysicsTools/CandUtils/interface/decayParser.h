#ifndef CANDCOMBINER_DECAYPARSER_H
#define CANDCOMBINER_DECAYPARSER_H
// -*- C++ -*-
//
// Package:     CandCombiner
// Class  :     decayParser
// 
/**\class decayParser decayParser.h PhysicsTools/CandCombiner/interface/decayParser.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Aug  7 20:26:36 EDT 2005
// $Id: decayParser.h,v 1.1 2005/10/24 10:16:57 llista Exp $
//

// system include files

// user include files
#include <vector>
#include <iostream>
#include <string>

#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>

// forward declarations

namespace candcombiner {
  struct ConjInfo {
    enum Mode { kPrimary, kBar, kPlus, kMinus } mode_;
    std::string label_;
    
    ConjInfo( const std::string& iLabel ) : mode_(kPrimary), label_(iLabel) {}
    ConjInfo(const char* iStart, const char* iEnd ) : mode_(kPrimary), label_(iStart,iEnd) {}
  };
  
  inline
  std::ostream& operator<<(std::ostream& oOstream, const ConjInfo& info ) {
    return oOstream << info.label_ << " " << ( 0 == info.mode_ ? "p" : ( info.mode_ == ConjInfo::kBar ? "b" : (info.mode_ == ConjInfo::kPlus ? "+" : "-" ) ) );
  }
  
  bool decayParser( const std::string& iValue, std::vector<ConjInfo>& oStrings );
}

#endif /* CANDCOMBINER_DECAYPARSER_H */
