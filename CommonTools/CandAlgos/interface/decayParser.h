#ifndef CANDCOMBINER_DECAYPARSER_H
#define CANDCOMBINER_DECAYPARSER_H
// -*- C++ -*-
//
// Package:     CandCombiner
// Class  :     decayParser
// 
/**\class decayParser decayParser.h CommonTools/CandCombiner/interface/decayParser.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Aug  7 20:26:36 EDT 2005
// $Id: decayParser.h,v 1.2 2009/05/08 12:54:34 llista Exp $
//

// system include files

// user include files
#include <vector>
#include <iostream>
#include <string>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>
#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations

namespace cand {
  namespace parser {
    struct ConjInfo {
      enum Mode { kPrimary, kBar, kPlus, kMinus } mode_;
      edm::InputTag tag_;
      ConjInfo( const std::string& tag ) : mode_( kPrimary ), tag_( tag ) { }
      ConjInfo( const char* begin, const char* end ) : mode_( kPrimary ), tag_( std::string( begin, end ) ) { }
    };
    
    inline
    std::ostream& operator<<(std::ostream& out, const ConjInfo& info ) {
      return out << info.tag_ .encode() << " " 
		 << ( 0 == info.mode_ ? "p" : ( info.mode_ == ConjInfo::kBar ? "b" : (info.mode_ == ConjInfo::kPlus ? "+" : "-" ) ) );
    }
    
    bool decayParser( const std::string& iValue, std::vector<ConjInfo>& oStrings );
  }
}

#endif /* CANDCOMBINER_DECAYPARSER_H */
