// -*- C++ -*-
//
// Package:     CandCombiner
// Class  :     decayParser
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Aug  7 20:26:31 EDT 2005
// $Id: decayParser.cc,v 1.2 2007/10/15 17:59:01 llista Exp $
//
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <vector>
#include "PhysicsTools/CandAlgos/interface/decayParser.h"

using namespace boost::spirit;
using namespace std;

namespace cand {
  namespace parser {
    
    class ModeSetter {
    public:
      ModeSetter( vector<ConjInfo>& iVect, ConjInfo::Mode iMode) : 
	conjInfo_(iVect), mode_(iMode) {}
      void operator()( const char ) const {
	conjInfo_.back().mode_ = mode_;
      }
    private:
      vector<ConjInfo>& conjInfo_;
      ConjInfo::Mode mode_;
    };
    
    typedef scanner_policies<skip_parser_iteration_policy<nothing_parser, 
							  iteration_policy>, 
			     match_policy,action_policy> ScannerPolicy;
    typedef scanner<const char*, ScannerPolicy > ScannerUsed_1;
    typedef rule<ScannerUsed_1> Rule_1;
    
    bool decayParser( const string& iValue, vector<ConjInfo>& oStrings ) {
      using namespace boost::spirit;
      
      Rule_1 label = ((+alnum_p) >> *ch_p('_') >>*alnum_p)[push_back_a(oStrings)];
      Rule_1 conj =  (ch_p('@') >> !((ch_p('b')>>ch_p('a')>>ch_p('r')[ModeSetter(oStrings,ConjInfo::kBar)] )| 
				     ch_p('+')[ModeSetter(oStrings,ConjInfo::kPlus)] | 
				     ch_p('-')[ModeSetter(oStrings,ConjInfo::kMinus)]) );
      
      return parse( iValue.c_str(),
		    (
		     (label >> ! conj)
		     % blank_p
		     )
		    ,
		    nothing_p).full;
    }
  }
}
