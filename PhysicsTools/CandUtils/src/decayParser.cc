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
// $Id: decayParser.cc,v 1.1 2005/10/24 10:16:57 llista Exp $
//
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <vector>
#include "PhysicsTools/CandUtils/interface/decayParser.h"

namespace candcombiner {

  class ModeSetter {
  public:
    ModeSetter( std::vector<ConjInfo>& iVect, ConjInfo::Mode iMode) : conjInfo_(iVect), mode_(iMode) {}
    
    void operator()( const char ) const {
      conjInfo_.back().mode_ = mode_;
    }
  private:
    std::vector<ConjInfo>& conjInfo_;
    ConjInfo::Mode mode_;
  };
  
  
  typedef boost::spirit::scanner_policies<boost::spirit::skip_parser_iteration_policy<boost::spirit::nothing_parser, boost::spirit::iteration_policy>, boost::spirit::match_policy, boost::spirit::action_policy> ScannerPolicy;
  typedef boost::spirit::scanner<const char*, ScannerPolicy > ScannerUsed_1;
  typedef boost::spirit::rule<ScannerUsed_1> Rule_1;
  
  bool decayParser( const std::string& iValue, std::vector<ConjInfo>& oStrings ) {
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
