#ifndef RecoJets_JetAlgorithms_CompoundPseudoJet_h
#define RecoJets_JetAlgorithms_CompoundPseudoJet_h



// -*- C++ -*-
//// -*- C++ -*-
//
// Package:    CompoundPseudoJet
// Class:      CompoundPseudoJet
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class CompoundPseudoJet CompoundPseudoJet.cc RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h
//!\brief CompoundPseudoJet holds an association of fastjet::PseudoJets that represent
//!       a "hard" top jet with subjets.
//!
//-------------------------------------------------------------------------------------
//
// Original Author:  Salvatore Rappoccio
//         Created:  Wed Nov 28 15:31:57 CST 2007
// $Id: CompoundPseudoJet.h,v 1.1 2008/10/07 20:44:27 srappocc Exp $
//
//-------------------------------------------------------------------------------------




#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>

#include <vector>
#include <algorithm>

class CompoundPseudoSubJet {
 public:
  CompoundPseudoSubJet() {}
  CompoundPseudoSubJet( fastjet::PseudoJet const & subjet,
		     std::vector<int> const & constituents ) :
    subjet_(subjet),
    constituents_(constituents.size() ) 
      {
	copy( constituents.begin(), constituents.end(), constituents_.begin() );
      }

  ~CompoundPseudoSubJet() {} 

  fastjet::PseudoJet const & subjet()       const { return subjet_; }
  std::vector<int> const &   constituents() const { return constituents_; }

 protected:
  fastjet::PseudoJet         subjet_;
  std::vector<int>           constituents_;
};

class CompoundPseudoJet {

 public:
  CompoundPseudoJet() {}
  CompoundPseudoJet(fastjet::PseudoJet const & hardJet,
		 std::vector<CompoundPseudoSubJet> const & subjets )  :
    hardJet_(hardJet),
    subjets_(subjets.size())
    {
      copy( subjets.begin(), subjets.end(),  subjets_.begin() );
    }

  ~CompoundPseudoJet() {}

  fastjet::PseudoJet const &              hardJet() const { return hardJet_; }
  std::vector<CompoundPseudoSubJet> const &  subjets() const { return subjets_; }


 protected:
  fastjet::PseudoJet                    hardJet_;
  std::vector<CompoundPseudoSubJet>    subjets_;
};



class GreaterByEtPseudoJet : 
  public std::binary_function<fastjet::PseudoJet const &, fastjet::PseudoJet const &, bool> {

public:
  bool operator()( fastjet::PseudoJet const & j1, fastjet::PseudoJet const & j2 ) {
    return j1.perp() > j2.perp();
  }
};

#endif
