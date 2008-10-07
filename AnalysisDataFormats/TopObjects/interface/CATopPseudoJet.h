#ifndef AnalysisDataFormats_TopObjects_CATopPseudoJet_h
#define AnalysisDataFormats_TopObjects_CATopPseudoJet_h



// -*- C++ -*-
//// -*- C++ -*-
//
// Package:    CATopPseudoJet
// Class:      CATopPseudoJet
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class CATopPseudoJet CATopPseudoJet.cc AnalysisDataFormats/interface/CATopPseudoJet.h
//!\brief CATopPseudoJet holds an association of fastjet::PseudoJets that represent
//!       a "hard" top jet with subjets.
//!
//-------------------------------------------------------------------------------------
//
// Original Author:  Salvatore Rappoccio
//         Created:  Wed Nov 28 15:31:57 CST 2007
// $Id: CATopPseudoJet.h,v 1.1 2008/09/05 15:03:30 srappocc Exp $
//
//-------------------------------------------------------------------------------------




#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>

#include <vector>
#include <algorithm>

class CATopPseudoSubJet {
 public:
  CATopPseudoSubJet() {}
  CATopPseudoSubJet( fastjet::PseudoJet const & subjet,
		     std::vector<int> const & constituents ) :
    subjet_(subjet),
    constituents_(constituents.size() ) 
      {
	copy( constituents.begin(), constituents.end(), constituents_.begin() );
      }

  ~CATopPseudoSubJet() {} 

  fastjet::PseudoJet const & subjet()       const { return subjet_; }
  std::vector<int> const &   constituents() const { return constituents_; }

 protected:
  fastjet::PseudoJet         subjet_;
  std::vector<int>           constituents_;
};

class CATopPseudoJet {

 public:
  CATopPseudoJet() {}
  CATopPseudoJet(fastjet::PseudoJet const & hardJet,
		 std::vector<CATopPseudoSubJet> const & subjets )  :
    hardJet_(hardJet),
    subjets_(subjets.size())
    {
      copy( subjets.begin(), subjets.end(),  subjets_.begin() );
    }

  ~CATopPseudoJet() {}

  fastjet::PseudoJet const &              hardJet() const { return hardJet_; }
  std::vector<CATopPseudoSubJet> const &  subjets() const { return subjets_; }


 protected:
  fastjet::PseudoJet                hardJet_;
  std::vector<CATopPseudoSubJet>    subjets_;
};


#endif
