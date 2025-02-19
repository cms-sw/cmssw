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
// $Id: CompoundPseudoJet.h,v 1.2 2011/03/24 17:02:39 schiefer Exp $
//
//-------------------------------------------------------------------------------------




#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>

#include <vector>
#include <algorithm>

class CompoundPseudoSubJet {
 public:
  CompoundPseudoSubJet() {}
  CompoundPseudoSubJet(fastjet::PseudoJet const & subjet,
		       std::vector<int> const & constituents ) :
    subjet_(subjet),
    subjetArea_(0.0),
    constituents_(constituents.size() ) 
  {
    copy( constituents.begin(), constituents.end(), constituents_.begin() );
  }
  CompoundPseudoSubJet(fastjet::PseudoJet const & subjet,
		       double subjetArea,
		       std::vector<int> const & constituents ) :
    subjet_(subjet),
    subjetArea_(subjetArea),
    constituents_(constituents.size() ) 
  {
    copy( constituents.begin(), constituents.end(), constituents_.begin() );
  }
  
  ~CompoundPseudoSubJet() {} 
  
  fastjet::PseudoJet const & subjet()       const { return subjet_; }
  double                     subjetArea()   const { return subjetArea_; }
  std::vector<int> const &   constituents() const { return constituents_; }
  
protected:
  fastjet::PseudoJet         subjet_;
  double                     subjetArea_;
  std::vector<int>           constituents_;
};

class CompoundPseudoJet {

public:
  CompoundPseudoJet() {}
  CompoundPseudoJet(fastjet::PseudoJet const & hardJet,
		    std::vector<CompoundPseudoSubJet> const & subjets )  :
    hardJet_(hardJet),
    hardJetArea_(0.0),
    subjets_(subjets.size())
  {
    copy( subjets.begin(), subjets.end(),  subjets_.begin() );
  }
  CompoundPseudoJet(fastjet::PseudoJet const & hardJet,
		    double hardJetArea,
		    std::vector<CompoundPseudoSubJet> const & subjets )  :
    hardJet_(hardJet),
    hardJetArea_(hardJetArea),
    subjets_(subjets.size())
  {
    copy( subjets.begin(), subjets.end(),  subjets_.begin() );
  }
  
  ~CompoundPseudoJet() {}
  
  fastjet::PseudoJet const &              hardJet()     const {return hardJet_;}
  double                                  hardJetArea() const {return hardJetArea_;}
  std::vector<CompoundPseudoSubJet>const& subjets()     const {return subjets_; }
  
  
protected:
  fastjet::PseudoJet                hardJet_;
  double                            hardJetArea_;
  std::vector<CompoundPseudoSubJet> subjets_;
};



class GreaterByEtPseudoJet : 
  public std::binary_function<fastjet::PseudoJet const &, fastjet::PseudoJet const &, bool> {
  
public:
  bool operator()( fastjet::PseudoJet const & j1, fastjet::PseudoJet const & j2 ) {
    return j1.perp() > j2.perp();
  }
};

#endif
