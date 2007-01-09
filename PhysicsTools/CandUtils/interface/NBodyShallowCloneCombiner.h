#ifndef CandUtils_NBodyShallowCloneCombiner_h
#define CandUtils_NBodyShallowCloneCombiner_h
/** \class NBodyShallowCloneCombiner
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Framework/interface/Handle.h"
#include <boost/shared_ptr.hpp>
#include <vector>
#include <utility>

class NBodyShallowCloneCombinerBase {
public:
  /// constructor from a selector, specifying optionally to check for charge
  NBodyShallowCloneCombinerBase( bool checkCharge, const std::vector <int> & );
  /// destructor
  virtual ~NBodyShallowCloneCombinerBase();
  /// return all selected candidate pairs
  std::auto_ptr<reco::CandidateCollection> 
  combine( const std::vector<reco::CandidateRefProd> & ) const;

private:
  /// verify that the two candidate don't overlap and check charge
  bool preselect( const reco::Candidate &, const reco::Candidate & ) const;
  /// returns a composite candidate combined from two daughters
  reco::Candidate * combine( const reco::CandidateRef &, const reco::CandidateRef & ) const;
  /// charge information flag
  enum ChargeInfo { undetermined, same, opposite, invalid };
  /// return charge information
  static ChargeInfo chargeInfo( int q1, int q2 ); 
  /// temporary candidate stack
  typedef std::vector<std::pair<reco::CandidateCollection::const_iterator, 
                                std::vector<reco::CandidateRefProd>::const_iterator> 
                     > CandStack;
  /// returns a composite candidate combined from two daughters
  void combine( size_t collectionIndex, ChargeInfo ch, CandStack &, 
		std::vector<reco::CandidateRefProd>::const_iterator begin,
		std::vector<reco::CandidateRefProd>::const_iterator end,
		std::auto_ptr<reco::CandidateCollection> & comps
		) const;
  /// select a candidate
  virtual bool select( const reco::Candidate & ) const = 0;
  /// set kinematics to reconstructed composite
  virtual void setup( reco::Candidate * ) const = 0;
  /// flag to specify the checking of electric charge
  bool checkCharge_;
  /// electric charges of the daughters
  std::vector<int> dauCharge_;
  /// utility to setup composite candidate kinematics from daughters
  AddFourMomenta addp4_;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap_;
};

template<typename S, typename Setup = AddFourMomenta>
class NBodyShallowCloneCombiner : public NBodyShallowCloneCombinerBase {
public:
  /// constructor from a selector, specifying optionally to check for charge
  template<typename B>
  NBodyShallowCloneCombiner( const B & select,
		 bool checkCharge, const std::vector <int> & dauCharge ) : 
    NBodyShallowCloneCombinerBase( checkCharge, dauCharge ), 
    select_( select ) { }
  /// constructor from a selector, specifying optionally to check for charge
  NBodyShallowCloneCombiner( const edm::ParameterSet & cfg,
		 bool checkCharge, const std::vector <int> & dauCharge ) : 
    NBodyShallowCloneCombinerBase( checkCharge, dauCharge ), 
    select_( cfg ) { }

private:
  /// select a candidate
  virtual bool select( const reco::Candidate & c ) const {
    return select_( c );
  } 
  /// set kinematics to reconstructed composite
  virtual void setup( reco::Candidate * c ) const {
    setup_.set( * c );
  }
  /// candidate selector
  S select_; 
  /// utility to setup composite candidate kinematics from daughters
  Setup setup_;
};

#endif
