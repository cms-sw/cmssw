#ifndef PhysicsTools_CandUtils_CandCombinerBase_h
#define PhysicsTools_CandUtils_CandCombinerBase_h
/** \class CandCombinerBase
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include <vector>

class CandCombinerBase {
public:
  /// default construct
  CandCombinerBase();
  /// construct from two charge values
  CandCombinerBase(int, int);
  /// construct from three charge values
  CandCombinerBase(int, int, int);
  /// construct from four charge values
  CandCombinerBase(int, int, int, int);
  /// constructor from a selector, specifying optionally to check for charge
  CandCombinerBase(bool checkCharge, const std::vector <int> &);
  /// destructor
  virtual ~CandCombinerBase();
  /// return all selected candidate pairs
  std::auto_ptr<reco::CompositeCandidateCollection> 
  combine(const std::vector<edm::Handle<reco::CandidateView> > &) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::CompositeCandidateCollection> 
  combine(const edm::Handle<reco::CandidateView> &) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::CompositeCandidateCollection> 
  combine(const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::CompositeCandidateCollection> 
  combine(const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::CompositeCandidateCollection> 
  combine(const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &) const;

private:
  /// verify that the two candidate don't overlap and check charge
  bool preselect(const reco::Candidate &, const reco::Candidate &) const;
  /// returns a composite candidate combined from two daughters
  void combine(reco::CompositeCandidate &, 
	       const reco::CandidateBaseRef &, 
	       const reco::CandidateBaseRef &) const;
  /// temporary candidate stack
  typedef std::vector<std::pair<std::pair<reco::CandidateBaseRef, size_t>, 
				std::vector<edm::Handle<reco::CandidateView> >::const_iterator> > CandStack;
  typedef std::vector<int> ChargeStack;
  /// returns a composite candidate combined from two daughters
  void combine(size_t collectionIndex, CandStack &, ChargeStack &,
	       std::vector<edm::Handle<reco::CandidateView> >::const_iterator begin,
	       std::vector<edm::Handle<reco::CandidateView> >::const_iterator end,
	       std::auto_ptr<reco::CompositeCandidateCollection> & comps
	       ) const;
  /// select a candidate
  virtual bool select(const reco::Candidate &) const = 0;
  /// select a candidate pair
  virtual bool selectPair(const reco::Candidate & c1, const reco::Candidate & c2) const = 0;
  /// set kinematics to reconstructed composite
  virtual void setup(reco::CompositeCandidate &) const = 0;
  /// add candidate daughter
  virtual void addDaughter(reco::CompositeCandidate & cmp, const reco::CandidateBaseRef & c) const = 0;
  /// flag to specify the checking of electric charge
  bool checkCharge_;
  /// electric charges of the daughters
  std::vector<int> dauCharge_;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap_;
};

#endif
