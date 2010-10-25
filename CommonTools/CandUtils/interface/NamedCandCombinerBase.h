#ifndef CommonTools_CandUtils_CandCombinerBase_h
#define CommonTools_CandUtils_CandCombinerBase_h
/** \class CandCombinerBase
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include <vector>
#include <string>

class NamedCandCombinerBase {
public:
  typedef std::vector<std::string>  string_coll;
  /// default construct
  NamedCandCombinerBase(std::string name);
  /// construct from two charge values
  NamedCandCombinerBase(std::string name, int, int);
  /// construct from three charge values
  NamedCandCombinerBase(std::string name, int, int, int);
  /// construct from four charge values
  NamedCandCombinerBase(std::string name, int, int, int, int);
  /// constructor from a selector, specifying optionally to check for charge
  NamedCandCombinerBase(std::string name, bool checkCharge, bool checkOverlap, const std::vector <int> &);
  /// destructor
  virtual ~NamedCandCombinerBase();
  /// return all selected candidate pairs
  std::auto_ptr<reco::NamedCompositeCandidateCollection> 
  combine(const std::vector<reco::CandidatePtrVector> &, string_coll const &) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::NamedCompositeCandidateCollection> 
  combine(const reco::CandidatePtrVector &, 
	  string_coll const &) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::NamedCompositeCandidateCollection> 
  combine(const reco::CandidatePtrVector &, 
	  const reco::CandidatePtrVector &, 
	  string_coll const &) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::NamedCompositeCandidateCollection> 
  combine(const reco::CandidatePtrVector &, 
	  const reco::CandidatePtrVector &, 
	  const reco::CandidatePtrVector &, 
	  string_coll const &) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::NamedCompositeCandidateCollection> 
  combine(const reco::CandidatePtrVector &, 
	  const reco::CandidatePtrVector &, 
	  const reco::CandidatePtrVector &, 
	  const reco::CandidatePtrVector &, 
	  string_coll const &) const;

private:
  /// verify that the two candidate don't overlap and check charge
  bool preselect(const reco::Candidate &, const reco::Candidate &) const;
  /// returns a composite candidate combined from two daughters
  void combine(reco::NamedCompositeCandidate &, 
	       const reco::CandidatePtr &, 
	       const reco::CandidatePtr &,
	       std::string,
	       std::string ) const;
  /// temporary candidate stack
  typedef std::vector<std::pair<std::pair<reco::CandidatePtr, size_t>, 
				std::vector<reco::CandidatePtrVector>::const_iterator> > CandStack;
  typedef std::vector<int> ChargeStack;
  /// returns a composite candidate combined from two daughters
  void combine(size_t collectionIndex, CandStack &, ChargeStack &,
	       string_coll const & names,
	       std::vector<reco::CandidatePtrVector>::const_iterator begin,
	       std::vector<reco::CandidatePtrVector>::const_iterator end,
	       std::auto_ptr<reco::NamedCompositeCandidateCollection> & comps
	       ) const;
  /// select a candidate
  virtual bool select(const reco::Candidate &) const = 0;
  /// select a candidate pair
  virtual bool selectPair(const reco::Candidate & c1, const reco::Candidate & c2) const = 0;
  /// set kinematics to reconstructed composite
  virtual void setup(reco::NamedCompositeCandidate &) const = 0;
  /// add candidate daughter
  virtual void addDaughter(reco::NamedCompositeCandidate & cmp, const reco::CandidatePtr & c, std::string name) const = 0;
  /// flag to specify the checking of electric charge
  bool checkCharge_;
  /// flag to specify the checking of overlaps
  bool checkOverlap_;
  /// electric charges of the daughters
  std::vector<int> dauCharge_;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap_;
  /// Name
  std::string name_;
};

#endif
