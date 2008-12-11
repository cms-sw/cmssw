#ifndef JetAlgorithms_JetRecoTypes_h
#define JetAlgorithms_JetRecoTypes_h

// Types used in Jet Reconstruction
// F.Ratnikov, UMd
// $Id: JetRecoTypes.h,v 1.4 2007/08/20 17:53:32 fedor Exp $

#include "DataFormats/Candidate/interface/CandidateFwd.h"
class ProtoJet;

namespace JetReco {
  class IndexedCandidate {
  public:
    typedef reco::Candidate value_type;
    IndexedCandidate () 
      : mCandidate (0), mIndex(0) {}
    IndexedCandidate (const value_type* fCandidate, unsigned fIndex)  
      : mCandidate (fCandidate), mIndex (fIndex) {}
    inline unsigned index () const {return mIndex;}
    inline const value_type& operator*() const {return *mCandidate;}
    inline const value_type* operator->() const {return mCandidate;}
    inline const bool operator==(const IndexedCandidate& other) const {return mCandidate == other.mCandidate;}
    inline const bool operator!=(const IndexedCandidate& other) const {return !operator==(other);}
    inline const value_type* get() const {return mCandidate;}
    inline bool operator! () const {return !mCandidate;}
  private:
    const value_type* mCandidate;
    unsigned mIndex;
  };
  
  typedef IndexedCandidate InputItem;
  typedef std::vector<InputItem> InputCollection;
  typedef std::vector <ProtoJet> OutputCollection;
  typedef reco::CandidateBaseRefVector JetConstituents;
}
#endif
