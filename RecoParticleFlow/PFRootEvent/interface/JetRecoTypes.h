#ifndef JetAlgorithms_JetRecoTypes_h
#define JetAlgorithms_JetRecoTypes_h

// Types used in Jet Reconstruction
// F.Ratnikov, UMd
// $Id: JetRecoTypes.h,v 1.1 2009/08/24 14:35:59 srappocc Exp $

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
  
  //CorrectedIntexedCandidate for use with corrected CaloTower to Vertex
  class CorrectedIndexedCandidate {
    public:
    typedef reco::Candidate value_type;
    CorrectedIndexedCandidate () 
      : mCandidate (0), oCandidate(0), mIndex(0) {}
    CorrectedIndexedCandidate (const value_type *cCandidate,const value_type* fCandidate, unsigned fIndex)  
      : mCandidate (cCandidate), oCandidate(fCandidate), mIndex (fIndex) {}
    CorrectedIndexedCandidate (const value_type* fCandidate, unsigned fIndex)  
      : mCandidate (fCandidate), mIndex (fIndex) {}
    inline unsigned index () const {return mIndex;}
    inline const value_type& operator*() const {return *mCandidate;}
    inline const value_type* operator->() const {return mCandidate;}
    inline const bool operator==(const CorrectedIndexedCandidate& other) const {return mCandidate == other.mCandidate;}
    inline const bool operator!=(const CorrectedIndexedCandidate& other) const {return !operator==(other);}
    inline const value_type* get() const {return mCandidate;}
    inline const value_type* getOriginal() const {return oCandidate;}
    inline void setBase(const value_type* a){mCandidate=a;}
    inline void setOriginal(const value_type* a){oCandidate=a;}
    inline void setIndex(unsigned nIndex){mIndex=nIndex;}
    inline bool operator! () const {return !mCandidate;}
  private:
    const value_type* mCandidate; //version used in algorithms
    const value_type* oCandidate; //original, use for constituents
    unsigned mIndex;
  };



    typedef CorrectedIndexedCandidate InputItem;
  //typedef IndexedCandidate InputItem;
  typedef std::vector<InputItem> InputCollection;
  typedef std::vector <ProtoJet> OutputCollection;
  typedef reco::CandidateBaseRefVector JetConstituents;
}
#endif
