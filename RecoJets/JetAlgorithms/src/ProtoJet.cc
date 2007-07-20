//  ProtJet2.cc
//  Revision History:  R. Harris 10/19/05  Modified to work with real CaloTowers from Jeremy Mans
//

#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include <vector>
#include <algorithm>               // include STL algorithm implementations
#include <numeric>		   // For the use of numeric

namespace {
  class ERecombination {
    public: double weight (const reco::Candidate& c) {return c.energy();}
  };
  class EtRecombination {
    public: double weight (const reco::Candidate& c) {return c.et();}
  };
  class PRecombination {
    public: double weight (const reco::Candidate& c) {return c.p();}
  };
  class PtRecombination {
    public: double weight (const reco::Candidate& c) {return c.pt();}
  };

  template <class T>
   ProtoJet::LorentzVector calculateLorentzVectorRecombination(const ProtoJet::Constituents& fConstituents) {
    T weightMaker;
    ProtoJet::LorentzVector result (0,0,0,0);
    double weights = 0;
    for(ProtoJet::Constituents::const_iterator i = fConstituents.begin(); i !=  fConstituents.end(); ++i) {
      const ProtoJet::Constituent c = *i;
      double weight = weightMaker.weight (*c);
      result += c->p4() * weight;
      weights += weight;
    } //end of loop over the jet constituents
    result = result / weights;
    return result;
  }

  struct CandidateRefGreaterByEt {
    bool operator() (const reco::CandidateRef& first, const reco::CandidateRef& second) {
      NumericSafeGreaterByEt<reco::Candidate> comparator;
      return comparator (*first, *second);
    }
  };

}

ProtoJet::ProtoJet()
  : mOrdered (false) {}

ProtoJet::ProtoJet(const Constituents& fConstituents) 
  : mConstituents (fConstituents),
    mOrdered (false)
{
  calculateLorentzVector(); 
}//end of constructor

ProtoJet::ProtoJet(const LorentzVector& fP4, const Constituents& fConstituents) 
  : mP4 (fP4), 
    mConstituents (fConstituents),
    mOrdered (false)
{}

const ProtoJet::Constituents& ProtoJet::getTowerList() {
  reorderTowers ();
  return mConstituents;
}
  
const ProtoJet::Constituents ProtoJet::getTowerList() const {
  if (mOrdered) return mConstituents;

  Constituents result = mConstituents;
  CandidateRefGreaterByEt comparator;
  std::sort (result.begin(), result.end(), comparator);
  return result;
}

void ProtoJet::putTowers(const Constituents& towers) {
  mConstituents = towers; 
  mOrdered = false;
  calculateLorentzVector();
}

void ProtoJet::reorderTowers () {
  if (!mOrdered) {
    CandidateRefGreaterByEt comparator;
    std::sort (mConstituents.begin(), mConstituents.end(), comparator); 
    mOrdered = true;
  }
}

void ProtoJet::calculateLorentzVectorERecombination() {
  mP4 = LorentzVector (0,0,0,0);
  for(Constituents::const_iterator i = mConstituents.begin(); i !=  mConstituents.end(); ++i) {
    const Constituent c = *i;
    mP4 += c->p4();
  } //end of loop over the jet constituents
}

void ProtoJet::calculateLorentzVectorEtRecombination() {
  mP4 = calculateLorentzVectorRecombination <EtRecombination> (mConstituents);
}


