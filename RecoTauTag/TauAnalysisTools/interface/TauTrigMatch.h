#ifndef TauTrigMatch_h
#define TauTrigMatch_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

class TauTrigMatch {
  public:
    // Default needed for persistency
    TauTrigMatch( ); 

    TauTrigMatch( const pat::Tau* tagTau, std::vector<const reco::Candidate*>* trigObj, unsigned int index, unsigned int nTotalObjects );
    
    // Get tag tau object
    const pat::Tau* tagTau() const;

    // Get match status of trigger filter object
    bool trigObjMatch(int a) const;

    // Get status of Discriminator 
    bool tagTauID(std::string DiscriminatorName) const;

    // Get the index of this match in the event.
    unsigned int index() const;
    // Get the total number of reco objects in this event.
    unsigned int nTotalObjects() const;

  private:
    const pat::Tau* tagTau_;
    std::vector<const reco::Candidate*>* trigObj_;
    unsigned int index_;
    unsigned int nTotalObjects_;

};

#endif /* end of include guard: TauTrigMatch_h */
