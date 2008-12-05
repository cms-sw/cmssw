
#ifndef DataFormats_PatCandidates_MHT_h
#define DataFormats_PatCandidates_MHT_h
                                                                                                                   
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
                                                                                                                   
namespace pat {
                                                                                                                   
  class MHT : public reco::CompositeRefBaseCandidate {
  public:
    MHT () {}
    MHT (const Candidate::LorentzVector& p4, double ht, double signif) :
      CompositeRefBaseCandidate(0,p4), ht_(ht), significance_(signif) {}
    virtual ~MHT () {}
                                                                                                                   
    double mht() const {return pt();};
    double phi() const {return phi();};
    double ht()  const {return ht_;};
    double significance() const {return significance_;};
    double error()  const{return 0.5*significance()*mht()*mht();};
                                                                                                                   
  private:
                                                                                                                   
    double ht_;
    double significance_;
                                                                                                                   
                                                                                                                   
  };
                                                                                                                   
  typedef std::vector<pat::MHT> MHTCollection;
}
                                                                                                                   
#endif






