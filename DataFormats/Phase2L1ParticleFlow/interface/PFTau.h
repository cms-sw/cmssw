#ifndef DataFormats_Phase2L1ParticleFlow_PFTau_h
#define DataFormats_Phase2L1ParticleFlow_PFTau_h

#include <algorithm>
#include <vector>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

namespace l1t
{

  class PFTau : public L1Candidate {
      public:
          PFTau() {}
	  enum  { unidentified=0, oneprong=1, oneprongpi0=2, threeprong=3};
          PFTau(const LorentzVector & p, float iso=-1, float fulliso=-1, int id=0,int hwpt=0, int hweta=0, int hwphi=0) :
   	  PFTau(PolarLorentzVector(p), iso, id, hwpt, hweta, hwphi) {}
		PFTau(const PolarLorentzVector & p, float iso=-1, float fulliso=-1,int id=0, int hwpt=0, int hweta=0, int hwphi=0) ; 
          float chargedIso() const { return iso_; }
          float fullIso()    const { return fullIso_; }
	  int   id()         const { return id_; }
	  bool  passLooseNN()  const { return iso_*(0.1+0.2*(min(pt(),100.)))*1./20.1 > 0.05;}
	  bool  passLoosePF()  const { return fullIso_ < 10.0;}
	  bool  passTightNN()  const { return iso_*(0.1+0.2*(min(pt(),100.)))*1./20.1 > 0.25;}
	  bool  passTightPF()  const { return fullIso_ < 5.0;}
	  

      private:
          float iso_;
          float fullIso_;
	  int   id_;
  };
  
  typedef std::vector<l1t::PFTau> PFTauCollection;
}
#endif

