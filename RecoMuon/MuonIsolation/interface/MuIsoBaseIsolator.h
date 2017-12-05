#ifndef MuonIsolation_MuIsoBaseIsolator_H
#define MuonIsolation_MuIsoBaseIsolator_H

#include <vector>
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace muonisolation {
  class MuIsoBaseIsolator {

  public:
    typedef reco::IsoDeposit::Veto Veto;
    typedef reco::IsoDeposit::Vetos Vetos;

    struct DepositAndVetos {
      DepositAndVetos(): dep(nullptr), vetos(nullptr) {}
      DepositAndVetos(const reco::IsoDeposit* depA, const Vetos* vetosA = nullptr):
	dep(depA), vetos(vetosA) {}
      const reco::IsoDeposit* dep;
      const Vetos* vetos;
    };
    typedef std::vector<DepositAndVetos> DepositContainer;
  
    enum ResultType {
      ISOL_INT_TYPE = 0,
      ISOL_FLOAT_TYPE,
      ISOL_BOOL_TYPE,
      ISOL_INVALID_TYPE
    };

    class Result {
    public:
      Result() : valInt(-999), valFloat(-999), valBool(false), typeF_(ISOL_INVALID_TYPE) {}
	Result(ResultType typ) : valInt(-999), valFloat(-999.), valBool(false), typeF_(typ) {}
	  
	  template <typename T> T val() const;
	  
	  int valInt;
	  float valFloat;
	  bool valBool;
	  ResultType typeF() const {return typeF_;}

    protected:
	  ResultType typeF_;
    };
    

    virtual ~MuIsoBaseIsolator(){}

    //! Compute and return the isolation variable
    virtual Result result(const DepositContainer& deposits, const edm::Event* = nullptr) const = 0;
    //! Compute and return the isolation variable, with vetoes and the muon
    virtual Result result(const DepositContainer& deposits, const reco::Candidate& muon, const edm::Event* = nullptr) const {
      return result(deposits);
    }
    //! Compute and return the isolation variable, with vetoes and the muon
    virtual Result result(const DepositContainer& deposits, const reco::Track& muon, const edm::Event* = nullptr) const {
      return result(deposits);
    }

    virtual ResultType resultType() const = 0;

  };

  template<> inline int MuIsoBaseIsolator::Result::val<int>() const { return valInt;}
  template<> inline float MuIsoBaseIsolator::Result::val<float>() const { return valFloat;}
  template<> inline bool MuIsoBaseIsolator::Result::val<bool>() const { return valBool;}
  
}
#endif

