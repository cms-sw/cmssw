#ifndef MuonIsolationProducers_CandIsolatorFromDeposits_H
#define MuonIsolationProducers_CandIsolatorFromDeposits_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVeto.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CandIsolatorFromDeposits : public edm::EDProducer {

public:
  typedef edm::ValueMap<double> CandDoubleMap;

  enum Mode { Sum, SumRelative, Sum2, Sum2Relative, Max, MaxRelative, Count, NearestDR,MeanDR,SumDR };
  CandIsolatorFromDeposits(const edm::ParameterSet&);

  virtual ~CandIsolatorFromDeposits();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  class SingleDeposit {
    public:
        SingleDeposit(const edm::ParameterSet &) ;
        void cleanup() ;
        void open(const edm::Event &iEvent, const edm::EventSetup &iSetup) ;
        double compute(const reco::CandidateBaseRef &cand) ;
        const reco::IsoDepositMap & map() { return *hDeps_; }
    private:
        Mode mode_;
        edm::InputTag src_;
        double deltaR_;
        bool   usesFunction_;
        double weight_;
        StringObjectFunction<reco::Candidate> weightExpr_;
        reco::isodeposit::AbsVetos vetos_;
        reco::isodeposit::EventDependentAbsVetos evdepVetos_; // note: these are a subset of the above. Don't delete twice!
        bool   skipDefaultVeto_; 
        edm::Handle<reco::IsoDepositMap> hDeps_; // transient
  };
  // datamembers
  std::vector<SingleDeposit> sources_;

};
#endif
