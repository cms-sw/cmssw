#ifndef MuonIsolationProducers_CandIsolatorFromDeposits_H
#define MuonIsolationProducers_CandIsolatorFromDeposits_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "PhysicsTools/Utilities/interface/StringObjectFunction.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include <string>

namespace edm { class Event; }
namespace edm { class EventSetup; }

class CandIsolatorFromDeposits : public edm::EDProducer {

public:
  //enum Mode { Sum, SumRelative, Max, MaxRelative, Count };
  enum Mode { Sum, SumRelative, Count };
  CandIsolatorFromDeposits(const edm::ParameterSet&);

  virtual ~CandIsolatorFromDeposits();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  class SingleDeposit {
    public:
        SingleDeposit(const edm::ParameterSet &) ;
        void cleanup() ;
        void open(const edm::Event &iEvent) ;
        double compute(const reco::CandidateBaseRef &cand) ;
        const reco::CandIsoDepositAssociationVector & vector() { return *hDeps_; }
    private:
        Mode mode_;
        edm::InputTag src_;
        double deltaR_;
        bool   usesFunction_;
        double weight_;
        StringObjectFunction<reco::Candidate> weightExpr_;
        reco::isodeposit::AbsVetos vetos_;
        bool   skipDefaultVeto_; 
        edm::Handle<reco::CandIsoDepositAssociationVector> hDeps_; // transient
  };
  // datamembers
  std::vector<SingleDeposit> sources_;

};
#endif
