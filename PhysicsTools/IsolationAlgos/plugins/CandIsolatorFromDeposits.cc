#include "PhysicsTools/IsolationAlgos/plugins/CandIsolatorFromDeposits.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <boost/regex.hpp>

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositVetoFactory.h"

using namespace edm;
using namespace reco;
using namespace reco::isodeposit;

bool isNumber(const std::string &str) {
   static boost::regex re("^[+-]?(\\d+\\.?|\\d*\\.\\d*)$");
   return regex_match(str.c_str(), re);
}
double toNumber(const std::string &str) {
    return atof(str.c_str());
}

CandIsolatorFromDeposits::SingleDeposit::SingleDeposit(const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC) :
  srcToken_(iC.consumes<reco::IsoDepositMap>(iConfig.getParameter<edm::InputTag>("src"))),
  deltaR_(iConfig.getParameter<double>("deltaR")),
  weightExpr_(iConfig.getParameter<std::string>("weight")),
  skipDefaultVeto_(iConfig.getParameter<bool>("skipDefaultVeto"))
						      //,vetos_(new AbsVetos())
{
  std::string mode = iConfig.getParameter<std::string>("mode");
  if (mode == "sum") mode_ = Sum;
  else if (mode == "sumRelative") mode_ = SumRelative;
  else if (mode == "sum2") mode_ = Sum2;
  else if (mode == "sum2Relative") mode_ = Sum2Relative;
  else if (mode == "max") mode_ = Max;
  else if (mode == "maxRelative") mode_ = MaxRelative;
  else if (mode == "nearestDR") mode_ = NearestDR;
  else if (mode == "count") mode_ = Count;
  else if (mode == "meanDR") mode_ = MeanDR;
  else if (mode == "sumDR") mode_ = SumDR;
  else throw cms::Exception("Not Implemented") << "Mode '" << mode << "' not implemented. " <<
    "Supported modes are 'sum', 'sumRelative', 'count'." <<
    //"Supported modes are 'sum', 'sumRelative', 'max', 'maxRelative', 'count'." << // TODO: on request only
    "New methods can be easily implemented if requested.";
  typedef std::vector<std::string> vstring;
  vstring vetos = iConfig.getParameter< vstring >("vetos");
  reco::isodeposit::EventDependentAbsVeto *evdep;
  for (vstring::const_iterator it = vetos.begin(), ed = vetos.end(); it != ed; ++it) {
    vetos_.push_back(IsoDepositVetoFactory::make(it->c_str(), evdep, iC));
    if (evdep) evdepVetos_.push_back(evdep);
  }
  std::string weight = iConfig.getParameter<std::string>("weight");
  if (isNumber(weight)) {
    //std::cout << "Weight is a simple number, " << toNumber(weight) << std::endl;
    weight_ = toNumber(weight);
    usesFunction_ = false;
  } else {
    usesFunction_ = true;
    //std::cout << "Weight is a function, this might slow you down... " << std::endl;
  }
  //std::cout << "CandIsolatorFromDeposits::SingleDeposit::SingleDeposit: Total of " << vetos_.size() << " vetos" << std::endl;
}
void CandIsolatorFromDeposits::SingleDeposit::cleanup() {
    for (AbsVetos::iterator it = vetos_.begin(), ed = vetos_.end(); it != ed; ++it) {
        delete *it;
    }
    vetos_.clear();
    // NOTE: we DON'T have to delete the evdepVetos_, they have already been deleted above. We just clear the vectors
    evdepVetos_.clear();
}
void CandIsolatorFromDeposits::SingleDeposit::open(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
    iEvent.getByToken(srcToken_, hDeps_);
    for (EventDependentAbsVetos::iterator it = evdepVetos_.begin(), ed = evdepVetos_.end(); it != ed; ++it) {
        (*it)->setEvent(iEvent,iSetup);
    }
}

double CandIsolatorFromDeposits::SingleDeposit::compute(const reco::CandidateBaseRef &cand) {
    const IsoDeposit &dep = (*hDeps_)[cand];
    double eta = dep.eta(), phi = dep.phi(); // better to center on the deposit direction
                                             // that could be, e.g., the impact point at calo
    for (AbsVetos::iterator it = vetos_.begin(), ed = vetos_.end(); it != ed; ++it) {
        (*it)->centerOn(eta, phi);
    }
    double weight = (usesFunction_ ? weightExpr_(*cand) : weight_);
    switch (mode_) {
        case Count:        return weight * dep.countWithin(deltaR_, vetos_, skipDefaultVeto_);
        case Sum:          return weight * dep.sumWithin(deltaR_, vetos_, skipDefaultVeto_);
        case SumRelative:  return weight * dep.sumWithin(deltaR_, vetos_, skipDefaultVeto_) / dep.candEnergy() ;
        case Sum2:         return weight * dep.sum2Within(deltaR_, vetos_, skipDefaultVeto_);
        case Sum2Relative: return weight * dep.sum2Within(deltaR_, vetos_, skipDefaultVeto_) / (dep.candEnergy() * dep.candEnergy()) ;
        case Max:          return weight * dep.maxWithin(deltaR_, vetos_, skipDefaultVeto_);
        case NearestDR:    return weight * dep.nearestDR(deltaR_, vetos_, skipDefaultVeto_);
        case MaxRelative:  return weight * dep.maxWithin(deltaR_, vetos_, skipDefaultVeto_) / dep.candEnergy() ;
        case MeanDR:  return weight * dep.algoWithin<reco::IsoDeposit::MeanDRAlgo>(deltaR_, vetos_, skipDefaultVeto_);
        case SumDR:  return weight * dep.algoWithin<reco::IsoDeposit::SumDRAlgo>(deltaR_, vetos_, skipDefaultVeto_);
    }
    throw cms::Exception("Logic error") << "Should not happen at " << __FILE__ << ", line " << __LINE__; // avoid gcc warning
}


/// constructor with config
CandIsolatorFromDeposits::CandIsolatorFromDeposits(const ParameterSet& par) {
  typedef std::vector<edm::ParameterSet> VPSet;
  VPSet depPSets = par.getParameter<VPSet>("deposits");
  for (VPSet::const_iterator it = depPSets.begin(), ed = depPSets.end(); it != ed; ++it) {
    sources_.push_back(SingleDeposit(*it, consumesCollector()));
  }
  if (sources_.size() == 0) throw cms::Exception("Configuration Error") << "Please specify at least one deposit!";
  produces<CandDoubleMap>();
}

/// destructor
CandIsolatorFromDeposits::~CandIsolatorFromDeposits() {
  std::vector<SingleDeposit>::iterator it, begin = sources_.begin(), end = sources_.end();
  for (it = begin; it != end; ++it) it->cleanup();
}

/// build deposits
void CandIsolatorFromDeposits::produce(Event& event, const EventSetup& eventSetup){

  std::vector<SingleDeposit>::iterator it, begin = sources_.begin(), end = sources_.end();
  for (it = begin; it != end; ++it) it->open(event, eventSetup);

  const IsoDepositMap & map = begin->map();

  if (map.size()==0) { // !!???
        event.put(std::auto_ptr<CandDoubleMap>(new CandDoubleMap()));
        return;
  }
  std::auto_ptr<CandDoubleMap> ret(new CandDoubleMap());
  CandDoubleMap::Filler filler(*ret);

  typedef reco::IsoDepositMap::const_iterator iterator_i;
  typedef reco::IsoDepositMap::container::const_iterator iterator_ii;
  iterator_i depI = map.begin();
  iterator_i depIEnd = map.end();
  for (; depI != depIEnd; ++depI){
    std::vector<double> retV(depI.size(),0);
    edm::Handle<edm::View<reco::Candidate> > candH;
    event.get(depI.id(), candH);
    const edm::View<reco::Candidate>& candV = *candH;

    iterator_ii depII = depI.begin();
    iterator_ii depIIEnd = depI.end();
    size_t iRet = 0;
    for (; depII != depIIEnd; ++depII,++iRet){
      double sum=0;
      for (it = begin; it != end; ++it) sum += it->compute(candV.refAt(iRet));
      retV[iRet] = sum;
    }
    filler.insert(candH, retV.begin(), retV.end());
  }
  filler.fill();
  event.put(ret);
}

DEFINE_FWK_MODULE( CandIsolatorFromDeposits );
