#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVeto.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositVetoFactory.h"

#include <memory>
#include <regex>
#include <string>

class PFCandIsolatorFromDeposits : public edm::stream::EDProducer<> {
public:
  typedef edm::ValueMap<double> CandDoubleMap;

  enum Mode { Sum, SumRelative, Sum2, Sum2Relative, Max, MaxRelative, Count, NearestDR };
  PFCandIsolatorFromDeposits(const edm::ParameterSet &);

  ~PFCandIsolatorFromDeposits() override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  class SingleDeposit {
  public:
    SingleDeposit(const edm::ParameterSet &, edm::ConsumesCollector &&iC);
    void cleanup();
    void open(const edm::Event &iEvent, const edm::EventSetup &iSetup);
    double compute(const reco::CandidateBaseRef &cand);
    const reco::IsoDepositMap &map() { return *hDeps_; }

  private:
    Mode mode_;
    edm::EDGetTokenT<reco::IsoDepositMap> srcToken_;
    double deltaR_;
    bool usesFunction_;
    double weight_;

    StringObjectFunction<reco::Candidate> weightExpr_;
    reco::isodeposit::AbsVetos barrelVetos_;
    reco::isodeposit::AbsVetos endcapVetos_;
    reco::isodeposit::EventDependentAbsVetos evdepVetos_;  // note: these are a subset of the above. Don't delete twice!
    bool skipDefaultVeto_;
    bool usePivotForBarrelEndcaps_;
    edm::Handle<reco::IsoDepositMap> hDeps_;  // transient

    bool isNumber(const std::string &str) const;
    double toNumber(const std::string &str) const;
  };
  // datamembers
  std::vector<SingleDeposit> sources_;
};

using namespace edm;
using namespace reco;
using namespace reco::isodeposit;

bool PFCandIsolatorFromDeposits::SingleDeposit::isNumber(const std::string &str) const {
  static const std::regex re("^[+-]?(\\d+\\.?|\\d*\\.\\d*)$");
  return regex_match(str.c_str(), re);
}
double PFCandIsolatorFromDeposits::SingleDeposit::toNumber(const std::string &str) const { return atof(str.c_str()); }

PFCandIsolatorFromDeposits::SingleDeposit::SingleDeposit(const edm::ParameterSet &iConfig, edm::ConsumesCollector &&iC)
    : srcToken_(iC.consumes<reco::IsoDepositMap>(iConfig.getParameter<edm::InputTag>("src"))),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      weightExpr_(iConfig.getParameter<std::string>("weight")),
      skipDefaultVeto_(iConfig.getParameter<bool>("skipDefaultVeto")),
      usePivotForBarrelEndcaps_(iConfig.getParameter<bool>("PivotCoordinatesForEBEE"))
//,vetos_(new AbsVetos())
{
  std::string mode = iConfig.getParameter<std::string>("mode");
  if (mode == "sum")
    mode_ = Sum;
  else if (mode == "sumRelative")
    mode_ = SumRelative;
  else if (mode == "sum2")
    mode_ = Sum2;
  else if (mode == "sum2Relative")
    mode_ = Sum2Relative;
  else if (mode == "max")
    mode_ = Max;
  else if (mode == "maxRelative")
    mode_ = MaxRelative;
  else if (mode == "nearestDR")
    mode_ = NearestDR;
  else if (mode == "count")
    mode_ = Count;
  else
    throw cms::Exception("Not Implemented") << "Mode '" << mode << "' not implemented. "
                                            << "Supported modes are 'sum', 'sumRelative', 'count'." <<
        //"Supported modes are 'sum', 'sumRelative', 'max', 'maxRelative', 'count'." << // TODO: on request only
        "New methods can be easily implemented if requested.";
  typedef std::vector<std::string> vstring;
  vstring vetos = iConfig.getParameter<vstring>("vetos");
  reco::isodeposit::EventDependentAbsVeto *evdep = nullptr;
  static const std::regex ecalSwitch("^Ecal(Barrel|Endcaps):(.*)");

  for (vstring::const_iterator it = vetos.begin(), ed = vetos.end(); it != ed; ++it) {
    std::cmatch match;
    // in that case, make two series of vetoes
    if (usePivotForBarrelEndcaps_) {
      if (regex_match(it->c_str(), match, ecalSwitch)) {
        if (match[1] == "Barrel") {
          //	    std::cout << " Adding Barrel veto " << std::string(match[2]) << std::endl;
          barrelVetos_.push_back(
              IsoDepositVetoFactory::make(std::string(match[2]).c_str(), evdep, iC));  // I don't know a better syntax
        }
        if (match[1] == "Endcaps") {
          //	    std::cout << " Adding Endcap veto " << std::string(match[2]) << std::endl;
          endcapVetos_.push_back(IsoDepositVetoFactory::make(std::string(match[2]).c_str(), evdep, iC));
        }
      } else {
        barrelVetos_.push_back(IsoDepositVetoFactory::make(it->c_str(), evdep, iC));
        endcapVetos_.push_back(IsoDepositVetoFactory::make(it->c_str(), evdep, iC));
      }
    } else {
      //only one serie of vetoes, just barrel
      barrelVetos_.push_back(IsoDepositVetoFactory::make(it->c_str(), evdep, iC));
    }
    if (evdep)
      evdepVetos_.push_back(evdep);
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
  //std::cout << "PFCandIsolatorFromDeposits::SingleDeposit::SingleDeposit: Total of " << vetos_.size() << " vetos" << std::endl;
}
void PFCandIsolatorFromDeposits::SingleDeposit::cleanup() {
  for (AbsVetos::iterator it = barrelVetos_.begin(), ed = barrelVetos_.end(); it != ed; ++it) {
    delete *it;
  }
  for (AbsVetos::iterator it = endcapVetos_.begin(), ed = endcapVetos_.end(); it != ed; ++it) {
    delete *it;
  }
  barrelVetos_.clear();
  endcapVetos_.clear();
  // NOTE: we DON'T have to delete the evdepVetos_, they have already been deleted above. We just clear the vectors
  evdepVetos_.clear();
}
void PFCandIsolatorFromDeposits::SingleDeposit::open(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  iEvent.getByToken(srcToken_, hDeps_);
  for (EventDependentAbsVetos::iterator it = evdepVetos_.begin(), ed = evdepVetos_.end(); it != ed; ++it) {
    (*it)->setEvent(iEvent, iSetup);
  }
}

double PFCandIsolatorFromDeposits::SingleDeposit::compute(const reco::CandidateBaseRef &cand) {
  const IsoDeposit &dep = (*hDeps_)[cand];
  double eta = dep.eta(), phi = dep.phi();  // better to center on the deposit direction
                                            // that could be, e.g., the impact point at calo
  bool barrel = true;
  if (usePivotForBarrelEndcaps_) {
    const reco::PFCandidate *myPFCand = dynamic_cast<const reco::PFCandidate *>(&(*cand));
    if (myPFCand) {
      // exact barrel boundary
      barrel = fabs(myPFCand->positionAtECALEntrance().eta()) < 1.479;
    } else {
      const reco::RecoCandidate *myRecoCand = dynamic_cast<const reco::RecoCandidate *>(&(*cand));
      if (myRecoCand) {
        // not optimal. isEB should be used.
        barrel = (fabs(myRecoCand->superCluster()->eta()) < 1.479);
      }
    }
  }
  // if ! usePivotForBarrelEndcaps_ only the barrel series is used, which does not prevent the vetoes do be different in barrel & endcaps
  reco::isodeposit::AbsVetos *vetos = (barrel) ? &barrelVetos_ : &endcapVetos_;

  for (AbsVetos::iterator it = vetos->begin(), ed = vetos->end(); it != ed; ++it) {
    (*it)->centerOn(eta, phi);
  }
  double weight = (usesFunction_ ? weightExpr_(*cand) : weight_);
  switch (mode_) {
    case Count:
      return weight * dep.countWithin(deltaR_, *vetos, skipDefaultVeto_);
    case Sum:
      return weight * dep.sumWithin(deltaR_, *vetos, skipDefaultVeto_);
    case SumRelative:
      return weight * dep.sumWithin(deltaR_, *vetos, skipDefaultVeto_) / dep.candEnergy();
    case Sum2:
      return weight * dep.sum2Within(deltaR_, *vetos, skipDefaultVeto_);
    case Sum2Relative:
      return weight * dep.sum2Within(deltaR_, *vetos, skipDefaultVeto_) / (dep.candEnergy() * dep.candEnergy());
    case Max:
      return weight * dep.maxWithin(deltaR_, *vetos, skipDefaultVeto_);
    case NearestDR:
      return weight * dep.nearestDR(deltaR_, *vetos, skipDefaultVeto_);
    case MaxRelative:
      return weight * dep.maxWithin(deltaR_, *vetos, skipDefaultVeto_) / dep.candEnergy();
  }
  throw cms::Exception("Logic error") << "Should not happen at " << __FILE__ << ", line "
                                      << __LINE__;  // avoid gcc warning
}

/// constructor with config
PFCandIsolatorFromDeposits::PFCandIsolatorFromDeposits(const ParameterSet &par) {
  typedef std::vector<edm::ParameterSet> VPSet;
  VPSet depPSets = par.getParameter<VPSet>("deposits");
  for (VPSet::const_iterator it = depPSets.begin(), ed = depPSets.end(); it != ed; ++it) {
    sources_.push_back(SingleDeposit(*it, consumesCollector()));
  }
  if (sources_.empty())
    throw cms::Exception("Configuration Error") << "Please specify at least one deposit!";
  produces<CandDoubleMap>();
}

/// destructor
PFCandIsolatorFromDeposits::~PFCandIsolatorFromDeposits() {
  std::vector<SingleDeposit>::iterator it, begin = sources_.begin(), end = sources_.end();
  for (it = begin; it != end; ++it)
    it->cleanup();
}

/// build deposits
void PFCandIsolatorFromDeposits::produce(Event &event, const EventSetup &eventSetup) {
  std::vector<SingleDeposit>::iterator it, begin = sources_.begin(), end = sources_.end();
  for (it = begin; it != end; ++it)
    it->open(event, eventSetup);

  const IsoDepositMap &map = begin->map();

  if (map.empty()) {  // !!???
    event.put(std::make_unique<CandDoubleMap>());
    return;
  }
  std::unique_ptr<CandDoubleMap> ret(new CandDoubleMap());
  CandDoubleMap::Filler filler(*ret);

  typedef reco::IsoDepositMap::const_iterator iterator_i;
  typedef reco::IsoDepositMap::container::const_iterator iterator_ii;
  iterator_i depI = map.begin();
  iterator_i depIEnd = map.end();
  for (; depI != depIEnd; ++depI) {
    std::vector<double> retV(depI.size(), 0);
    edm::Handle<edm::View<reco::Candidate> > candH;
    event.get(depI.id(), candH);
    const edm::View<reco::Candidate> &candV = *candH;

    iterator_ii depII = depI.begin();
    iterator_ii depIIEnd = depI.end();
    size_t iRet = 0;
    for (; depII != depIIEnd; ++depII, ++iRet) {
      double sum = 0;
      for (it = begin; it != end; ++it)
        sum += it->compute(candV.refAt(iRet));
      retV[iRet] = sum;
    }
    filler.insert(candH, retV.begin(), retV.end());
  }
  filler.fill();
  event.put(std::move(ret));
}

DEFINE_FWK_MODULE(PFCandIsolatorFromDeposits);
