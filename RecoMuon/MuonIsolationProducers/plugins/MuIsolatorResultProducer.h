#ifndef MuonIsolationProducers_MuIsolatorResultProducer_H
#define MuonIsolationProducers_MuIsolatorResultProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <string>

namespace edm {
  class EventSetup;
}

struct muisorhelper {
  typedef muonisolation::MuIsoBaseIsolator Isolator;
  typedef Isolator::Result Result;
  typedef Isolator::ResultType ResultType;
  typedef std::vector<Result> Results;
  typedef Isolator::DepositContainer DepositContainer;

  template <typename BT>
  class CandMap {
  public:
    typedef typename edm::RefToBase<BT> key_type;
    typedef typename edm::Handle<edm::View<BT>> handle_type;
    typedef DepositContainer value_type;
    typedef std::pair<key_type, value_type> pair_type;
    typedef typename std::vector<pair_type> map_type;
    typedef typename map_type::iterator iterator;

    map_type& get() { return cMap_; }
    const map_type& get() const { return cMap_; }
    const handle_type& handle() const { return handle_; }
    void setHandle(const handle_type& rhs) { handle_ = rhs; }

  private:
    map_type cMap_;
    handle_type handle_;
  };
};

//! BT == base type
template <typename BT = reco::Candidate>
class MuIsolatorResultProducer : public edm::stream::EDProducer<> {
public:
  MuIsolatorResultProducer(const edm::ParameterSet&);

  ~MuIsolatorResultProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  typedef muisorhelper::Isolator Isolator;
  typedef muisorhelper::Result Result;
  typedef muisorhelper::ResultType ResultType;
  typedef muisorhelper::Results Results;
  typedef muisorhelper::DepositContainer DepositContainer;

  typedef muisorhelper::CandMap<BT> CandMap;

  struct DepositConf {
    edm::InputTag tag;
    double weight;
    double threshold;
  };

  struct VetoCuts {
    bool selectAll;
    double muAbsEtaMax;
    double muPtMin;
    double muAbsZMax;
    double muD0Max;
  };

  void callWhatProduces();

  unsigned int initAssociation(edm::Event& event, CandMap& candMapT) const;

  void initVetos(reco::TrackBase::Point const& theBeam,
                 std::vector<reco::IsoDeposit::Vetos*>& vetos,
                 CandMap& candMap) const;

  template <typename RT>
  void writeOutImpl(edm::Event& event, const CandMap& candMapT, const Results& results) const;

  void writeOut(edm::Event& event, const CandMap& candMap, const Results& results) const;

  edm::ParameterSet theConfig;
  std::vector<DepositConf> theDepositConfs;

  //!choose which muon vetos should be removed from all deposits
  bool theRemoveOtherVetos;
  VetoCuts theVetoCuts;

  //!the isolator
  Isolator* theIsolator;
  ResultType theResultType;

  //! beam spot
  std::string theBeamlineOption;
  edm::InputTag theBeamSpotLabel;
};

//! actually do the writing here
template <typename BT>
template <typename RT>
inline void MuIsolatorResultProducer<BT>::writeOutImpl(edm::Event& event,
                                                       const CandMap& candMapT,
                                                       const Results& results) const {
  //! make an output vec of what's to be written with a concrete type
  std::vector<RT> resV(results.size());
  for (unsigned int i = 0; i < resV.size(); ++i)
    resV[i] = results[i].val<RT>();
  auto outMap = std::make_unique<edm::ValueMap<RT>>();
  typename edm::ValueMap<RT>::Filler filler(*outMap);

  //! fill/insert of non-empty values only
  if (!candMapT.get().empty()) {
    filler.insert(candMapT.handle(), resV.begin(), resV.end());
    filler.fill();
  }

  event.put(std::move(outMap));
}

//! choose which result type to write here
template <typename BT>
inline void MuIsolatorResultProducer<BT>::writeOut(edm::Event& event,
                                                   const CandMap& candMapT,
                                                   const Results& results) const {
  std::string metname = "RecoMuon|MuonIsolationProducers";
  LogDebug(metname) << "Before calling writeOutMap  with result type " << theIsolator->resultType();

  if (theResultType == Isolator::ISOL_INT_TYPE)
    writeOutImpl<int>(event, candMapT, results);
  if (theResultType == Isolator::ISOL_FLOAT_TYPE)
    writeOutImpl<float>(event, candMapT, results);
  if (theResultType == Isolator::ISOL_BOOL_TYPE)
    writeOutImpl<bool>(event, candMapT, results);
}

//! declare what's going to be produced
template <typename BT>
inline void MuIsolatorResultProducer<BT>::callWhatProduces() {
  if (theResultType == Isolator::ISOL_FLOAT_TYPE)
    produces<edm::ValueMap<float>>();
  if (theResultType == Isolator::ISOL_INT_TYPE)
    produces<edm::ValueMap<int>>();
  if (theResultType == Isolator::ISOL_BOOL_TYPE)
    produces<edm::ValueMap<bool>>();
}

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "RecoMuon/MuonIsolation/interface/IsolatorByDeposit.h"
#include "RecoMuon/MuonIsolation/interface/IsolatorByDepositCount.h"
#include "RecoMuon/MuonIsolation/interface/IsolatorByNominalEfficiency.h"

#include <string>

//! constructor with config
template <typename BT>
MuIsolatorResultProducer<BT>::MuIsolatorResultProducer(const edm::ParameterSet& par)
    : theConfig(par), theRemoveOtherVetos(par.getParameter<bool>("RemoveOtherVetos")), theIsolator(nullptr) {
  LogDebug("RecoMuon|MuonIsolation") << " MuIsolatorResultProducer CTOR";

  //! read input config for deposit types and weights and thresholds to apply to them
  std::vector<edm::ParameterSet> depositInputs = par.getParameter<std::vector<edm::ParameterSet>>("InputMuIsoDeposits");

  std::vector<double> dWeights(depositInputs.size());
  std::vector<double> dThresholds(depositInputs.size());

  for (unsigned int iDep = 0; iDep < depositInputs.size(); ++iDep) {
    DepositConf dConf;
    dConf.tag = depositInputs[iDep].getParameter<edm::InputTag>("DepositTag");
    dConf.weight = depositInputs[iDep].getParameter<double>("DepositWeight");
    dConf.threshold = depositInputs[iDep].getParameter<double>("DepositThreshold");

    dWeights[iDep] = dConf.weight;
    dThresholds[iDep] = dConf.threshold;

    theDepositConfs.push_back(dConf);
  }

  edm::ParameterSet isoPset = par.getParameter<edm::ParameterSet>("IsolatorPSet");
  //! will switch to a factory at some point
  std::string isolatorType = isoPset.getParameter<std::string>("ComponentName");
  if (isolatorType == "IsolatorByDeposit") {
    std::string coneSizeType = isoPset.getParameter<std::string>("ConeSizeType");
    if (coneSizeType == "FixedConeSize") {
      float coneSize(isoPset.getParameter<double>("coneSize"));

      theIsolator = new muonisolation::IsolatorByDeposit(coneSize, dWeights, dThresholds);

      //      theIsolator = new IsolatorByDeposit(isoPset);
    } else if (coneSizeType == "CutsConeSize") {
      //! FIXME
      //       Cuts cuts(isoPset.getParameter<edm::ParameterSet>("CutsPSet"));

      //       theIsolator = new IsolatorByDeposit(coneSize, dWeights, dThresholds);
    }
  } else if (isolatorType == "IsolatorByNominalEfficiency") {
    //! FIXME: need to get the file name here
    theIsolator =
        new muonisolation::IsolatorByNominalEfficiency("noname", std::vector<std::string>(1, "8:0.97"), dWeights);
  } else if (isolatorType == "IsolatorByDepositCount") {
    std::string coneSizeType = isoPset.getParameter<std::string>("ConeSizeType");
    if (coneSizeType == "FixedConeSize") {
      float coneSize(isoPset.getParameter<double>("coneSize"));

      theIsolator = new muonisolation::IsolatorByDepositCount(coneSize, dThresholds);

      //      theIsolator = new IsolatorByDeposit(isoPset);
    } else if (coneSizeType == "CutsConeSize") {
      //       Cuts cuts(isoPset.getParameter<edm::ParameterSet>("CutsPSet"));

      //       theIsolator = new IsolatorByDeposit(coneSize, dWeights, dThresholds);
    }
  }

  if (theIsolator == nullptr) {
    edm::LogError("MuonIsolationProducers") << "Failed to initialize an isolator";
  }
  theResultType = theIsolator->resultType();

  callWhatProduces();

  if (theRemoveOtherVetos) {
    edm::ParameterSet vetoPSet = par.getParameter<edm::ParameterSet>("VetoPSet");
    theVetoCuts.selectAll = vetoPSet.getParameter<bool>("SelectAll");

    //! "other vetoes" is limited to the same collection now
    //! for non-trivial choice an external map with pre-made selection flags
    //! can be a better choice
    if (!theVetoCuts.selectAll) {
      theVetoCuts.muAbsEtaMax = vetoPSet.getParameter<double>("MuAbsEtaMax");
      theVetoCuts.muPtMin = vetoPSet.getParameter<double>("MuPtMin");
      theVetoCuts.muAbsZMax = vetoPSet.getParameter<double>("MuAbsZMax");
      theVetoCuts.muD0Max = vetoPSet.getParameter<double>("MuD0Max");
      theBeamlineOption = par.getParameter<std::string>("BeamlineOption");
      theBeamSpotLabel = par.getParameter<edm::InputTag>("BeamSpotLabel");
    }
  }
}

//! destructor
template <typename BT>
MuIsolatorResultProducer<BT>::~MuIsolatorResultProducer() {
  if (theIsolator)
    delete theIsolator;
  LogDebug("RecoMuon|MuIsolatorResultProducer") << " MuIsolatorResultProducer DTOR";
}

template <typename BT>
void MuIsolatorResultProducer<BT>::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  std::string metname = "RecoMuon|MuonIsolationProducers";
  LogDebug(metname) << " Muon Deposit producing..."
                    << " BEGINING OF EVENT "
                    << "================================";

  reco::TrackBase::Point theBeam = reco::TrackBase::Point(0, 0, 0);

  //! do it only if needed
  if (theRemoveOtherVetos && !theVetoCuts.selectAll) {
    if (theBeamlineOption == "BeamSpotFromEvent") {
      //pick beamSpot
      reco::BeamSpot beamSpot;
      edm::Handle<reco::BeamSpot> beamSpotH;

      event.getByLabel(theBeamSpotLabel, beamSpotH);

      if (beamSpotH.isValid()) {
        theBeam = beamSpotH->position();
        LogTrace(metname) << "Extracted beam point at " << theBeam << std::endl;
      }
    }
  }

  //! "smart" container
  //! used to repackage deposits_type_candIndex into deposits_candIndex_type
  //! have to have it for veto removal (could do away without it otherwise)
  //! IMPORTANT: ALL THE REFERENCING BUSINESS IS DONE THROUGH POINTERS
  //! Access to the mapped values as reference type HAS TO BE AVAILABLE
  CandMap candMapT;

  unsigned int colSize = initAssociation(event, candMapT);

  //! isolator results will be here
  Results results(colSize);

  //! extra vetos will be filled here
  std::vector<reco::IsoDeposit::Vetos*> vetoDeps(theDepositConfs.size(), nullptr);

  if (colSize != 0) {
    if (theRemoveOtherVetos) {
      initVetos(theBeam, vetoDeps, candMapT);
    }

    //! call the isolator result, passing {[deposit,vetos]_type} set and the candidate
    for (unsigned int muI = 0; muI < colSize; ++muI) {
      results[muI] = theIsolator->result(candMapT.get()[muI].second, *(candMapT.get()[muI].first));

      if (results[muI].typeF() != theIsolator->resultType()) {
        edm::LogError("MuonIsolationProducers") << "Failed to get result from the isolator";
      }
    }
  }

  LogDebug(metname) << "Ready to write out results of size " << results.size();
  writeOut(event, candMapT, results);

  for (unsigned int iDep = 0; iDep < vetoDeps.size(); ++iDep) {
    //! do cleanup
    if (vetoDeps[iDep]) {
      delete vetoDeps[iDep];
      vetoDeps[iDep] = nullptr;
    }
  }
}

template <typename BT>
unsigned int MuIsolatorResultProducer<BT>::initAssociation(edm::Event& event, CandMap& candMapT) const {
  std::string metname = "RecoMuon|MuonIsolationProducers";

  typedef reco::IsoDepositMap::container CT;

  for (unsigned int iMap = 0; iMap < theDepositConfs.size(); ++iMap) {
    edm::Handle<reco::IsoDepositMap> depH;
    event.getByLabel(theDepositConfs[iMap].tag, depH);
    LogDebug(metname) << "Got Deposits of size " << depH->size();
    if (depH->empty())
      continue;

    //! WARNING: the input ValueMaps are better be for a single key product ID
    //! no effort is done (FIXME) for more complex cases
    typename CandMap::handle_type keyH;
    event.get(depH->begin().id(), keyH);
    candMapT.setHandle(keyH);
    typename CT::const_iterator depHCI = depH->begin().begin();
    typename CT::const_iterator depEnd = depH->begin().end();
    unsigned int keyI = 0;
    for (; depHCI != depEnd; ++depHCI, ++keyI) {
      typename CandMap::key_type muPtr(keyH->refAt(keyI));
      //! init {muon, {[deposit,veto]_type}} container
      if (iMap == 0)
        candMapT.get().push_back(typename CandMap::pair_type(muPtr, DepositContainer(theDepositConfs.size())));
      typename CandMap::iterator muI = candMapT.get().begin();
      for (; muI != candMapT.get().end(); ++muI) {
        if (muI->first == muPtr)
          break;
      }
      if (muI->first != muPtr) {
        edm::LogError("MuonIsolationProducers") << "Failed to align muon map";
      }
      muI->second[iMap].dep = &*depHCI;
    }
  }

  LogDebug(metname) << "Picked and aligned nDeps = " << candMapT.get().size();
  return candMapT.get().size();
}

template <typename BT>
void MuIsolatorResultProducer<BT>::initVetos(reco::TrackBase::Point const& theBeam,
                                             std::vector<reco::IsoDeposit::Vetos*>& vetos,
                                             CandMap& candMapT) const {
  std::string metname = "RecoMuon|MuonIsolationProducers";

  if (theRemoveOtherVetos) {
    LogDebug(metname) << "Start checking for vetos based on input/expected vetos.size of " << vetos.size()
                      << " passed at " << &vetos << " and an input map.size of " << candMapT.get().size();

    unsigned int muI = 0;
    for (; muI < candMapT.get().size(); ++muI) {
      typename CandMap::key_type mu = candMapT.get()[muI].first;
      double d0 = ((mu->vx() - theBeam.x()) * mu->py() - (mu->vy() - theBeam.y()) * mu->px()) / mu->pt();
      LogDebug(metname) << "Muon at index " << muI;
      if (theVetoCuts.selectAll || (fabs(mu->eta()) < theVetoCuts.muAbsEtaMax && mu->pt() > theVetoCuts.muPtMin &&
                                    fabs(mu->vz()) < theVetoCuts.muAbsZMax && fabs(d0) < theVetoCuts.muD0Max)) {
        LogDebug(metname) << "muon passes the cuts";
        for (unsigned int iDep = 0; iDep < candMapT.get()[muI].second.size(); ++iDep) {
          if (vetos[iDep] == nullptr)
            vetos[iDep] = new reco::IsoDeposit::Vetos();

          vetos[iDep]->push_back(candMapT.get()[muI].second[iDep].dep->veto());
        }
      }
    }

    LogDebug(metname) << "Assigning vetos";
    muI = 0;
    for (; muI < candMapT.get().size(); ++muI) {
      for (unsigned int iDep = 0; iDep < candMapT.get()[muI].second.size(); ++iDep) {
        candMapT.get()[muI].second[iDep].vetos = vetos[iDep];
      }
    }
    LogDebug(metname) << "Done with vetos";
  }
}

#endif
