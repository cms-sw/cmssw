// -*- C++ -*-
//
// Package:    L1Trigger/L1TTrackMatch
// Class:      L1TrackSelectionProducer
//
/**\class L1TrackSelectionProducer L1TrackSelectionProducer.cc L1Trigger/L1TTrackMatch/plugins/L1TrackSelectionProducer.cc

 Description: Selects a set of L1Tracks based on a set of predefined criteria.

 Implementation:
     Inputs:
         std::vector<TTTrack> - Each floating point TTTrack inside this collection inherits from
                                a bit-accurate TTTrack_TrackWord, used for emulation purposes.
     Outputs:
         std::vector<TTTrack> - A collection of TTTracks selected from cuts on the TTTrack properties
         std::vector<TTTrack> - A collection of TTTracks selected from cuts on the TTTrack_TrackWord properties
*/
//
// Original Author:  Alexx Perloff
//         Created:  Thu, 16 Dec 2021 19:02:50 GMT
//
//

// system include files
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// Xilinx HLS includes
#include <ap_fixed.h>
#include <ap_int.h>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CommonTools/Utils/interface/AndSelector.h"
#include "CommonTools/Utils/interface/EtaRangeSelector.h"
#include "CommonTools/Utils/interface/MinSelector.h"
#include "CommonTools/Utils/interface/MinFunctionSelector.h"
#include "CommonTools/Utils/interface/MinNumberSelector.h"
#include "CommonTools/Utils/interface/PtMinSelector.h"
#include "CommonTools/Utils/interface/Selection.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

//
// class declaration
//

class L1TrackSelectionProducer : public edm::global::EDProducer<> {
public:
  explicit L1TrackSelectionProducer(const edm::ParameterSet&);
  ~L1TrackSelectionProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------constants, enums and typedefs ---------
  // Relevant constants for the converted track word
  enum TrackBitWidths {
    kPtSize = TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1,  // Width of pt
    kPtMagSize = 9,                                              // Width of pt magnitude (unsigned)
    kEtaSize = TTTrack_TrackWord::TrackBitWidths::kTanlSize,     // Width of eta
    kEtaMagSize = 3,                                             // Width of eta magnitude (signed)
  };

  typedef TTTrack<Ref_Phase2TrackerDigi_> L1Track;
  typedef std::vector<L1Track> TTTrackCollection;
  typedef edm::Handle<TTTrackCollection> TTTrackCollectionHandle;
  typedef edm::Ref<TTTrackCollection> TTTrackRef;
  typedef edm::RefVector<TTTrackCollection> TTTrackRefCollection;
  typedef std::unique_ptr<TTTrackRefCollection> TTTrackRefCollectionUPtr;

  // ----------member functions ----------------------
  void printDebugInfo(const TTTrackCollectionHandle& l1TracksHandle,
                      const TTTrackRefCollectionUPtr& vTTTrackOutput,
                      const TTTrackRefCollectionUPtr& vTTTrackEmulationOutput,
                      const TTTrackRefCollectionUPtr& vTTTrackAssociatedOutput,
                      const TTTrackRefCollectionUPtr& vTTTrackAssociatedEmulationOutput) const;
  void printTrackInfo(edm::LogInfo& log, const L1Track& track, bool printEmulation = false) const;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------selectors -----------------------------
  // Based on recommendations from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenericSelectors
  struct TTTrackPtMinSelector {
    TTTrackPtMinSelector(double ptMin) : ptMin_(ptMin) {}
    TTTrackPtMinSelector(const edm::ParameterSet& cfg) : ptMin_(cfg.template getParameter<double>("ptMin")) {}
    bool operator()(const L1Track& t) const { return t.momentum().perp() >= ptMin_; }

  private:
    double ptMin_;
  };
  struct TTTrackWordPtMinSelector {
    TTTrackWordPtMinSelector(double ptMin) : ptMin_(ptMin) {}
    TTTrackWordPtMinSelector(const edm::ParameterSet& cfg) : ptMin_(cfg.template getParameter<double>("ptMin")) {}
    bool operator()(const L1Track& t) const {
      ap_uint<TrackBitWidths::kPtSize> ptEmulationBits = t.getTrackWord()(
          TTTrack_TrackWord::TrackBitLocations::kRinvMSB - 1, TTTrack_TrackWord::TrackBitLocations::kRinvLSB);
      ap_ufixed<TrackBitWidths::kPtSize, TrackBitWidths::kPtMagSize> ptEmulation;
      ptEmulation.V = ptEmulationBits.range();
      //edm::LogInfo("L1TrackSelectionProducer") << "produce::Emulation track properties::ap_uint(bits) = " << ptEmulationBits.to_string(2)
      //                                         << " ap_ufixed(bits) = " << ptEmulation.to_string(2) << " ap_ufixed(float) = " << ptEmulation.to_double();
      return ptEmulation.to_double() >= ptMin_;
    }

  private:
    double ptMin_;
  };
  struct TTTrackAbsEtaMaxSelector {
    TTTrackAbsEtaMaxSelector(double absEtaMax) : absEtaMax_(absEtaMax) {}
    TTTrackAbsEtaMaxSelector(const edm::ParameterSet& cfg)
        : absEtaMax_(cfg.template getParameter<double>("absEtaMax")) {}
    bool operator()(const L1Track& t) const { return std::abs(t.momentum().eta()) <= absEtaMax_; }

  private:
    double absEtaMax_;
  };
  struct TTTrackWordAbsEtaMaxSelector {
    TTTrackWordAbsEtaMaxSelector(double absEtaMax) : absEtaMax_(absEtaMax) {}
    TTTrackWordAbsEtaMaxSelector(const edm::ParameterSet& cfg)
        : absEtaMax_(cfg.template getParameter<double>("absEtaMax")) {}
    bool operator()(const L1Track& t) const {
      TTTrack_TrackWord::tanl_t etaEmulationBits = t.getTanlWord();
      ap_fixed<TrackBitWidths::kEtaSize, TrackBitWidths::kEtaMagSize> etaEmulation;
      etaEmulation.V = etaEmulationBits.range();
      return std::abs(etaEmulation.to_double()) <= absEtaMax_;
    }

  private:
    double absEtaMax_;
  };
  struct TTTrackAbsZ0MaxSelector {
    TTTrackAbsZ0MaxSelector(double absZ0Max) : absZ0Max_(absZ0Max) {}
    TTTrackAbsZ0MaxSelector(const edm::ParameterSet& cfg) : absZ0Max_(cfg.template getParameter<double>("absZ0Max")) {}
    bool operator()(const L1Track& t) const { return std::abs(t.z0()) <= absZ0Max_; }

  private:
    double absZ0Max_;
  };
  struct TTTrackWordAbsZ0MaxSelector {
    TTTrackWordAbsZ0MaxSelector(double absZ0Max) : absZ0Max_(absZ0Max) {}
    TTTrackWordAbsZ0MaxSelector(const edm::ParameterSet& cfg)
        : absZ0Max_(cfg.template getParameter<double>("absZ0Max")) {}
    bool operator()(const L1Track& t) const { return std::abs(t.getZ0()) <= absZ0Max_; }

  private:
    double absZ0Max_;
  };
  struct TTTrackNStubsMinSelector {
    TTTrackNStubsMinSelector(double nStubsMin) : nStubsMin_(nStubsMin) {}
    TTTrackNStubsMinSelector(const edm::ParameterSet& cfg)
        : nStubsMin_(cfg.template getParameter<double>("nStubsMin")) {}
    bool operator()(const L1Track& t) const { return t.getStubRefs().size() >= nStubsMin_; }

  private:
    double nStubsMin_;
  };
  struct TTTrackWordNStubsMinSelector {
    TTTrackWordNStubsMinSelector(double nStubsMin) : nStubsMin_(nStubsMin) {}
    TTTrackWordNStubsMinSelector(const edm::ParameterSet& cfg)
        : nStubsMin_(cfg.template getParameter<double>("nStubsMin")) {}
    bool operator()(const L1Track& t) const { return t.getNStubs() >= nStubsMin_; }

  private:
    double nStubsMin_;
  };
  struct TTTrackNPSStubsMinSelector {
    TTTrackNPSStubsMinSelector(double nStubsMin, const TrackerTopology& tTopo)
        : nPSStubsMin_(nStubsMin), tTopo_(tTopo) {}
    TTTrackNPSStubsMinSelector(const edm::ParameterSet& cfg, const TrackerTopology& tTopo)
        : nPSStubsMin_(cfg.template getParameter<double>("nPSStubsMin")), tTopo_(tTopo) {}
    bool operator()(const L1Track& t) const {
      int nPSStubs = 0;
      for (const auto& stub : t.getStubRefs()) {
        DetId detId(stub->getDetId());
        if (detId.det() == DetId::Detector::Tracker) {
          if ((detId.subdetId() == StripSubdetector::TOB && tTopo_.tobLayer(detId) <= 3) ||
              (detId.subdetId() == StripSubdetector::TID && tTopo_.tidRing(detId) <= 9))
            nPSStubs++;
        }
      }
      return nPSStubs >= nPSStubsMin_;
    }

  private:
    double nPSStubsMin_;
    const TrackerTopology& tTopo_;
  };
  struct TTTrackBendChi2MaxSelector {
    TTTrackBendChi2MaxSelector(double bendChi2Max) : bendChi2Max_(bendChi2Max) {}
    TTTrackBendChi2MaxSelector(const edm::ParameterSet& cfg)
        : bendChi2Max_(cfg.template getParameter<double>("reducedBendChi2Max")) {}
    bool operator()(const L1Track& t) const { return t.stubPtConsistency() < bendChi2Max_; }

  private:
    double bendChi2Max_;
  };
  struct TTTrackWordBendChi2MaxSelector {
    TTTrackWordBendChi2MaxSelector(double bendChi2Max) : bendChi2Max_(bendChi2Max) {}
    TTTrackWordBendChi2MaxSelector(const edm::ParameterSet& cfg)
        : bendChi2Max_(cfg.template getParameter<double>("reducedBendChi2Max")) {}
    bool operator()(const L1Track& t) const { return t.getBendChi2() < bendChi2Max_; }

  private:
    double bendChi2Max_;
  };
  struct TTTrackChi2RZMaxSelector {
    TTTrackChi2RZMaxSelector(double reducedChi2RZMax) : reducedChi2RZMax_(reducedChi2RZMax) {}
    TTTrackChi2RZMaxSelector(const edm::ParameterSet& cfg)
        : reducedChi2RZMax_(cfg.template getParameter<double>("reducedChi2RZMax")) {}
    bool operator()(const L1Track& t) const { return t.chi2ZRed() < reducedChi2RZMax_; }

  private:
    double reducedChi2RZMax_;
  };
  struct TTTrackWordChi2RZMaxSelector {
    TTTrackWordChi2RZMaxSelector(double reducedChi2RZMax) : reducedChi2RZMax_(reducedChi2RZMax) {}
    TTTrackWordChi2RZMaxSelector(const edm::ParameterSet& cfg)
        : reducedChi2RZMax_(cfg.template getParameter<double>("reducedChi2RZMax")) {}
    bool operator()(const L1Track& t) const { return t.getChi2RZ() < reducedChi2RZMax_; }

  private:
    double reducedChi2RZMax_;
  };
  struct TTTrackChi2RPhiMaxSelector {
    TTTrackChi2RPhiMaxSelector(double reducedChi2RPhiMax) : reducedChi2RPhiMax_(reducedChi2RPhiMax) {}
    TTTrackChi2RPhiMaxSelector(const edm::ParameterSet& cfg)
        : reducedChi2RPhiMax_(cfg.template getParameter<double>("reducedChi2RPhiMax")) {}
    bool operator()(const L1Track& t) const { return t.chi2XYRed() < reducedChi2RPhiMax_; }

  private:
    double reducedChi2RPhiMax_;
  };
  struct TTTrackWordChi2RPhiMaxSelector {
    TTTrackWordChi2RPhiMaxSelector(double reducedChi2RPhiMax) : reducedChi2RPhiMax_(reducedChi2RPhiMax) {}
    TTTrackWordChi2RPhiMaxSelector(const edm::ParameterSet& cfg)
        : reducedChi2RPhiMax_(cfg.template getParameter<double>("reducedChi2RPhiMax")) {}
    bool operator()(const L1Track& t) const { return t.getChi2RPhi() < reducedChi2RPhiMax_; }

  private:
    double reducedChi2RPhiMax_;
  };
  struct TTTrackDeltaZMaxSelector {
    TTTrackDeltaZMaxSelector(const std::vector<double>& deltaZMaxEtaBounds, const std::vector<double>& deltaZMax)
        : deltaZMaxEtaBounds_(deltaZMaxEtaBounds), deltaZMax_(deltaZMax) {}
    TTTrackDeltaZMaxSelector(const edm::ParameterSet& cfg)
        : deltaZMaxEtaBounds_(cfg.template getParameter<double>("deltaZMaxEtaBounds")),
          deltaZMax_(cfg.template getParameter<double>("deltaZMax")) {}
    bool operator()(const L1Track& t, const l1t::Vertex& v) const {
      size_t etaIndex =
          std::lower_bound(deltaZMaxEtaBounds_.begin(), deltaZMaxEtaBounds_.end(), std::abs(t.momentum().eta())) -
          deltaZMaxEtaBounds_.begin() - 1;
      if (etaIndex > deltaZMax_.size() - 1)
        etaIndex = deltaZMax_.size() - 1;
      return std::abs(v.z0() - t.z0()) <= deltaZMax_[etaIndex];
    }

  private:
    std::vector<double> deltaZMaxEtaBounds_;
    std::vector<double> deltaZMax_;
  };
  struct TTTrackWordDeltaZMaxSelector {
    TTTrackWordDeltaZMaxSelector(const std::vector<double>& deltaZMaxEtaBounds, const std::vector<double>& deltaZMax)
        : deltaZMaxEtaBounds_(deltaZMaxEtaBounds), deltaZMax_(deltaZMax) {}
    TTTrackWordDeltaZMaxSelector(const edm::ParameterSet& cfg)
        : deltaZMaxEtaBounds_(cfg.template getParameter<double>("deltaZMaxEtaBounds")),
          deltaZMax_(cfg.template getParameter<double>("deltaZMax")) {}
    bool operator()(const L1Track& t, const l1t::VertexWord& v) const {
      size_t etaIndex =
          std::lower_bound(deltaZMaxEtaBounds_.begin(), deltaZMaxEtaBounds_.end(), std::abs(t.momentum().eta())) -
          deltaZMaxEtaBounds_.begin() - 1;
      if (etaIndex > deltaZMax_.size() - 1)
        etaIndex = deltaZMax_.size() - 1;
      return std::abs(v.z0() - t.getZ0()) <= deltaZMax_[etaIndex];
    }

  private:
    std::vector<double> deltaZMaxEtaBounds_;
    std::vector<double> deltaZMax_;
  };

  typedef AndSelector<TTTrackPtMinSelector, TTTrackAbsEtaMaxSelector, TTTrackAbsZ0MaxSelector, TTTrackNStubsMinSelector>
      TTTrackPtMinEtaMaxZ0MaxNStubsMinSelector;
  typedef AndSelector<TTTrackWordPtMinSelector,
                      TTTrackWordAbsEtaMaxSelector,
                      TTTrackWordAbsZ0MaxSelector,
                      TTTrackWordNStubsMinSelector>
      TTTrackWordPtMinEtaMaxZ0MaxNStubsMinSelector;
  typedef AndSelector<TTTrackBendChi2MaxSelector, TTTrackChi2RZMaxSelector, TTTrackChi2RPhiMaxSelector>
      TTTrackBendChi2Chi2RZChi2RPhiMaxSelector;
  typedef AndSelector<TTTrackWordBendChi2MaxSelector, TTTrackWordChi2RZMaxSelector, TTTrackWordChi2RPhiMaxSelector>
      TTTrackWordBendChi2Chi2RZChi2RPhiMaxSelector;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<TTTrackCollection> l1TracksToken_;
  edm::EDGetTokenT<l1t::VertexCollection> l1VerticesToken_;
  edm::EDGetTokenT<l1t::VertexWordCollection> l1VerticesEmulationToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const std::string outputCollectionName_;
  const edm::ParameterSet cutSet_;
  const double ptMin_, absEtaMax_, absZ0Max_, bendChi2Max_, reducedChi2RZMax_, reducedChi2RPhiMax_;
  const int nStubsMin_, nPSStubsMin_;
  std::vector<double> deltaZMaxEtaBounds_, deltaZMax_;
  const double useDisplacedTracksDeltaZOverride_;
  bool processSimulatedTracks_, processEmulatedTracks_, doDeltaZCutSim_, doDeltaZCutEmu_;
  int debug_;
};

//
// constructors and destructor
//
L1TrackSelectionProducer::L1TrackSelectionProducer(const edm::ParameterSet& iConfig)
    : l1TracksToken_(consumes<TTTrackCollection>(iConfig.getParameter<edm::InputTag>("l1TracksInputTag"))),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""))),
      outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")),
      cutSet_(iConfig.getParameter<edm::ParameterSet>("cutSet")),

      ptMin_(cutSet_.getParameter<double>("ptMin")),
      absEtaMax_(cutSet_.getParameter<double>("absEtaMax")),
      absZ0Max_(cutSet_.getParameter<double>("absZ0Max")),
      bendChi2Max_(cutSet_.getParameter<double>("reducedBendChi2Max")),
      reducedChi2RZMax_(cutSet_.getParameter<double>("reducedChi2RZMax")),
      reducedChi2RPhiMax_(cutSet_.getParameter<double>("reducedChi2RPhiMax")),
      nStubsMin_(cutSet_.getParameter<int>("nStubsMin")),
      nPSStubsMin_(cutSet_.getParameter<int>("nPSStubsMin")),
      deltaZMaxEtaBounds_(cutSet_.getParameter<std::vector<double>>("deltaZMaxEtaBounds")),
      deltaZMax_(cutSet_.getParameter<std::vector<double>>("deltaZMax")),

      useDisplacedTracksDeltaZOverride_(iConfig.getParameter<double>("useDisplacedTracksDeltaZOverride")),
      processSimulatedTracks_(iConfig.getParameter<bool>("processSimulatedTracks")),
      processEmulatedTracks_(iConfig.getParameter<bool>("processEmulatedTracks")),
      debug_(iConfig.getParameter<int>("debug")) {
  // Confirm the the configuration makes sense
  if (!processSimulatedTracks_ && !processEmulatedTracks_) {
    throw cms::Exception("You must process at least one of the track collections (simulated or emulated).");
  }

  if (deltaZMax_.size() != deltaZMaxEtaBounds_.size() - 1) {
    throw cms::Exception("The number of deltaZ cuts does not match the number of eta bins!");
  }

  if (useDisplacedTracksDeltaZOverride_ >= 0) {
    deltaZMax_ = std::vector<double>(deltaZMax_.size(), useDisplacedTracksDeltaZOverride_);
  }

  // Get additional input tags and define the EDM output based on the previous configuration parameters
  doDeltaZCutSim_ = false;
  doDeltaZCutEmu_ = false;
  if (processSimulatedTracks_) {
    produces<TTTrackRefCollection>(outputCollectionName_);
    if (iConfig.exists("l1VerticesInputTag")) {
      l1VerticesToken_ = consumes<l1t::VertexCollection>(iConfig.getParameter<edm::InputTag>("l1VerticesInputTag"));
      doDeltaZCutSim_ = true;
      produces<TTTrackRefCollection>(outputCollectionName_ + "Associated");
    }
  }
  if (processEmulatedTracks_) {
    produces<TTTrackRefCollection>(outputCollectionName_ + "Emulation");
    if (iConfig.exists("l1VerticesEmulationInputTag")) {
      l1VerticesEmulationToken_ =
          consumes<l1t::VertexWordCollection>(iConfig.getParameter<edm::InputTag>("l1VerticesEmulationInputTag"));
      doDeltaZCutEmu_ = true;
      produces<TTTrackRefCollection>(outputCollectionName_ + "AssociatedEmulation");
    }
  }
}

L1TrackSelectionProducer::~L1TrackSelectionProducer() {}

//
// member functions
//

void L1TrackSelectionProducer::printDebugInfo(const TTTrackCollectionHandle& l1TracksHandle,
                                              const TTTrackRefCollectionUPtr& vTTTrackOutput,
                                              const TTTrackRefCollectionUPtr& vTTTrackEmulationOutput,
                                              const TTTrackRefCollectionUPtr& vTTTrackAssociatedOutput,
                                              const TTTrackRefCollectionUPtr& vTTTrackAssociatedEmulationOutput) const {
  edm::LogInfo log("L1TrackSelectionProducer");
  log << "The original track collection (pt, eta, phi, nstub, bendchi2, chi2rz, chi2rphi, z0) values are ... \n";
  for (const auto& track : *l1TracksHandle) {
    printTrackInfo(log, track, debug_ >= 4);
  }
  log << "\t---\n\tNumber of tracks in this selection = " << l1TracksHandle->size() << "\n\n";
  if (processSimulatedTracks_) {
    log << "The selected track collection (pt, eta, phi, nstub, bendchi2, chi2rz, chi2rphi, z0) values are ... \n";
    for (const auto& track : *vTTTrackOutput) {
      printTrackInfo(log, *track, debug_ >= 4);
    }
    log << "\t---\n\tNumber of tracks in this selection = " << vTTTrackOutput->size() << "\n\n";
  }
  if (processEmulatedTracks_) {
    log << "The emulation selected track collection (pt, eta, phi, nstub, bendchi2, chi2rz, chi2rphi, z0) values are "
           "... \n";
    for (const auto& track : *vTTTrackEmulationOutput) {
      printTrackInfo(log, *track, debug_ >= 4);
    }
    log << "\t---\n\tNumber of tracks in this selection = " << vTTTrackEmulationOutput->size() << "\n\n";
  }
  if (processSimulatedTracks_ && processEmulatedTracks_) {
    TTTrackRefCollection inSimButNotEmu;
    TTTrackRefCollection inEmuButNotSim;
    std::set_difference(vTTTrackOutput->begin(),
                        vTTTrackOutput->end(),
                        vTTTrackEmulationOutput->begin(),
                        vTTTrackEmulationOutput->end(),
                        std::back_inserter(inSimButNotEmu));
    std::set_difference(vTTTrackEmulationOutput->begin(),
                        vTTTrackEmulationOutput->end(),
                        vTTTrackOutput->begin(),
                        vTTTrackOutput->end(),
                        std::back_inserter(inEmuButNotSim));
    log << "The set of tracks selected via cuts on the simulated values which are not in the set of tracks selected "
           "by cutting on the emulated values ... \n";
    for (const auto& track : inSimButNotEmu) {
      printTrackInfo(log, *track, debug_ >= 3);
    }
    log << "\t---\n\tNumber of tracks in this selection = " << inSimButNotEmu.size() << "\n\n"
        << "The set of tracks selected via cuts on the emulated values which are not in the set of tracks selected "
           "by cutting on the simulated values ... \n";
    for (const auto& track : inEmuButNotSim) {
      printTrackInfo(log, *track, debug_ >= 3);
    }
    log << "\t---\n\tNumber of tracks in this selection = " << inEmuButNotSim.size() << "\n\n";
  }
  if (processSimulatedTracks_) {
    log << "The selected and leading vertex associated track collection (pt, eta, phi, nstub, bendchi2, chi2rz, "
           "chi2rphi, z0) values are ... \n";
    for (const auto& track : *vTTTrackAssociatedOutput) {
      printTrackInfo(log, *track, debug_ >= 4);
    }
    log << "\t---\n\tNumber of tracks in this selection = " << vTTTrackAssociatedOutput->size() << "\n\n";
  }
  if (processEmulatedTracks_) {
    log << "The emulation selected and leading vertex associated track collection (pt, eta, phi, nstub, bendchi2, "
           "chi2rz, chi2rphi, z0) values are "
           "... \n";
    for (const auto& track : *vTTTrackAssociatedEmulationOutput) {
      printTrackInfo(log, *track, debug_ >= 4);
    }
    log << "\t---\n\tNumber of tracks in this selection = " << vTTTrackAssociatedEmulationOutput->size() << "\n\n";
  }
  if (processSimulatedTracks_ && processEmulatedTracks_) {
    TTTrackRefCollection inSimButNotEmu;
    TTTrackRefCollection inEmuButNotSim;
    std::set_difference(vTTTrackAssociatedOutput->begin(),
                        vTTTrackAssociatedOutput->end(),
                        vTTTrackAssociatedEmulationOutput->begin(),
                        vTTTrackAssociatedEmulationOutput->end(),
                        std::back_inserter(inSimButNotEmu));
    std::set_difference(vTTTrackAssociatedEmulationOutput->begin(),
                        vTTTrackAssociatedEmulationOutput->end(),
                        vTTTrackAssociatedOutput->begin(),
                        vTTTrackAssociatedOutput->end(),
                        std::back_inserter(inEmuButNotSim));
    log << "The set of tracks selected via cuts on the simulated values which are not in the set of tracks selected "
           "by cutting on the emulated values ... \n";
    for (const auto& track : inSimButNotEmu) {
      printTrackInfo(log, *track, debug_ >= 3);
    }
    log << "\t---\n\tNumber of tracks in this selection = " << inSimButNotEmu.size() << "\n\n"
        << "The set of tracks selected via cuts on the emulated values which are not in the set of tracks selected "
           "by cutting on the simulated values ... \n";
    for (const auto& track : inEmuButNotSim) {
      printTrackInfo(log, *track, debug_ >= 3);
    }
    log << "\t---\n\tNumber of tracks in this selection = " << inEmuButNotSim.size() << "\n\n";
  }
}

void L1TrackSelectionProducer::printTrackInfo(edm::LogInfo& log, const L1Track& track, bool printEmulation) const {
  log << "\t(" << track.momentum().perp() << ", " << track.momentum().eta() << ", " << track.momentum().phi() << ", "
      << track.getStubRefs().size() << ", " << track.stubPtConsistency() << ", " << track.chi2ZRed() << ", "
      << track.chi2XYRed() << ", " << track.z0() << ")\n";

  if (printEmulation) {
    ap_uint<TrackBitWidths::kPtSize> ptEmulationBits = track.getTrackWord()(
        TTTrack_TrackWord::TrackBitLocations::kRinvMSB - 1, TTTrack_TrackWord::TrackBitLocations::kRinvLSB);
    ap_ufixed<TrackBitWidths::kPtSize, TrackBitWidths::kPtMagSize> ptEmulation;
    ptEmulation.V = ptEmulationBits.range();
    TTTrack_TrackWord::tanl_t etaEmulationBits = track.getTanlWord();
    ap_fixed<TrackBitWidths::kEtaSize, TrackBitWidths::kEtaMagSize> etaEmulation;
    etaEmulation.V = etaEmulationBits.range();
    log << "\t\t(" << ptEmulation.to_double() << ", " << etaEmulation.to_double() << ", " << track.getPhi() << ", "
        << track.getNStubs() << ", " << track.getBendChi2() << ", " << track.getChi2RZ() << ", " << track.getChi2RPhi()
        << ", " << track.getZ0() << ")\n";
  }
}

// ------------ method called to produce the data  ------------
void L1TrackSelectionProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto vTTTrackOutput = std::make_unique<TTTrackRefCollection>();
  auto vTTTrackAssociatedOutput = std::make_unique<TTTrackRefCollection>();
  auto vTTTrackEmulationOutput = std::make_unique<TTTrackRefCollection>();
  auto vTTTrackAssociatedEmulationOutput = std::make_unique<TTTrackRefCollection>();

  // Tracker Topology
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);

  TTTrackCollectionHandle l1TracksHandle;
  edm::Handle<l1t::VertexCollection> l1VerticesHandle;
  edm::Handle<l1t::VertexWordCollection> l1VerticesEmulationHandle;

  l1t::Vertex leadingVertex;
  l1t::VertexWord leadingEmulationVertex;

  iEvent.getByToken(l1TracksToken_, l1TracksHandle);
  size_t nOutputApproximate = l1TracksHandle->size();
  if (processSimulatedTracks_) {
    if (doDeltaZCutSim_) {
      iEvent.getByToken(l1VerticesToken_, l1VerticesHandle);
      leadingVertex = l1VerticesHandle->at(0);
      if (debug_ >= 2) {
        edm::LogInfo("L1TrackSelectionProducer") << "leading vertex z0 = " << leadingVertex.z0();
      }
    }
    vTTTrackOutput->reserve(nOutputApproximate);
    vTTTrackAssociatedOutput->reserve(nOutputApproximate);
  }
  if (processEmulatedTracks_) {
    if (doDeltaZCutEmu_) {
      iEvent.getByToken(l1VerticesEmulationToken_, l1VerticesEmulationHandle);
      leadingEmulationVertex = l1VerticesEmulationHandle->at(0);
      if (debug_ >= 2) {
        edm::LogInfo("L1TrackSelectionProducer") << "leading emulation vertex z0 = " << leadingEmulationVertex.z0();
      }
    }
    vTTTrackEmulationOutput->reserve(nOutputApproximate);
    vTTTrackAssociatedEmulationOutput->reserve(nOutputApproximate);
  }

  TTTrackPtMinEtaMaxZ0MaxNStubsMinSelector kinSel(ptMin_, absEtaMax_, absZ0Max_, nStubsMin_);
  TTTrackWordPtMinEtaMaxZ0MaxNStubsMinSelector kinSelEmu(ptMin_, absEtaMax_, absZ0Max_, nStubsMin_);
  TTTrackBendChi2Chi2RZChi2RPhiMaxSelector chi2Sel(bendChi2Max_, reducedChi2RZMax_, reducedChi2RPhiMax_);
  TTTrackWordBendChi2Chi2RZChi2RPhiMaxSelector chi2SelEmu(bendChi2Max_, reducedChi2RZMax_, reducedChi2RPhiMax_);
  TTTrackDeltaZMaxSelector deltaZSel(deltaZMaxEtaBounds_, deltaZMax_);
  TTTrackWordDeltaZMaxSelector deltaZSelEmu(deltaZMaxEtaBounds_, deltaZMax_);
  TTTrackNPSStubsMinSelector nPSStubsSel(nPSStubsMin_, tTopo);

  for (size_t i = 0; i < nOutputApproximate; i++) {
    const auto& track = l1TracksHandle->at(i);

    // Select tracks based on the floating point TTTrack
    if (processSimulatedTracks_ && kinSel(track) && nPSStubsSel(track) && chi2Sel(track)) {
      vTTTrackOutput->push_back(TTTrackRef(l1TracksHandle, i));
      if (doDeltaZCutSim_ && deltaZSel(track, leadingVertex)) {
        vTTTrackAssociatedOutput->push_back(TTTrackRef(l1TracksHandle, i));
      }
    }

    // Select tracks based on the bitwise accurate TTTrack_TrackWord
    if (processEmulatedTracks_ && kinSelEmu(track) && chi2SelEmu(track)) {
      vTTTrackEmulationOutput->push_back(TTTrackRef(l1TracksHandle, i));
      if (doDeltaZCutEmu_ && deltaZSelEmu(track, leadingEmulationVertex)) {
        vTTTrackAssociatedEmulationOutput->push_back(TTTrackRef(l1TracksHandle, i));
      }
    }
  }

  if (debug_ >= 2) {
    printDebugInfo(l1TracksHandle,
                   vTTTrackOutput,
                   vTTTrackEmulationOutput,
                   vTTTrackAssociatedOutput,
                   vTTTrackAssociatedEmulationOutput);
  }

  // Put the outputs into the event
  if (processSimulatedTracks_) {
    iEvent.put(std::move(vTTTrackOutput), outputCollectionName_);
    if (doDeltaZCutSim_) {
      iEvent.put(std::move(vTTTrackAssociatedOutput), outputCollectionName_ + "Associated");
    }
  }
  if (processEmulatedTracks_) {
    iEvent.put(std::move(vTTTrackEmulationOutput), outputCollectionName_ + "Emulation");
    if (doDeltaZCutEmu_) {
      iEvent.put(std::move(vTTTrackAssociatedEmulationOutput), outputCollectionName_ + "AssociatedEmulation");
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TrackSelectionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //L1TrackSelectionProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1TracksInputTag", edm::InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.addOptional<edm::InputTag>("l1VerticesInputTag", edm::InputTag("l1tVertexFinder", "l1vertices"));
  desc.addOptional<edm::InputTag>("l1VerticesEmulationInputTag",
                                  edm::InputTag("l1tVertexFinderEmulator", "l1verticesEmulation"));
  desc.add<std::string>("outputCollectionName", "Level1TTTracksSelected");
  {
    edm::ParameterSetDescription descCutSet;
    descCutSet.add<double>("ptMin", 2.0)->setComment("pt must be greater than this value, [GeV]");
    descCutSet.add<double>("absEtaMax", 2.4)->setComment("absolute value of eta must be less than this value");
    descCutSet.add<double>("absZ0Max", 15.0)->setComment("z0 must be less than this value, [cm]");
    descCutSet.add<int>("nStubsMin", 4)->setComment("number of stubs must be greater than or equal to this value");
    descCutSet.add<int>("nPSStubsMin", 0)
        ->setComment("number of stubs in the PS Modules must be greater than or equal to this value");

    descCutSet.add<double>("reducedBendChi2Max", 2.25)->setComment("bend chi2 must be less than this value");
    descCutSet.add<double>("reducedChi2RZMax", 5.0)->setComment("chi2rz/dof must be less than this value");
    descCutSet.add<double>("reducedChi2RPhiMax", 20.0)->setComment("chi2rphi/dof must be less than this value");

    descCutSet.add<std::vector<double>>("deltaZMaxEtaBounds", {0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4})
        ->setComment("these values define the bin boundaries in |eta|");
    descCutSet.add<std::vector<double>>("deltaZMax", {0.37, 0.50, 0.60, 0.75, 1.00, 1.60})
        ->setComment(
            "delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, "
            "[cm]");
    desc.add<edm::ParameterSetDescription>("cutSet", descCutSet);
  }
  desc.add<double>("useDisplacedTracksDeltaZOverride", -1.0)
      ->setComment("override the deltaZ cut value for displaced tracks");
  desc.add<bool>("processSimulatedTracks", true)
      ->setComment("return selected tracks after cutting on the floating point values");
  desc.add<bool>("processEmulatedTracks", true)
      ->setComment("return selected tracks after cutting on the bitwise emulated values");
  desc.add<int>("debug", 0)->setComment("Verbosity levels: 0, 1, 2, 3");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackSelectionProducer);
