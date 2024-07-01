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
// Updates: Claire Savard (claire.savard@colorado.edu), Nov. 2023
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
                      const TTTrackRefCollectionUPtr& vTTTrackEmulationOutput) const;
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
    bool operator()(const L1Track& t) const {
      double floatZ0 = t.undigitizeSignedValue(
          t.getZ0Bits(), TTTrack_TrackWord::TrackBitWidths::kZ0Size, TTTrack_TrackWord::stepZ0, 0.0);
      return std::abs(floatZ0) <= absZ0Max_;
    }

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
  struct TTTrackNStubsMinEtaOverlapSelector {
    TTTrackNStubsMinEtaOverlapSelector(double nStubsMinEtaOverlap) : nStubsMinEtaOverlap_(nStubsMinEtaOverlap) {}
    TTTrackNStubsMinEtaOverlapSelector(const edm::ParameterSet& cfg)
        : nStubsMinEtaOverlap_(cfg.template getParameter<double>("nStubsMinEtaOverlap")) {}
    bool operator()(const L1Track& t) const {
      if((std::abs(t.momentum().eta())<1.7) && (std::abs(t.momentum().eta())>1.1)){
	return (t.getStubRefs().size() >= nStubsMinEtaOverlap_);
      }
      else{
	return true;
      }
    }

  private:
    double nStubsMinEtaOverlap_;
  };
  struct TTTrackWordNStubsMinEtaOverlapSelector {
    TTTrackWordNStubsMinEtaOverlapSelector(double nStubsMinEtaOverlap) : nStubsMinEtaOverlap_(nStubsMinEtaOverlap) {}
    TTTrackWordNStubsMinEtaOverlapSelector(const edm::ParameterSet& cfg)
        : nStubsMinEtaOverlap_(cfg.template getParameter<double>("nStubsMinEtaOverlap")) {}
    bool operator()(const L1Track& t) const {
      TTTrack_TrackWord::tanl_t etaEmulationBits = t.getTanlWord();
      ap_fixed<TrackBitWidths::kEtaSize, TrackBitWidths::kEtaMagSize> etaEmulation;
      etaEmulation.V = etaEmulationBits.range();
      if((std::abs(etaEmulation.to_double())<1.7) && (std::abs(etaEmulation.to_double())>1.1)){
	return (t.getNStubs() >= nStubsMinEtaOverlap_);
      }
      else{
	return true;
      }
    }

  private:
    double nStubsMinEtaOverlap_;
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
  struct TTTrackPromptMVAMinSelector {
    TTTrackPromptMVAMinSelector(double promptMVAMin) : promptMVAMin_(promptMVAMin) {}
    TTTrackPromptMVAMinSelector(const edm::ParameterSet& cfg)
        : promptMVAMin_(cfg.template getParameter<double>("promptMVAMin")) {}
    bool operator()(const L1Track& t) const { return t.trkMVA1() >= promptMVAMin_; }

  private:
    double promptMVAMin_;
  };
  struct TTTrackWordPromptMVAMinSelector {
    TTTrackWordPromptMVAMinSelector(double promptMVAMin) : promptMVAMin_(promptMVAMin) {}
    TTTrackWordPromptMVAMinSelector(const edm::ParameterSet& cfg)
        : promptMVAMin_(cfg.template getParameter<double>("promptMVAMin")) {}
    bool operator()(const L1Track& t) const { return t.trkMVA1() >= promptMVAMin_; }  //change when mva bins in word are set

  private:
    double promptMVAMin_;
  };
  struct TTTrackPromptMVAMinD0Min1Selector {
    TTTrackPromptMVAMinD0Min1Selector(double promptMVAMinD0Min1) : promptMVAMinD0Min1_(promptMVAMinD0Min1) {}
    TTTrackPromptMVAMinD0Min1Selector(const edm::ParameterSet& cfg)
        : promptMVAMinD0Min1_(cfg.template getParameter<double>("promptMVAMinD0Min1")) {}
    bool operator()(const L1Track& t) const {
      if(std::abs(t.d0())>1.0){
	return (t.trkMVA1() > promptMVAMinD0Min1_);
      }
      else{
	return true;
      }
    }

  private:
    double promptMVAMinD0Min1_;
  };
  struct TTTrackWordPromptMVAMinD0Min1Selector {
    TTTrackWordPromptMVAMinD0Min1Selector(double promptMVAMinD0Min1) : promptMVAMinD0Min1_(promptMVAMinD0Min1) {}
    TTTrackWordPromptMVAMinD0Min1Selector(const edm::ParameterSet& cfg)
        : promptMVAMinD0Min1_(cfg.template getParameter<double>("promptMVAMinD0Min1")) {}
    bool operator()(const L1Track& t) const {
      double floatD0 = t.undigitizeSignedValue(t.getD0Bits(), TTTrack_TrackWord::TrackBitWidths::kD0Size, TTTrack_TrackWord::stepD0, 0.0);
      if(std::abs(floatD0)>1.0){
	return t.trkMVA1() >= promptMVAMinD0Min1_;
      }
      else{
	return true;
      }
    }  //change when mva bins in word are set

  private:
    double promptMVAMinD0Min1_;
  };
  struct TTTrackDisplacedMVAMinSelector {
    TTTrackDisplacedMVAMinSelector(double displacedMVAMin) : displacedMVAMin_(displacedMVAMin) {}
    TTTrackDisplacedMVAMinSelector(const edm::ParameterSet& cfg)
        : displacedMVAMin_(cfg.template getParameter<double>("displacedMVAMin")) {}
    bool operator()(const L1Track& t) const { return t.trkMVA2() >= displacedMVAMin_; }

  private:
    double displacedMVAMin_;
  };
  struct TTTrackWordDisplacedMVAMinSelector {
    TTTrackWordDisplacedMVAMinSelector(double displacedMVAMin) : displacedMVAMin_(displacedMVAMin) {}
    TTTrackWordDisplacedMVAMinSelector(const edm::ParameterSet& cfg)
        : displacedMVAMin_(cfg.template getParameter<double>("displacedMVAMin")) {}
    bool operator()(const L1Track& t) const { return t.trkMVA2() >= displacedMVAMin_; }  //change when mva bins in word are set

  private:
    double displacedMVAMin_;
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
  struct TTTrackChi2RZMaxNstubSelector {
    TTTrackChi2RZMaxNstubSelector(double reducedChi2RZMaxNstub4, double reducedChi2RZMaxNstub5)
        : reducedChi2RZMaxNstub4_(reducedChi2RZMaxNstub4), reducedChi2RZMaxNstub5_(reducedChi2RZMaxNstub5) {}
    TTTrackChi2RZMaxNstubSelector(const edm::ParameterSet& cfg)
        : reducedChi2RZMaxNstub4_(cfg.template getParameter<double>("reducedChi2RZMaxNstub4")),
          reducedChi2RZMaxNstub5_(cfg.template getParameter<double>("reducedChi2RZMaxNstub5")) {}
    bool operator()(const L1Track& t) const {
      return (((t.chi2ZRed() < reducedChi2RZMaxNstub4_) && (t.getStubRefs().size() == 4)) ||
              ((t.chi2ZRed() < reducedChi2RZMaxNstub5_) && (t.getStubRefs().size() > 4)));
    }

  private:
    double reducedChi2RZMaxNstub4_;
    double reducedChi2RZMaxNstub5_;
  };
  struct TTTrackWordChi2RZMaxNstubSelector {
    TTTrackWordChi2RZMaxNstubSelector(double reducedChi2RZMaxNstub4, double reducedChi2RZMaxNstub5)
        : reducedChi2RZMaxNstub4_(reducedChi2RZMaxNstub4), reducedChi2RZMaxNstub5_(reducedChi2RZMaxNstub5) {}
    TTTrackWordChi2RZMaxNstubSelector(const edm::ParameterSet& cfg)
        : reducedChi2RZMaxNstub4_(cfg.template getParameter<double>("reducedChi2RZMaxNstub4")),
          reducedChi2RZMaxNstub5_(cfg.template getParameter<double>("reducedChi2RZMaxNstub5")) {}
    bool operator()(const L1Track& t) const {
      return (((t.getChi2RZ() < reducedChi2RZMaxNstub4_) && (t.getNStubs() == 4)) ||
              ((t.getChi2RZ() < reducedChi2RZMaxNstub5_) && (t.getNStubs() > 4)));
    }

  private:
    double reducedChi2RZMaxNstub4_;
    double reducedChi2RZMaxNstub5_;
  };
  struct TTTrackChi2RPhiMaxNstubSelector {
    TTTrackChi2RPhiMaxNstubSelector(double reducedChi2RPhiMaxNstub4, double reducedChi2RPhiMaxNstub5)
        : reducedChi2RPhiMaxNstub4_(reducedChi2RPhiMaxNstub4), reducedChi2RPhiMaxNstub5_(reducedChi2RPhiMaxNstub5) {}
    TTTrackChi2RPhiMaxNstubSelector(const edm::ParameterSet& cfg)
        : reducedChi2RPhiMaxNstub4_(cfg.template getParameter<double>("reducedChi2RPhiMaxNstub4")),
          reducedChi2RPhiMaxNstub5_(cfg.template getParameter<double>("reducedChi2RPhiMaxNstub5")) {}
    bool operator()(const L1Track& t) const {
      return (((t.chi2XYRed() < reducedChi2RPhiMaxNstub4_) && (t.getStubRefs().size() == 4)) ||
              ((t.chi2XYRed() < reducedChi2RPhiMaxNstub5_) && (t.getStubRefs().size() > 4)));
    }

  private:
    double reducedChi2RPhiMaxNstub4_;
    double reducedChi2RPhiMaxNstub5_;
  };
  struct TTTrackWordChi2RPhiMaxNstubSelector {  // using simulated chi2 since not implemented in track word, updates needed
    TTTrackWordChi2RPhiMaxNstubSelector(double reducedChi2RPhiMaxNstub4, double reducedChi2RPhiMaxNstub5)
        : reducedChi2RPhiMaxNstub4_(reducedChi2RPhiMaxNstub4), reducedChi2RPhiMaxNstub5_(reducedChi2RPhiMaxNstub5) {}
    TTTrackWordChi2RPhiMaxNstubSelector(const edm::ParameterSet& cfg)
        : reducedChi2RPhiMaxNstub4_(cfg.template getParameter<double>("reducedChi2RPhiMaxNstub4")),
          reducedChi2RPhiMaxNstub5_(cfg.template getParameter<double>("reducedChi2RPhiMaxNstub5")) {}
    bool operator()(const L1Track& t) const {
      return (((t.getChi2RPhi() < reducedChi2RPhiMaxNstub4_) && (t.getNStubs() == 4)) ||
              ((t.getChi2RPhi() < reducedChi2RPhiMaxNstub5_) && (t.getNStubs() > 4)));
    }

  private:
    double reducedChi2RPhiMaxNstub4_;
    double reducedChi2RPhiMaxNstub5_;
  };
  struct TTTrackBendChi2MaxNstubSelector {
    TTTrackBendChi2MaxNstubSelector(double reducedBendChi2MaxNstub4, double reducedBendChi2MaxNstub5)
        : reducedBendChi2MaxNstub4_(reducedBendChi2MaxNstub4), reducedBendChi2MaxNstub5_(reducedBendChi2MaxNstub5) {}
    TTTrackBendChi2MaxNstubSelector(const edm::ParameterSet& cfg)
        : reducedBendChi2MaxNstub4_(cfg.template getParameter<double>("reducedBendChi2MaxNstub4")),
          reducedBendChi2MaxNstub5_(cfg.template getParameter<double>("reducedBendChi2MaxNstub5")) {}
    bool operator()(const L1Track& t) const {
      return (((t.stubPtConsistency() < reducedBendChi2MaxNstub4_) && (t.getStubRefs().size() == 4)) ||
              ((t.stubPtConsistency() < reducedBendChi2MaxNstub5_) && (t.getStubRefs().size() > 4)));
    }

  private:
    double reducedBendChi2MaxNstub4_;
    double reducedBendChi2MaxNstub5_;
  };
  struct TTTrackWordBendChi2MaxNstubSelector {
    TTTrackWordBendChi2MaxNstubSelector(double reducedBendChi2MaxNstub4, double reducedBendChi2MaxNstub5)
        : reducedBendChi2MaxNstub4_(reducedBendChi2MaxNstub4), reducedBendChi2MaxNstub5_(reducedBendChi2MaxNstub5) {}
    TTTrackWordBendChi2MaxNstubSelector(const edm::ParameterSet& cfg)
        : reducedBendChi2MaxNstub4_(cfg.template getParameter<double>("reducedBendChi2MaxNstub4")),
          reducedBendChi2MaxNstub5_(cfg.template getParameter<double>("reducedBendChi2MaxNstub5")) {}
    bool operator()(const L1Track& t) const {
      return (((t.getBendChi2() < reducedBendChi2MaxNstub4_) && (t.getNStubs() == 4)) ||
              ((t.getBendChi2() < reducedBendChi2MaxNstub5_) && (t.getNStubs() > 4)));
    }

  private:
    double reducedBendChi2MaxNstub4_;
    double reducedBendChi2MaxNstub5_;
  };

  struct TTTrackAbsD0MinEtaSelector {
    TTTrackAbsD0MinEtaSelector(double absD0MinEtaMin0p95, double absD0MinEtaMax0p95)
        : absD0MinEtaMin0p95_(absD0MinEtaMin0p95), absD0MinEtaMax0p95_(absD0MinEtaMax0p95) {}
    TTTrackAbsD0MinEtaSelector(const edm::ParameterSet& cfg)
        : absD0MinEtaMin0p95_(cfg.template getParameter<double>("absD0MinEtaMin0p95")),
          absD0MinEtaMax0p95_(cfg.template getParameter<double>("absD0MinEtaMax0p95")) {}
    bool operator()(const L1Track& t) const {
      return (((std::abs(t.d0())>absD0MinEtaMin0p95_) && (std::abs(t.momentum().eta())>0.95)) || ((std::abs(t.d0())>absD0MinEtaMax0p95_) && (std::abs(t.momentum().eta())<=0.95)));
    }

  private:
    double absD0MinEtaMin0p95_;
    double absD0MinEtaMax0p95_;
  };
  struct TTTrackWordAbsD0MinEtaSelector {
    TTTrackWordAbsD0MinEtaSelector(double absD0MinEtaMin0p95, double absD0MinEtaMax0p95)
        : absD0MinEtaMin0p95_(absD0MinEtaMin0p95), absD0MinEtaMax0p95_(absD0MinEtaMax0p95) {}
    TTTrackWordAbsD0MinEtaSelector(const edm::ParameterSet& cfg)
        : absD0MinEtaMin0p95_(cfg.template getParameter<double>("absD0MinEtaMin0p95")),
          absD0MinEtaMax0p95_(cfg.template getParameter<double>("absD0MinEtaMax0p95")) {}
    bool operator()(const L1Track& t) const {
      double floatD0 = t.undigitizeSignedValue(t.getD0Bits(), TTTrack_TrackWord::TrackBitWidths::kD0Size, TTTrack_TrackWord::stepD0, 0.0);
      TTTrack_TrackWord::tanl_t etaEmulationBits = t.getTanlWord();
      ap_fixed<TrackBitWidths::kEtaSize, TrackBitWidths::kEtaMagSize> etaEmulation;
      etaEmulation.V = etaEmulationBits.range();
      return (((std::abs(floatD0)>absD0MinEtaMin0p95_) && (std::abs(etaEmulation.to_double())>0.95)) || ((std::abs(floatD0)>absD0MinEtaMax0p95_) && (std::abs(etaEmulation.to_double())<=0.95)));
    }

  private:
    double absD0MinEtaMin0p95_;
    double absD0MinEtaMax0p95_;
  };

  typedef AndSelector<TTTrackPtMinSelector, TTTrackAbsEtaMaxSelector, TTTrackAbsZ0MaxSelector, TTTrackNStubsMinSelector, TTTrackNStubsMinEtaOverlapSelector>
      TTTrackPtMinEtaMaxZ0MaxNStubsMinSelector;
  typedef AndSelector<TTTrackWordPtMinSelector,
                      TTTrackWordAbsEtaMaxSelector,
                      TTTrackWordAbsZ0MaxSelector,
                      TTTrackWordNStubsMinSelector,
		      TTTrackWordNStubsMinEtaOverlapSelector>
      TTTrackWordPtMinEtaMaxZ0MaxNStubsMinSelector;
  typedef AndSelector<TTTrackBendChi2MaxSelector, TTTrackChi2RZMaxSelector, TTTrackChi2RPhiMaxSelector>
      TTTrackBendChi2Chi2RZChi2RPhiMaxSelector;
  typedef AndSelector<TTTrackWordBendChi2MaxSelector, TTTrackWordChi2RZMaxSelector, TTTrackWordChi2RPhiMaxSelector>
      TTTrackWordBendChi2Chi2RZChi2RPhiMaxSelector;
  typedef AndSelector<TTTrackChi2RZMaxNstubSelector, TTTrackChi2RPhiMaxNstubSelector, TTTrackBendChi2MaxNstubSelector>
      TTTrackChi2MaxNstubSelector;
  typedef AndSelector<TTTrackWordChi2RZMaxNstubSelector,
                      TTTrackWordChi2RPhiMaxNstubSelector,
                      TTTrackWordBendChi2MaxNstubSelector>
      TTTrackWordChi2MaxNstubSelector;
  typedef AndSelector<TTTrackPromptMVAMinSelector,
		      TTTrackPromptMVAMinD0Min1Selector,
		      TTTrackDisplacedMVAMinSelector>
      TTTrackMVAMinSelector;
  typedef AndSelector<TTTrackWordPromptMVAMinSelector,
		      TTTrackWordPromptMVAMinD0Min1Selector,
		      TTTrackWordDisplacedMVAMinSelector>
      TTTrackWordMVAMinSelector;
  
  // ----------member data ---------------------------
  const edm::EDGetTokenT<TTTrackCollection> l1TracksToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const std::string outputCollectionName_;
  const edm::ParameterSet cutSet_;
  const double ptMin_, absEtaMax_, absZ0Max_, promptMVAMin_, promptMVAMinD0Min1_, displacedMVAMin_, bendChi2Max_, reducedChi2RZMax_, reducedChi2RPhiMax_;
  const double reducedChi2RZMaxNstub4_, reducedChi2RZMaxNstub5_, reducedChi2RPhiMaxNstub4_, reducedChi2RPhiMaxNstub5_, reducedBendChi2MaxNstub4_, reducedBendChi2MaxNstub5_, absD0MinEtaMin0p95_, absD0MinEtaMax0p95_;
  const int nStubsMin_, nStubsMinEtaOverlap_, nPSStubsMin_;
  bool processSimulatedTracks_, processEmulatedTracks_;
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
      promptMVAMin_(cutSet_.getParameter<double>("promptMVAMin")),
      promptMVAMinD0Min1_(cutSet_.getParameter<double>("promptMVAMinD0Min1")),
      displacedMVAMin_(cutSet_.getParameter<double>("displacedMVAMin")),
      bendChi2Max_(cutSet_.getParameter<double>("reducedBendChi2Max")),
      reducedChi2RZMax_(cutSet_.getParameter<double>("reducedChi2RZMax")),
      reducedChi2RPhiMax_(cutSet_.getParameter<double>("reducedChi2RPhiMax")),
      reducedChi2RZMaxNstub4_(cutSet_.getParameter<double>("reducedChi2RZMaxNstub4")),
      reducedChi2RZMaxNstub5_(cutSet_.getParameter<double>("reducedChi2RZMaxNstub5")),
      reducedChi2RPhiMaxNstub4_(cutSet_.getParameter<double>("reducedChi2RPhiMaxNstub4")),
      reducedChi2RPhiMaxNstub5_(cutSet_.getParameter<double>("reducedChi2RPhiMaxNstub5")),
      reducedBendChi2MaxNstub4_(cutSet_.getParameter<double>("reducedBendChi2MaxNstub4")),
      reducedBendChi2MaxNstub5_(cutSet_.getParameter<double>("reducedBendChi2MaxNstub5")),
      absD0MinEtaMin0p95_(cutSet_.getParameter<double>("absD0MinEtaMin0p95")),
      absD0MinEtaMax0p95_(cutSet_.getParameter<double>("absD0MinEtaMax0p95")),
      nStubsMin_(cutSet_.getParameter<int>("nStubsMin")),
      nStubsMinEtaOverlap_(cutSet_.getParameter<int>("nStubsMinEtaOverlap")),
      nPSStubsMin_(cutSet_.getParameter<int>("nPSStubsMin")),
      processSimulatedTracks_(iConfig.getParameter<bool>("processSimulatedTracks")),
      processEmulatedTracks_(iConfig.getParameter<bool>("processEmulatedTracks")),
      debug_(iConfig.getParameter<int>("debug")) {
  // Confirm the the configuration makes sense
  if (!processSimulatedTracks_ && !processEmulatedTracks_) {
    throw cms::Exception("You must process at least one of the track collections (simulated or emulated).");
  }

  if (processSimulatedTracks_) {
    produces<TTTrackRefCollection>(outputCollectionName_);
  }
  if (processEmulatedTracks_) {
    produces<TTTrackRefCollection>(outputCollectionName_ + "Emulation");
  }
}

L1TrackSelectionProducer::~L1TrackSelectionProducer() {}

//
// member functions
//

void L1TrackSelectionProducer::printDebugInfo(const TTTrackCollectionHandle& l1TracksHandle,
                                              const TTTrackRefCollectionUPtr& vTTTrackOutput,
                                              const TTTrackRefCollectionUPtr& vTTTrackEmulationOutput) const {
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
    double floatTkZ0 = track.undigitizeSignedValue(
        track.getZ0Bits(), TTTrack_TrackWord::TrackBitWidths::kZ0Size, TTTrack_TrackWord::stepZ0, 0.0);
    double floatTkPhi = track.undigitizeSignedValue(
        track.getPhiBits(), TTTrack_TrackWord::TrackBitWidths::kPhiSize, TTTrack_TrackWord::stepPhi0, 0.0);
    log << "\t\t(" << ptEmulation.to_double() << ", " << etaEmulation.to_double() << ", " << floatTkPhi << ", "
        << track.getNStubs() << ", " << track.getBendChi2() << ", " << track.getChi2RZ() << ", " << track.getChi2RPhi()
        << ", " << floatTkZ0 << ")\n";
  }
}

// ------------ method called to produce the data  ------------
void L1TrackSelectionProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto vTTTrackOutput = std::make_unique<TTTrackRefCollection>();
  auto vTTTrackEmulationOutput = std::make_unique<TTTrackRefCollection>();

  // Tracker Topology
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);

  TTTrackCollectionHandle l1TracksHandle;

  iEvent.getByToken(l1TracksToken_, l1TracksHandle);
  size_t nOutputApproximate = l1TracksHandle->size();
  if (processSimulatedTracks_) {
    vTTTrackOutput->reserve(nOutputApproximate);
  }
  if (processEmulatedTracks_) {
    vTTTrackEmulationOutput->reserve(nOutputApproximate);
  }

  TTTrackPtMinEtaMaxZ0MaxNStubsMinSelector kinSel(ptMin_, absEtaMax_, absZ0Max_, nStubsMin_, nStubsMinEtaOverlap_);
  TTTrackWordPtMinEtaMaxZ0MaxNStubsMinSelector kinSelEmu(ptMin_, absEtaMax_, absZ0Max_, nStubsMin_, nStubsMinEtaOverlap_);
  TTTrackBendChi2Chi2RZChi2RPhiMaxSelector chi2Sel(bendChi2Max_, reducedChi2RZMax_, reducedChi2RPhiMax_);
  TTTrackWordBendChi2Chi2RZChi2RPhiMaxSelector chi2SelEmu(bendChi2Max_, reducedChi2RZMax_, reducedChi2RPhiMax_);
  TTTrackNPSStubsMinSelector nPSStubsSel(nPSStubsMin_, tTopo);
  TTTrackMVAMinSelector mvaSel(promptMVAMin_, promptMVAMinD0Min1_, displacedMVAMin_);
  TTTrackWordMVAMinSelector mvaSelEmu(promptMVAMin_, promptMVAMinD0Min1_, displacedMVAMin_);
  TTTrackChi2MaxNstubSelector chi2NstubSel({reducedChi2RZMaxNstub4_, reducedChi2RZMaxNstub5_},
                                           {reducedChi2RPhiMaxNstub4_, reducedChi2RPhiMaxNstub5_},
                                           {reducedBendChi2MaxNstub4_, reducedBendChi2MaxNstub5_});
  TTTrackWordChi2MaxNstubSelector chi2NstubSelEmu({reducedChi2RZMaxNstub4_, reducedChi2RZMaxNstub5_},
                                                  {reducedChi2RPhiMaxNstub4_, reducedChi2RPhiMaxNstub5_},
                                                  {reducedBendChi2MaxNstub4_, reducedBendChi2MaxNstub5_});
  TTTrackAbsD0MinEtaSelector d0Sel(absD0MinEtaMin0p95_,absD0MinEtaMax0p95_);
  TTTrackWordAbsD0MinEtaSelector d0SelEmu(absD0MinEtaMin0p95_,absD0MinEtaMax0p95_);

  for (size_t i = 0; i < nOutputApproximate; i++) {
    const auto& track = l1TracksHandle->at(i);

    // Select tracks based on the floating point TTTrack
    if (processSimulatedTracks_ && kinSel(track) && nPSStubsSel(track) && chi2Sel(track) && mvaSel(track) && d0Sel(track) &&
        chi2NstubSel(track)) {
      vTTTrackOutput->push_back(TTTrackRef(l1TracksHandle, i));
    }

    // Select tracks based on the bitwise accurate TTTrack_TrackWord
    if (processEmulatedTracks_ && kinSelEmu(track) && chi2SelEmu(track) && mvaSelEmu(track) && d0SelEmu(track) && chi2NstubSelEmu(track)) {
      vTTTrackEmulationOutput->push_back(TTTrackRef(l1TracksHandle, i));
    }
  }

  if (debug_ >= 2) {
    printDebugInfo(l1TracksHandle, vTTTrackOutput, vTTTrackEmulationOutput);
  }

  // Put the outputs into the event
  if (processSimulatedTracks_) {
    iEvent.put(std::move(vTTTrackOutput), outputCollectionName_);
  }
  if (processEmulatedTracks_) {
    iEvent.put(std::move(vTTTrackEmulationOutput), outputCollectionName_ + "Emulation");
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TrackSelectionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //L1TrackSelectionProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1TracksInputTag", edm::InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.add<std::string>("outputCollectionName", "Level1TTTracksSelected");
  {
    edm::ParameterSetDescription descCutSet;
    descCutSet.add<double>("ptMin", 2.0)->setComment("pt must be greater than this value, [GeV]");
    descCutSet.add<double>("absEtaMax", 2.4)->setComment("absolute value of eta must be less than this value");
    descCutSet.add<double>("absZ0Max", 15.0)->setComment("z0 must be less than this value, [cm]");
    descCutSet.add<int>("nStubsMin", 4)->setComment("number of stubs must be greater than or equal to this value");
    descCutSet.add<int>("nStubsMinEtaOverlap", 4)->setComment("number of stubs must be greater than or equal to this value for tracks with 1.1<|eta|<1.7");
    descCutSet.add<int>("nPSStubsMin", 0)
        ->setComment("number of stubs in the PS Modules must be greater than or equal to this value");

    descCutSet.add<double>("promptMVAMin", -1.0)->setComment("MVA must be greater than this value");
    descCutSet.add<double>("promptMVAMinD0Min1", -1.0)->setComment("MVA for tracks with |d0|>1cm must be greater than this value");
    descCutSet.add<double>("displacedMVAMin", -1.0)->setComment("Displaced MVA must be greater than this value");
    descCutSet.add<double>("reducedBendChi2Max", 2.25)->setComment("bend chi2 must be less than this value");
    descCutSet.add<double>("reducedChi2RZMax", 5.0)->setComment("chi2rz/dof must be less than this value");
    descCutSet.add<double>("reducedChi2RPhiMax", 20.0)->setComment("chi2rphi/dof must be less than this value");
    descCutSet.add<double>("reducedChi2RZMaxNstub4", 999.9)
        ->setComment("chi2rz/dof must be less than this value in nstub==4");
    descCutSet.add<double>("reducedChi2RZMaxNstub5", 999.9)
        ->setComment("chi2rz/dof must be less than this value in nstub>4");
    descCutSet.add<double>("reducedChi2RPhiMaxNstub4", 999.9)
        ->setComment("chi2rphi/dof must be less than this value in nstub==4");
    descCutSet.add<double>("reducedChi2RPhiMaxNstub5", 999.9)
        ->setComment("chi2rphi/dof must be less than this value in nstub>4");
    descCutSet.add<double>("reducedBendChi2MaxNstub4", 999.9)
        ->setComment("bend chi2 must be less than this value in nstub==4");
    descCutSet.add<double>("reducedBendChi2MaxNstub5", 999.9)
        ->setComment("bend chi2 must be less than this value in nstub>4");
    descCutSet.add<double>("absD0MinEtaMin0p95", -1.0)->setComment("absolute value of d0 must be greater than this value for tracks with |eta|>0.95");
    descCutSet.add<double>("absD0MinEtaMax0p95", -1.0)->setComment("absolute value of d0 must be greater than this value for tracks with |eta|<=0.95");

    desc.add<edm::ParameterSetDescription>("cutSet", descCutSet);
  }
  desc.add<bool>("processSimulatedTracks", true)
      ->setComment("return selected tracks after cutting on the floating point values");
  desc.add<bool>("processEmulatedTracks", true)
      ->setComment("return selected tracks after cutting on the bitwise emulated values");
  desc.add<int>("debug", 0)->setComment("Verbosity levels: 0, 1, 2, 3");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackSelectionProducer);
