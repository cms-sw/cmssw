#ifndef PhysicsTools_PatAlgos_PATTriggerProducer_h
#define PhysicsTools_PatAlgos_PATTriggerProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerProducer
//
//
/**
  \class    pat::PATTriggerProducer PATTriggerProducer.h "PhysicsTools/PatAlgos/plugins/PATTriggerProducer.h"
  \brief    Produces the full or stand-alone PAT trigger information collections

   This producer extracts the trigger information from
   - the edm::TriggerResults written by the HLT process,
   - the corresponding trigger::TriggerEvent,
   - the provenance information,
   - the process history,
   - the GlobalTrigger information in the event and the event setup and
   - the L1 object collections ("l1extra")
   re-arranges it and writes it either (full mode) to
   - a pat::TriggerObjectCollection,
   - a pat::TriggerFilterCollection,
   - a pat::TriggerPathCollection,
   - a pat::TriggerAlgorithmCollection (optionally filled or empty) and
   - a pat::TriggerConditionCollection (optionally filled or empty)
   or (stand-alone mode) to
   - a pat::TriggerObjectStandAloneCollection

   For me information, s.
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger

  \author   Volker Adler
  \version  $Id: PATTriggerProducer.h,v 1.20 2012/09/11 22:45:29 vadler Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMaps.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"

namespace pat {

  class PATTriggerProducer : public edm::stream::EDProducer<> {

    public:

      explicit PATTriggerProducer( const edm::ParameterSet & iConfig );
      ~PATTriggerProducer() {};

    private:

      virtual void beginRun(const edm::Run & iRun, const edm::EventSetup& iSetup) override;
      virtual void beginLuminosityBlock(const edm::LuminosityBlock & iLuminosityBlock, const edm::EventSetup& iSetup) override;
      virtual void produce( edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      std::string nameProcess_;     // configuration
      bool        autoProcessName_;
      bool        onlyStandAlone_;  // configuration
      bool        firstInRun_;
      // L1
      edm::ParameterSet * l1PSet_;
      bool                addL1Algos_;                    // configuration (optional with default)
      edm::InputTag       tagL1GlobalTriggerObjectMaps_;  // configuration (optional with default)
      edm::EDGetTokenT< L1GlobalTriggerObjectMaps > l1GlobalTriggerObjectMapsToken_;
      edm::InputTag       tagL1ExtraMu_;                  // configuration (optional)
      edm::GetterOfProducts< l1extra::L1MuonParticleCollection > l1ExtraMuGetter_;
      edm::InputTag       tagL1ExtraNoIsoEG_;             // configuration (optional)
      edm::GetterOfProducts< l1extra::L1EmParticleCollection > l1ExtraNoIsoEGGetter_;
      edm::InputTag       tagL1ExtraIsoEG_;               // configuration (optional)
      edm::GetterOfProducts< l1extra::L1EmParticleCollection > l1ExtraIsoEGGetter_;
      edm::InputTag       tagL1ExtraCenJet_;              // configuration (optional)
      edm::GetterOfProducts< l1extra::L1JetParticleCollection > l1ExtraCenJetGetter_;
      edm::InputTag       tagL1ExtraForJet_;              // configuration (optional)
      edm::GetterOfProducts< l1extra::L1JetParticleCollection > l1ExtraForJetGetter_;
      edm::InputTag       tagL1ExtraTauJet_;              // configuration (optional)
      edm::GetterOfProducts< l1extra::L1JetParticleCollection > l1ExtraTauJetGetter_;
      edm::InputTag       tagL1ExtraETM_;                 // configuration (optional)
      edm::GetterOfProducts< l1extra::L1EtMissParticleCollection > l1ExtraETMGetter_;
      edm::InputTag       tagL1ExtraHTM_;                 // configuration (optional)
      edm::GetterOfProducts< l1extra::L1EtMissParticleCollection > l1ExtraHTMGetter_;
      bool                autoProcessNameL1ExtraMu_;
      bool                autoProcessNameL1ExtraNoIsoEG_;
      bool                autoProcessNameL1ExtraIsoEG_;
      bool                autoProcessNameL1ExtraCenJet_;
      bool                autoProcessNameL1ExtraForJet_;
      bool                autoProcessNameL1ExtraTauJet_;
      bool                autoProcessNameL1ExtraETM_;
      bool                autoProcessNameL1ExtraHTM_;
      bool                mainBxOnly_;                    // configuration (optional with default)
      bool                saveL1Refs_;                    // configuration (optional with default)
      // HLT
      HLTPrescaleProvider hltPrescaleProvider_;
      bool                      hltConfigInit_;
      edm::InputTag             tagTriggerResults_;     // configuration (optional with default)
      edm::GetterOfProducts< edm::TriggerResults > triggerResultsGetter_;
      edm::InputTag             tagTriggerEvent_;       // configuration (optional with default)
      edm::GetterOfProducts< trigger::TriggerEvent > triggerEventGetter_;
      std::string               hltPrescaleLabel_;      // configuration (optional)
      std::string               labelHltPrescaleTable_; // configuration (optional)
      edm::GetterOfProducts< trigger::HLTPrescaleTable > hltPrescaleTableRunGetter_;
      edm::GetterOfProducts< trigger::HLTPrescaleTable > hltPrescaleTableLumiGetter_;
      edm::GetterOfProducts< trigger::HLTPrescaleTable > hltPrescaleTableEventGetter_;
      trigger::HLTPrescaleTable hltPrescaleTableRun_;
      trigger::HLTPrescaleTable hltPrescaleTableLumi_;
      bool                       addPathModuleLabels_;  // configuration (optional with default)
      std::vector< std::string > exludeCollections_;    // configuration (optional)
      bool                      packPathNames_;         // configuration (optional width default)
      bool                      packLabels_;         // configuration (optional width default)
      bool                      packPrescales_;         // configuration (optional width default)

      class ModuleLabelToPathAndFlags {
          public:
              struct PathAndFlags {
                PathAndFlags(const std::string &name, unsigned int index, bool last, bool l3) : pathName(name), pathIndex(index), lastFilter(last), l3Filter(l3) {}
                PathAndFlags() {}
                std::string pathName;
                unsigned int pathIndex;
                bool lastFilter;
                bool l3Filter;
              };
              void init(const HLTConfigProvider &) ;
              void clear() { map_.clear(); }
              const std::vector<PathAndFlags> & operator[](const std::string & filter) const {
                  std::map<std::string,std::vector<PathAndFlags> >::const_iterator it = map_.find(filter);
                  return (it == map_.end() ? empty_ : it->second);
              }
          private:
              void insert(const std::string & filter, const std::string &path, unsigned int pathIndex, bool lastFilter, bool l3Filter) {
                  map_[filter].push_back(PathAndFlags(path, pathIndex, lastFilter, l3Filter));
              }
              std::map<std::string,std::vector<PathAndFlags> > map_;
              const std::vector<PathAndFlags> empty_;
      };
      ModuleLabelToPathAndFlags moduleLabelToPathAndFlags_;

  };
}


#endif
