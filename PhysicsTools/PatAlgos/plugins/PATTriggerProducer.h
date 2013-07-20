#ifndef PhysicsTools_PatAlgos_PATTriggerProducer_h
#define PhysicsTools_PatAlgos_PATTriggerProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerProducer
//
// $Id: PATTriggerProducer.h,v 1.21 2013/02/27 23:26:56 wmtan Exp $
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
  \version  $Id: PATTriggerProducer.h,v 1.21 2013/02/27 23:26:56 wmtan Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

namespace pat {

  class PATTriggerProducer : public edm::EDProducer {

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
      L1GtUtils           l1GtUtils_;
      edm::ParameterSet * l1PSet_;
      bool                addL1Algos_;                    // configuration (optional with default)
      edm::InputTag       tagL1GlobalTriggerObjectMaps_;  // configuration (optional with default)
      edm::InputTag       tagL1ExtraMu_;                  // configuration (optional)
      edm::InputTag       tagL1ExtraNoIsoEG_;             // configuration (optional)
      edm::InputTag       tagL1ExtraIsoEG_;               // configuration (optional)
      edm::InputTag       tagL1ExtraCenJet_;              // configuration (optional)
      edm::InputTag       tagL1ExtraForJet_;              // configuration (optional)
      edm::InputTag       tagL1ExtraTauJet_;              // configuration (optional)
      edm::InputTag       tagL1ExtraETM_;                 // configuration (optional)
      edm::InputTag       tagL1ExtraHTM_;                 // configuration (optional)
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
      HLTConfigProvider         hltConfig_;
      bool                      hltConfigInit_;
      edm::InputTag             tagTriggerResults_;     // configuration (optional with default)
      edm::InputTag             tagTriggerEvent_;       // configuration (optional with default)
      std::string               hltPrescaleLabel_;      // configuration (optional)
      std::string               labelHltPrescaleTable_; // configuration (optional)
      trigger::HLTPrescaleTable hltPrescaleTableRun_;
      trigger::HLTPrescaleTable hltPrescaleTableLumi_;
      bool                       addPathModuleLabels_;  // configuration (optional with default)
      std::vector< std::string > exludeCollections_;    // configuration (optional)

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
              void init(const HLTConfigProvider &conf) ;
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
