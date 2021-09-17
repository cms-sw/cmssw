#ifndef DQM_L1TMONITOR_L1TMENUHELPER_H
#define DQM_L1TMONITOR_L1TMENUHELPER_H

/*
 * \file L1TMenuHelper.h
 *
 * \author J. Pela
 *
*/

// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include "TString.h"

// Simplified structure for single object conditions information
struct SingleObjectCondition {
  std::string name;
  L1GtConditionCategory conditionCategory;
  L1GtConditionType conditionType;
  L1GtObject object;
  unsigned int quality;
  unsigned int etaRange;
  unsigned int threshold;
};

// Simplified structure for single object conditions information
struct SingleObjectTrigger {
  L1GtObject object;
  std::string alias;
  unsigned int bit;
  int prescale;
  unsigned int threshold;  //
  unsigned int quality;    // Only aplicable to Muons
  unsigned int etaRange;   // Only aplicable to Muons

  bool operator<(const SingleObjectTrigger& iSOT) const {
    if (this->etaRange > iSOT.etaRange) {
      return true;
    } else if (this->etaRange < iSOT.etaRange) {
      return false;
    }

    if (this->prescale < iSOT.prescale) {
      return true;
    } else if (this->prescale > iSOT.prescale) {
      return false;
    }

    if (this->quality > iSOT.quality) {
      return true;
    } else if (this->quality < iSOT.quality) {
      return false;
    }

    return this->threshold < iSOT.threshold;
  }
};

class L1TMenuHelper {
public:
  struct Tokens {
    edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> menu;
    edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd> l1GtPfAlgo;
  };

  template <edm::Transition Tr = edm::Transition::Event>
  static Tokens consumes(edm::ConsumesCollector iC) {
    Tokens tok;
    tok.menu = iC.esConsumes<Tr>();
    tok.l1GtPfAlgo = iC.esConsumes<Tr>();
    return tok;
  }

  L1TMenuHelper(const edm::EventSetup& iSetup, const Tokens& tokens);  // Constructor
  ~L1TMenuHelper();                                                    // Destructor

  // Get Lowest Unprescaled Single Object Triggers
  std::map<std::string, std::string> getLUSOTrigger(const std::map<std::string, bool>& iCategories,
                                                    int IndexRefPrescaleFactors,
                                                    L1GtUtils const& myUtils);
  std::map<std::string, std::string> testAlgos(const std::map<std::string, std::string>&);

  // To convert enum to strings
  std::string enumToStringL1GtObject(L1GtObject iObject);
  std::string enumToStringL1GtConditionType(L1GtConditionType iConditionType);
  std::string enumToStringL1GtConditionCategory(L1GtConditionCategory iConditionCategory);

  // Getters
  int getPrescaleByAlias(const TString& iCategory, const TString& iAlias);
  unsigned int getEtaRangeByAlias(const TString& iCategory, const TString& iAlias);
  unsigned int getQualityAlias(const TString& iCategory, const TString& iAlias);

private:
  const L1GtTriggerMenu* m_l1GtMenu;
  const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;

  // Vectors to hold significant information about single object triggers
  std::vector<SingleObjectTrigger> m_vTrigMu;
  std::vector<SingleObjectTrigger> m_vTrigEG;
  std::vector<SingleObjectTrigger> m_vTrigIsoEG;
  std::vector<SingleObjectTrigger> m_vTrigJet;
  std::vector<SingleObjectTrigger> m_vTrigCenJet;
  std::vector<SingleObjectTrigger> m_vTrigForJet;
  std::vector<SingleObjectTrigger> m_vTrigTauJet;
  std::vector<SingleObjectTrigger> m_vTrigETM;
  std::vector<SingleObjectTrigger> m_vTrigETT;
  std::vector<SingleObjectTrigger> m_vTrigHTT;
  std::vector<SingleObjectTrigger> m_vTrigHTM;
};

#endif
