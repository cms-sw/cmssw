#ifndef HLTHighLevel_h
#define HLTHighLevel_h

/** \class HLTHighLevel
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  HLT bits
 *
 *
 *  \author Martin Grunewald
 *
 */

// C++ headers
#include <vector>
#include <string>
#include <optional>

// CMSSW headers
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

// forward declarations
namespace edm {
  class ConfigurationDescriptions;
  class TriggerResults;
}  // namespace edm

class AlCaRecoTriggerBits;
class AlCaRecoTriggerBitsRcd;

//
// class declaration
//

class HLTHighLevel : public edm::stream::EDFilter<> {
public:
  explicit HLTHighLevel(const edm::ParameterSet &);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  bool filter(edm::Event &, const edm::EventSetup &) override;

  /// get HLTPaths with key 'key' from EventSetup (AlCaRecoTriggerBitsRcd)
  std::vector<std::string> pathsFromSetup(const std::string &key,
                                          const edm::Event &,
                                          const edm::EventSetup &iSetup) const;

private:
  /// initialize the trigger conditions (call this if the trigger paths have changed)
  void init(const edm::TriggerResults &results,
            const edm::Event &,
            const edm::EventSetup &iSetup,
            const edm::TriggerNames &triggerNames);

  /// HLT TriggerResults EDProduct
  edm::InputTag inputTag_;
  edm::EDGetTokenT<edm::TriggerResults> inputToken_;

  /// HLT trigger names
  edm::ParameterSetID triggerNamesID_;

  /// false = and-mode (all requested triggers), true = or-mode (at least one)
  bool andOr_;

  /// throw on any requested trigger being unknown
  bool throw_;

  /// stolen from HLTFilter
  std::string const &pathName(const edm::Event &) const;
  std::string const &moduleLabel() const;

  /// not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
  const std::string eventSetupPathsKey_;
  /// Watcher to be created and used if 'eventSetupPathsKey_' non empty:
  std::optional<edm::ESWatcher<AlCaRecoTriggerBitsRcd>> watchAlCaRecoTriggerBitsRcd_;
  /// ESGetToken to read AlCaRecoTriggerBits
  edm::ESGetToken<AlCaRecoTriggerBits, AlCaRecoTriggerBitsRcd> alcaRecotriggerBitsToken_;

  /// input patterns that will be expanded into trigger names
  std::vector<std::string> HLTPatterns_;

  /// list of required HLT triggers by HLT name
  std::vector<std::string> HLTPathsByName_;

  /// list of required HLT triggers by HLT index
  std::vector<unsigned int> HLTPathsByIndex_;
};

#endif  //HLTHighLevel_h
