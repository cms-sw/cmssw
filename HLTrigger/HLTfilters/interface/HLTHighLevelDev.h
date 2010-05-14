#ifndef HLTHighLevelDev_h
#define HLTHighLevelDev_h

/** \class HLTHighLevelDev
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  HLT bits
 *
 *  $Date: 2010/02/11 00:12:12 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/ESWatcher.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include <vector>
#include <string>

// forward declarations
namespace edm {
  class TriggerResults;
  class TriggerNames;
}
class AlCaRecoTriggerBitsRcd;

//
// class declaration
//

class HLTHighLevelDev : public HLTFilter {

  public:

    explicit HLTHighLevelDev(const edm::ParameterSet&);
    ~HLTHighLevelDev();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

    /// get HLTPaths with key 'key' from EventSetup (AlCaRecoTriggerBitsRcd)
    std::vector<std::string> pathsFromSetup(const std::string &key,
                                            const edm::EventSetup &iSetup) const;
  private:
    /// initialize the trigger conditions (call this if the trigger paths have changed)
    void init(const edm::TriggerResults & results,
              const edm::EventSetup &iSetup,
              const edm::TriggerNames & triggerNames);

    /// HLT TriggerResults EDProduct
    edm::InputTag inputTag_;

    /// HLT trigger names
    edm::ParameterSetID triggerNamesID_;

    /// false = and-mode (all requested triggers), true = or-mode (at least one)
    bool andOr_;

    /// throw on any requested trigger being unknown
    bool throw_;

    /// not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
    const std::string eventSetupPathsKey_;
    /// Watcher to be created and used if 'eventSetupPathsKey_' non empty:
    edm::ESWatcher<AlCaRecoTriggerBitsRcd> *watchAlCaRecoTriggerBitsRcd_;

    /// input patterns that will be expanded into trigger names
    std::vector<std::string>  HLTPatterns_;

    /// for each trigger/pattern define a prescale.
    /// Only a subset of the events per trigger will be selected
    std::vector<uint32_t> HLTPrescales_;
    /// This variable stores the prescales after pattern matching and expansion
    std::vector<uint32_t> HLTPrescalesExpanded_; 
    /// Scalers for the prescaling, L1 style.
    std::vector<uint32_t> HLTPrescalesScalers; 

    /// You can define a prescale after the filter logic has been applied
    /// this is different from the single prescales. It can be applied on
    /// top of the single trigger prescales
    uint32_t HLTOverallPrescale_ ;
    uint32_t HLTOverallPrescalesScaler_;

    /// list of required HLT triggers by HLT name
    std::vector<std::string> HLTPathsByName_;

    /// list of required HLT triggers by HLT index
    std::vector<unsigned int> HLTPathsByIndex_;
};

#endif //HLTHighLevelDev_h
