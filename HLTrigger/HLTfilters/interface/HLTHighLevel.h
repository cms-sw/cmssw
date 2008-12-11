#ifndef HLTHighLevel_h
#define HLTHighLevel_h

/** \class HLTHighLevel
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  HLT bits
 *
 *  $Date: 2008/11/05 14:03:10 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/ESWatcher.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include <vector>
#include <string>

// forward declarations
namespace edm {
  class TriggerResults;
}
class AlCaRecoTriggerBitsRcd;

//
// class declaration
//

class HLTHighLevel : public HLTFilter {

  public:

    explicit HLTHighLevel(const edm::ParameterSet&);
    ~HLTHighLevel();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

    /// get HLTPaths with key 'key' from EventSetup (AlCaRecoTriggerBitsRcd)
    std::vector<std::string> pathsFromSetup(const std::string &key,
					    const edm::EventSetup &iSetup) const;
  private:
    /// initialize the trigger conditions (call this if the trigger paths have changed)
    void init(const edm::TriggerResults & results, const edm::EventSetup &iSetup);

    /// HLT TriggerResults EDProduct
    edm::InputTag inputTag_;

    /// HLT trigger names
    edm::TriggerNames triggerNames_;

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

    /// list of required HLT triggers by HLT name
    std::vector<std::string>  HLTPathsByName_;

    /// list of required HLT triggers by HLT index
    std::vector<unsigned int> HLTPathsByIndex_;
};

#endif //HLTHighLevel_h
