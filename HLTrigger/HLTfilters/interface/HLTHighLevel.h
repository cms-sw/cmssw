#ifndef HLTHighLevel_h
#define HLTHighLevel_h

/** \class HLTHighLevel
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  HLT bits
 *
 *  $Date: 2008/10/17 09:57:43 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include <vector>
#include <string>

// forward declarations
namespace edm {
  class TriggerResults;
}

//
// class declaration
//

class HLTHighLevel : public HLTFilter {

  public:

    explicit HLTHighLevel(const edm::ParameterSet&);
    ~HLTHighLevel();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:
    /// initialize the trigger conditions (call this if the trigger paths have changed)
    void init(const edm::TriggerResults & results);

    /// HLT TriggerResults EDProduct
    edm::InputTag inputTag_;

    /// HLT trigger names
    edm::TriggerNames triggerNames_;

    /// false = and-mode (all requested triggers), true = or-mode (at least one)
    bool andOr_;

    /// throw on any requested trigger being unknown
    bool throw_;

    /// input patterns that wil be expanded into trigger names
    std::vector<std::string>  HLTPatterns_;

    /// list of required HLT triggers by HLT name
    std::vector<std::string>  HLTPathsByName_;

    /// list of required HLT triggers by HLT index
    std::vector<unsigned int> HLTPathsByIndex_;
};

#endif //HLTHighLevel_h
