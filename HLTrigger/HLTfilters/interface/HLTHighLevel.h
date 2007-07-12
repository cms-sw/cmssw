#ifndef HLTHighLevel_h
#define HLTHighLevel_h

/** \class HLTHighLevel
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  HLT bits
 *
 *  $Date: 2007/06/19 12:31:18 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include<vector>
#include<string>

//
// class declaration
//

class HLTHighLevel : public HLTFilter {

  public:

    explicit HLTHighLevel(const edm::ParameterSet&);
    ~HLTHighLevel();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:

    /// HLT TriggerResults EDProduct
    edm::InputTag inputTag_;
    /// HLT trigger names
    edm::TriggerNames triggerNames_;

    /// false=and-mode (all requested triggers), true=or-mode (at least one)
    bool andOr_;

    /*
    // user provides: true: HLT Names (vstring), or false: HLT Index (vuint32)
    // bool byName_;
    // disabled: user must always provide names, never indices
    */

    /// number of HLT trigger paths requested in configuration
    unsigned int n_;

    /// list of required HLT triggers by HLT name
    std::vector<std::string > HLTPathsByName_;
    /// list of required HLT triggers by HLT index
    std::vector<unsigned int> HLTPathsByIndex_;

};

#endif //HLTHighLevel_h
