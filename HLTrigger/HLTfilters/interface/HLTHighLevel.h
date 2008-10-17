#ifndef HLTHighLevel_h
#define HLTHighLevel_h

/** \class HLTHighLevel
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  HLT bits
 *
 *  $Date: 2007/07/12 08:50:55 $
 *  $Revision: 1.3 $
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
    virtual bool beginRun(edm::Run&, const edm::EventSetup&);
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:

    /// HLT TriggerResults EDProduct
    edm::InputTag inputTag_;
    /// HLT trigger names
    edm::TriggerNames triggerNames_;

    /// false=and-mode (all requested triggers), true=or-mode (at least one)
    bool andOr_;

    /// throw on any requested trigger being unknown
    bool throw_;

    /// number of HLT trigger paths requested in configuration
    unsigned int n_;

    /// first message
    bool first_;

    /// list of required HLT triggers by HLT name
    std::vector<std::string > HLTPathsByName_;
    /// list of required HLT triggers by HLT index
    std::vector<unsigned int> HLTPathsByIndex_;

};

#endif //HLTHighLevel_h
