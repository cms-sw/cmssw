#ifndef HLTHighLevel_h
#define HLTHighLevel_h

/** \class HLTHighLevel
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  HLT bits
 *
 *  $Date: 2006/08/23 17:03:01 $
 *  $Revision: 1.14 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>
#include<string>

//
// class decleration
//

class HLTHighLevel : public HLTFilter {

  public:

    explicit HLTHighLevel(const edm::ParameterSet&);
    ~HLTHighLevel();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:

    edm::InputTag TriggerResultsTag_;
    // HLT TriggerResults EDProduct

    bool andOr_;
    // false=and-mode (all), true=or-mode(at least one)

    bool byName_;
    // list of HLT triggers provided by: 
    // true: HLT Names (vstring) or false: HLT Index (vuint32)

    unsigned int n_;
    // number of HLT trigger paths requested in configuration

    std::vector<std::string > HLTPathByName_;
    std::vector<unsigned int> HLTPathByIndex_;
    // list of required HLT triggers by HLT name and by HLT index

};

#endif //HLTHighLevel_h
