#ifndef HLTHighLevel_h
#define HLTHighLevel_h

/** \class HLTHighLevel
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  HLT bits
 *
 *  $Date: 2007/04/13 15:57:57 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
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

    edm::InputTag inputTag_; // HLT TriggerResults EDProduct

    bool andOr_;
    // false=and-mode (all), true=or-mode(at least one)

    bool byName_;
    // list of HLT triggers provided by: 
    // true: HLT Names (vstring) or false: HLT Index (vuint32)

    unsigned int n_;
    // number of HLT trigger paths requested in configuration

    std::vector<std::string > HLTPathsByName_;
    std::vector<unsigned int> HLTPathsByIndex_;
    // list of required HLT triggers by HLT name and by HLT index

};

#endif //HLTHighLevel_h
