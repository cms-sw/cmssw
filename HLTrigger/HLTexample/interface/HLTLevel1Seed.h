#ifndef HLTLevel1Seed_h
#define HLTLevel1Seed_h

/** \class HLTLevel1Seed
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  Level-1 bits and extraction of seed objects.
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
// class declaration
//

class HLTLevel1Seed : public HLTFilter {

  public:

    explicit HLTLevel1Seed(const edm::ParameterSet&);
    ~HLTLevel1Seed();
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:

    edm::InputTag inputTag_; // L1 Extra EDProduct

    bool andOr_;
    // false=and-mode (all), true=or-mode(at least one)

    bool byName_;
    // list of L1 triggers provided by: 
    // true: L1 Names (vstring) or false: L1 Types (vuint32)

    std::vector<std::string > L1SeedsByName_;
    std::vector<unsigned int> L1SeedsByType_;
    // list of required L1 triggers by L1 name and by L1 type

};

#endif //HLTLevel1Seed_h
