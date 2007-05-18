#ifndef HLTLevel1Seed_h
#define HLTLevel1Seed_h

/** \class HLTLevel1Seed
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  Level-1 bits and extraction of seed objects.
 *
 *  $Date: 2007/04/13 15:57:57 $
 *  $Revision: 1.19 $
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

    edm::InputTag l1CollectionsTag_;  // L1 Extra EDProduct for particle collections
    edm::InputTag l1ParticleMapTag_;  // L1 Extra EDProduct for particle map
    edm::InputTag l1GTReadoutRecTag_; // L1 Extra EDProduct for L1 GT RR

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
