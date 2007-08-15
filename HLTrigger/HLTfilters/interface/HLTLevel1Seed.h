#ifndef HLTLevel1Seed_h
#define HLTLevel1Seed_h

/** \class HLTLevel1Seed
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  Level-1 bits and extraction of seed objects.
 *
 *  $Date: 2007/07/12 08:50:55 $
 *  $Revision: 1.2 $
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

    /// L1 Extra EDProduct for particle collections 
    edm::InputTag l1CollectionsTag_;
    /// L1 Extra EDProduct for particle map
    edm::InputTag l1ParticleMapTag_;
    /// L1 Extra EDProduct for L1 GT RR
    edm::InputTag l1GTReadoutRecTag_;

    /// false=and-mode (all requested triggers), true=or-mode (at least one)
    bool andOr_;

    /// module label
    std::string moduleLabel_;

    /*
    // user provides: true: L1 Names (vstring), or false: L1 Types (vuint32)
    // bool byName_;
    // disabled: user must always provide names, never indices
    */

    /// list of required L1 triggers by L1 name
    std::vector<std::string > L1SeedsByName_;
    /// list of required L1 triggers by L1 type
    std::vector<unsigned int> L1SeedsByType_;

};

#endif //HLTLevel1Seed_h
