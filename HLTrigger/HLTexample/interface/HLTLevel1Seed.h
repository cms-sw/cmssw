#ifndef HLTLevel1Seed_h
#define HLTLevel1Seed_h

/** \class HLTLevel1Seed
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing filtering on
 *  Level-1 bits and extraction of seed objects.
 *
 *  $Date: 2006/06/26 23:39:24 $
 *  $Revision: 1.1 $
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

class HLTLevel1Seed : public HLTFilter {

   public:
      explicit HLTLevel1Seed(const edm::ParameterSet&);
      ~HLTLevel1Seed();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      bool                      andOr_;  // false=and-mode (all), true=or-mode(at least one)
      bool                      byName_; // list of l1 triggers provided by Name or not
      std::vector<unsigned int> L1SeedsByType_; // list of required L1 triggers by type/number
      std::vector<std::string > L1SeedsByName_; // list of required L1 triggers by name
};

#endif //HLTLevel1Seed_h
