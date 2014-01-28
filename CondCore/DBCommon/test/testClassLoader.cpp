#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "CondCore/DBCommon/interface/ClassInfoLoader.h"
#include "StorageSvc/DbReflex.h"
#include <string>

#include <iostream>


std::string const tokens[] = {
  "[DB=00000000-0000-0000-0000-000000000000][CNT=CSCPedestalsRcd][CLID=E1D4BE86-63E6-21C8-41B3-1116BCFBDE24][TECH=00000B01][OID=00000004-00000003]",
    "[DB=00000000-0000-0000-0000-000000000000][CNT=Fake][CLID=E1D4BE86-0000-21C8-41B3-1116BCFBDE24][TECH=00000B01][OID=00000004-00000003]",
"[DB=00000000-0000-0000-0000-000000000000][CNT=CSCPedestalsRcd][CLID=E1D4BE86-63E6-21C8-41B3-1116BCFBDE24][TECH=00000B01][OID=00000004-00000005]",
"[DB=00000000-0000-0000-0000-000000000000][CNT=Alignments][CLID=2F16F0A9-79D5-4881-CE0B-C271DD84A7F3][TECH=00000B01][OID=00000004-00000000]",
"[DB=00000000-0000-0000-0000-000000000000][CNT=DSW<Pedestals>][CLID=917E2774-69AD-833A-8007-C8568FA6EC70][TECH=00000B01][OID=00000004-00000000]"    
};

size_t N=5;

int main() {
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());


  //edm::AssertHandler ah;
  
  for (size_t i=0; i<N; i++) {
    std::string const & token = tokens[i];
    try {
      ROOT::Reflex::Type type = cond::reflexTypeByToken(token);
      std::cout << "class " << type.Name(ROOT::Reflex::SCOPED)
		<< " for " << token << std::endl;
    } catch (cms::Exception const & e) {
      std::cout << e.what() << std::endl;
    }
  }

  return 0;

}
