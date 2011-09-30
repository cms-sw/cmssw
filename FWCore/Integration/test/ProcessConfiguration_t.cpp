#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <assert.h>
#include <iostream>
#include <string>

int main() {
  try {
    edm::ParameterSet dummyPset;
    dummyPset.registerIt();
    edm::ParameterSetID id = dummyPset.id();
    {
      edm::ProcessConfiguration pc1;
      pc1.setParameterSetID(id);
      assert(pc1 == pc1);
    }
    {
      edm::ProcessConfiguration pc1;
      edm::ProcessConfiguration pc2;
      pc1.setParameterSetID(id);
      pc2.setParameterSetID(id);
      assert(pc1 == pc2);
    }
    {
      edm::ProcessConfiguration pc1;
      edm::ProcessConfiguration pc2("reco2", edm::ParameterSetID(), std::string(), std::string());
      edm::ProcessConfiguration pc3("reco3", edm::ParameterSetID(), std::string(), std::string());
      edm::ProcessConfiguration pc4("reco2", edm::ParameterSetID(), std::string(), std::string());
      pc1.setParameterSetID(id);
      pc2.setParameterSetID(id);
      pc3.setParameterSetID(id);
      pc4.setParameterSetID(id);
      edm::ProcessConfigurationID id1 = pc1.id();
      edm::ProcessConfigurationID id2 = pc2.id();
      edm::ProcessConfigurationID id3 = pc3.id();
   
      assert(id1 != id2);
      assert(id2 != id3);
      assert(id3 != id1);
   
      edm::ProcessConfigurationID id4 = pc4.id();
      assert(pc4 == pc2);
      assert (id4 == id2);
    }
    return 0;
  }
  catch(cms::Exception const& e) {
    std::cerr << e.explainSelf() << std::endl;
    return 1;
  }
}
