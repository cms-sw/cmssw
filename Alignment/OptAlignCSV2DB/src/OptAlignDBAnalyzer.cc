
#include "../interface/OptAlignDBAnalyzer.h"

#include <string>
#include <iostream>
#include <vector>

#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h"

void OptAlignDBAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
{
  using namespace edm::eventsetup;
  edm::ESHandle<OpticalAlignments> pObjs;
  context.get<OpticalAlignmentsRcd>().get(pObjs);
  const OpticalAlignments* myobj=pObjs.product();
  std::vector<OpticalAlignInfo>::const_iterator it;
  for( it=myobj->opticalAlignments_.begin();it!=myobj->opticalAlignments_.end(); ++it ){
    std::cout<<"@@@@@ OpticalAlignInfo READ "<< *it << std::endl;
  }
}


