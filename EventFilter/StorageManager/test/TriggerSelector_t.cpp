// $Id: TriggerSelector_t.cpp,v 1.1 2009/12/01 13:58:09 mommsen Exp $

#include "EventFilter/StorageManager/interface/TriggerSelector.h"
#include <iostream>

using namespace stor;
using namespace std;

int main()
{

  std::vector<std::string> fl;
  fl.push_back( "DiMuon" );
  fl.push_back( "CalibPath" );
  fl.push_back( "DiElectron" );
  fl.push_back( "HighPT" );

  std::string f2 = "(* || 2) || (DixMuo* && (!Di* || CalibPath))";

  std::vector<std::string> triggerList;
  triggerList.push_back("DiMuon");
  triggerList.push_back("CalibPath");
  triggerList.push_back("DiElectron");
  triggerList.push_back("HighPT");

  boost::shared_ptr<TriggerSelector> triggerSelector;

  triggerSelector.reset(new TriggerSelector(f2,triggerList));

  edm::HLTGlobalStatus tr(4);
 
  edm::HLTPathStatus pass = edm::hlt::Pass;
  edm::HLTPathStatus fail = edm::hlt::Fail;

  tr[0] = pass;
  tr[1] = pass;
  tr[2] = fail;
  tr[3] = fail;

  std::cout << "RESULT: " << triggerSelector->returnStatus(tr) << std::endl;

  std::cout << "\nend of test\n";
  return 0;

}
