#include "OHltMenu.h"

using namespace std;

OHltMenu::OHltMenu() {
};


void OHltMenu::SetMapL1SeedsOfStandardHLTPath(std::map<TString, std::vector<TString> > smap) {
  map_L1SeedsOfStandardHLTPath = smap;
  //std::cout<<map_L1SeedsOfStandardHLTPath.size()<<std::endl;
}

void OHltMenu::AddTrigger(TString trign, int presc, float eventS) {
  names.push_back(trign);
  prescales[trign] 	       	= presc;
  eventSizes[trign] 	       	= eventS;
}

void OHltMenu::AddTrigger(TString trign, TString seedcond, int presc, float eventS) {
  names.push_back(trign);
  seedcondition[trign] 	       	= seedcond;
  prescales[trign] 	       	= presc;
  eventSizes[trign] 	       	= eventS;
}

void OHltMenu::AddL1forPreLoop(TString trign, int presc) {
  L1names.push_back(trign);
  L1prescales[trign] 	       	= presc;
}

void OHltMenu::print() {
  cout << "Menu - isL1Menu="<<isL1Menu << " - doL1preloop="<<doL1preloop<<  endl;
  cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<  endl;
  for (unsigned int i=0;i<names.size();i++) {
    cout<<names[i]<<" \""<<seedcondition[names[i]]<<"\" "<<prescales[names[i]]<<" "<<eventSizes[names[i]]<<" "<<endl;
  }
  cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<  endl;

  if (doL1preloop) {
    cout << endl << "L1 Menu - for L1preloop"<<  endl;
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<  endl;
    for (unsigned int i=0;i<L1names.size();i++) {
      cout<<L1names[i]<<" "<<L1prescales[L1names[i]]<<endl;
    }
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" <<  endl;
  }
  cout << endl;
}

