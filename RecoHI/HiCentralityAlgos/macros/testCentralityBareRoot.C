#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "TFile.h"
#include <iostream>
using namespace std;

void testCentralityBareRoot(){

  TFile* inf = new TFile("centralityfile.root");

  CentralityBins::RunMap cmap = getCentralityFromFile(inf, "makeCentralityTableTFile", "HFhitsAMPT_2760GeV", 0, 2);
  //  or, alternatively:
  //  TDirectoryFile* dir = (TDirectoryFile*) inf->Get("makeCentralityTableTFile");
  //  CentralityBins::RunMap cmap = getCentralityFromFile(dir, "HFhitsAMPT_2760GeV", 0, 2);

  cout<<"map size "<<cmap.size()<<endl;
  int bin = (cmap[1])->getBin(23000);
  cout<<bin<<endl;

}
