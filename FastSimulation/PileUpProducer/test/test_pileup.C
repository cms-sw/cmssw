#include "DataFormats/FWLite/interface/Handle.h"

/*
Don't forget these commands, if they are not in your rootlogon.C:

   gSystem->Load("libFWCoreFWLite.so"); 
   AutoLibraryLoader::enable();
   gSystem->Load("libDataFormatsFWLite.so");

*/

void test_pileup()
{
  TFile file("TTbar_Tauola_7TeV_cfi_py_GEN_FASTSIM_HLT_PU.root");
  
  fwlite::Event ev(&file);

  for( ev.toBegin(); ! ev.atEnd(); ++ev) {
    fwlite::Handle< PileupMixingContent > pmc;
    pmc.getByLabel(ev,"famosPileUp");
    // now can access data
    std::cout <<" bunch crossing "<<pmc.ptr()->getMix_bunchCrossing().at(0)<<std::endl;
    std::cout <<" interaction number  "<<pmc.ptr()->getMix_Ninteractions().at(0)<<std::endl;
  }
}
