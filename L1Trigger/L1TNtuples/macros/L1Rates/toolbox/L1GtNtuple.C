#define L1GtNtuple_cxx
#include "L1GtNtuple.h"


void L1GtNtuple::Test()
{ 

  if (fChain == 0)  return;
 
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  unsigned int nevents =0;

  std::cout << nentries << " events to process"<<std::endl;
  for (Long64_t jentry=0; jentry<nentries;jentry++)
  {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    
    //fChain->GetEvent(jentry);
  
    nevents++;
    if (nevents<9)  //eight first events
      { 
    std::cout << "--------------------- Event "<<jentry<<" ---------------------"<<std::endl;

    //event_
    std::cout << "L1Tree         : run number = "<<event_->run<<std::endl;

   
    //gt_
    std::cout << "L1Tree         : gttw1.size = "<<gt_->tw1.size()<<std::endl;

    }
  }
   
}


