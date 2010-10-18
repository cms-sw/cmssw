#ifndef UETrigger_h
#define UETrigger_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TFile.h>

#include <TH1F.h>
#include <TH2D.h>
#include <TLorentzVector.h>
#include <TProfile.h>

#include <TClonesArray.h>

///
///_________________________________________________________
///
class UETriggerHistograms {
  
  public :
    
  UETriggerHistograms( const char*, std::string* );
  ~UETriggerHistograms() { file->Write(); file->Close(); }; 
  
  void fill( TClonesArray&,Double_t);
  
  private :
    
  TFile* file;
  
  std::string HLTBitNames[12];
  TH1D* h_triggerAccepts;
  TH1D* h_eventScale;  
};

#endif
