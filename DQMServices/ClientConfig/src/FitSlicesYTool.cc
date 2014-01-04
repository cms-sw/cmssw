#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TDirectory.h"

using namespace std;

FitSlicesYTool::FitSlicesYTool(MonitorElement* me)
{ 
  const bool oldAddDir = TH1::AddDirectoryStatus();
  TH1::AddDirectory(true);
  // ... create your hists
  TH2F * h = me->getTH2F();
  h->FitSlicesY();
  string name(h->GetName());
  h2D = (TH2F*)h->Clone();
  h0 = (TH1*)gDirectory->Get((name+"_0").c_str());
  h1 = (TH1*)gDirectory->Get((name+"_1").c_str());
  h2 = (TH1*)gDirectory->Get((name+"_2").c_str());
  h3 = (TH1*)gDirectory->Get((name+"_chi2").c_str());
  TH1::AddDirectory(oldAddDir);
}

// FitSlicesYTool::FitSlicesYTool(TH2F* h){
//   h->FitSlicesY();
//   string name(h->GetName());
//   h0 = (TH1*)gDirectory->Get((name+"_0").c_str());
//   h1 = (TH1*)gDirectory->Get((name+"_1").c_str());
//   h2 = (TH1*)gDirectory->Get((name+"_2").c_str());
//   h3 = (TH1*)gDirectory->Get((name+"_chi2").c_str());
// }
FitSlicesYTool::~FitSlicesYTool(){
  delete h2D;
  delete h0;  
  delete h1;  
  delete h2;  
  delete h3;  
}
void FitSlicesYTool::getFittedMean(MonitorElement * me){
  if (!(h1&&me)) throw cms::Exception("FitSlicesYTool") << "Pointer =0 : h1=" << h1 << " me=" << me;
  if (h1->GetNbinsX()==me->getNbinsX()){
    for (int bin=0;bin!=h1->GetNbinsX();bin++){
      me->setBinContent(bin+1,h1->GetBinContent(bin+1));
//       me->setBinEntries(bin+1, 1.);
    }
  } else {
    throw cms::Exception("FitSlicesYTool") << "Different number of bins!";
  }
}
void FitSlicesYTool::getFittedSigma(MonitorElement * me){
  if (!(h2&&me)) throw cms::Exception("FitSlicesYTool") << "Pointer =0 : h1=" << h1 << " me=" << me;
  if (h2->GetNbinsX()==me->getNbinsX()){
    for (int bin=0;bin!=h2->GetNbinsX();bin++){
      me->setBinContent(bin+1,h2->GetBinContent(bin+1));
//       me->setBinEntries(bin+1, 1.);
    }
  } else {
    throw cms::Exception("FitSlicesYTool") << "Different number of bins!";
  }
}
void FitSlicesYTool::getFittedMeanWithError(MonitorElement * me){
  if (!(h1&&me)) throw cms::Exception("FitSlicesYTool") << "Pointer =0 : h1=" << h1 << " me=" << me;
  if (h1->GetNbinsX()==me->getNbinsX()){
    for (int bin=0;bin!=h1->GetNbinsX();bin++){
      me->setBinContent(bin+1,h1->GetBinContent(bin+1));
//       me->setBinEntries(bin+1, 1.);
      me->setBinError(bin+1,h1->GetBinError(bin+1));
    }
  } else {
    throw cms::Exception("FitSlicesYTool") << "Different number of bins!";
  }
}
void FitSlicesYTool::getFittedSigmaWithError(MonitorElement * me){
  if (!(h2&&me)) throw cms::Exception("FitSlicesYTool") << "Pointer =0 : h1=" << h1 << " me=" << me;
  if (h2->GetNbinsX()==me->getNbinsX()){
    for (int bin=0;bin!=h2->GetNbinsX();bin++){
      me->setBinContent(bin+1,h2->GetBinContent(bin+1));
//       me->setBinEntries(bin+1, 1.);
      me->setBinError(bin+1,h2->GetBinError(bin+1));
    }
  } else {
    throw cms::Exception("FitSlicesYTool") << "Different number of bins!";
  }
}
void FitSlicesYTool::getRMS(MonitorElement * me){
  if (!(h2D&&me)) throw cms::Exception("FitSlicesYTool") << "Pointer =0 : h2D=" << h2D << " me=" << me;
  if (h2D->GetNbinsX()==me->getNbinsX()){
    for (int bin=1;bin!=h2D->GetNbinsX();bin++){
      TH1D * tmp = h2D->ProjectionY(" ", bin, bin);
      double rms = tmp->GetRMS();
      tmp->Delete();
      me->setBinContent(bin,rms);
    }
  } else {
    throw cms::Exception("FitSlicesYTool") << "Different number of bins!";
  }
}
