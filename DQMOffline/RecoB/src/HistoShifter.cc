#include "DQMOffline/RecoB/interface/HistoShifter.h"

#include "TH1F.h"

bool HistoShifter::insertAndShift(TH1F* in, const float& value){
  const int& nBins = in->GetNbinsX();
  
  for (int i=2; i<nBins; ++i){
    in->SetBinContent(i-1,in->GetBinContent(i));
    in->SetBinError(i-1,in->GetBinError(i));
  }
  in->SetBinContent(nBins,value);
    
  return true;
}

bool HistoShifter::insertAndShift(TH1F* in, const float& value, const float& error){
  const bool& ok = insertAndShift(in, value);
  in->SetBinError(in->GetNbinsX(),error);
  return ok;
}
