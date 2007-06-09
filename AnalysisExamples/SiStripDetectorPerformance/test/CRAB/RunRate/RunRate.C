RunRate(char* input, bool logy, bool cumulate)
{
  gStyle->SetLabelSize(0.05,"x");
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetOptDate(21);

  gROOT->ProcessLine(".L /analysis/sw/CRAB/RunRate/CreatePlots.C+");
  gSystem->Load("/analysis/sw/CRAB/RunRate/CreatePlots_C.so");
  CreatePlots(input,"RunNb","Nentries",logy,cumulate);
}

