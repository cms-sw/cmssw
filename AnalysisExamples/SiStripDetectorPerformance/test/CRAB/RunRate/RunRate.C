RunRate(char* input, bool logy, bool cumulate)
{
  gROOT->ProcessLine(".L /analysis/sw/CRAB/RunRate/CreatePlots.C+");
  gSystem->Load("/analysis/sw/CRAB/RunRate/CreatePlots_C.so");
  CreatePlots(input,"RunNb","Nentries",logy,cumulate);
}

