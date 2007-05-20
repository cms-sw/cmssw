RunBadStripsFromPosition(char* input, char* output)
{
  gROOT->ProcessLine(".L /analysis/sw/CRAB/macros/BadStripsFromPosition.C+");
  gSystem->Load("/analysis/sw/CRAB/macros/BadStripsFromPosition_C.so");
  BadStripsFromPosition(input,output,1.e-7);
}
