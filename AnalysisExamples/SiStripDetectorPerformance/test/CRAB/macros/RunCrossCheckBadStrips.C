RunCrossCheckBadStrips(char* input1, char* input2)
{
  gROOT->ProcessLine(".L /analysis/sw/CRAB/macros/CrossCheckBadStrips.C+");
  gSystem->Load("/analysis/sw/CRAB/macros/CrossCheckBadStrips_C.so");
  CrossCheckBadStrips(input1,input2);
}
