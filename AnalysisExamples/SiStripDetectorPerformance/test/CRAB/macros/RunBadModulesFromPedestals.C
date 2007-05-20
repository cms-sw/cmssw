RunBadModulesFromPedestals(char* input, char* output)
{
  gROOT->ProcessLine(".L /analysis/sw/CRAB/macros/BadModulesFromPedestals.C+");
  gSystem->Load("/analysis/sw/CRAB/macros/BadModulesFromPedestals_C.so");
  BadModulesFromPedestals(input,output,768);
}
