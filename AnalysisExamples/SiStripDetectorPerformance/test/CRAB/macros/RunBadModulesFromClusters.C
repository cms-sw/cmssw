RunBadModulesFromClusters(char* input, char* output)
{
  gROOT->ProcessLine(".L /analysis/sw/CRAB/macros/BadModulesFromClusters.C+");
  gSystem->Load("/analysis/sw/CRAB/macros/BadModulesFromClusters_C.so");
  BadModulesFromClusters(input,output,14,1,100);
}
