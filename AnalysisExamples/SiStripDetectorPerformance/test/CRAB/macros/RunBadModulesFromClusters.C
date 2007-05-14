RunBadModulesFromClusters(char* input, char* output)
{
  gSystem->Load("/analysis/sw/CRAB/macros/BadModulesFromClusters_C.so");
  BadModulesFromClusters(input,output,14,1,100);
}
