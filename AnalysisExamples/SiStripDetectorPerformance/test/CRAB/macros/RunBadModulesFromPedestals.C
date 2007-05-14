RunBadModulesFromPedestals(char* input, char* output)
{
  gSystem->Load("/analysis/sw/CRAB/macros/BadModulesFromPedestals_C.so");
  BadModulesFromPedestals(input,output,768);
}
