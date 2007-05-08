RunBadStripsFromDBNoise(char* input, char* output)
{
  gSystem->Load("/analysis/sw/CRAB/macros/BadStripsFromDBNoise_C.so");
  BadStripsFromDBNoise(input,output,3,5,1,10);
}
