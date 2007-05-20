RunBadStripsFromDBNoise(char* input, char* output)
{
  gROOT->ProcessLine(".L /analysis/sw/CRAB/macros/BadStripsFromDBNoise.C+");
  gSystem->Load("/analysis/sw/CRAB/macros/BadStripsFromDBNoise_C.so");
  BadStripsFromDBNoise(input,output,7,7,1,12);
}
