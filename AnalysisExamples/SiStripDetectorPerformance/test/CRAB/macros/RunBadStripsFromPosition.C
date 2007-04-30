RunBadStripsFromPosition(char* input, char* output)
{
  gSystem->Load("/analysis/sw/CRAB/macros/BadStripsFromPosition_C.so");
  BadStripsFromPosition(input,output,1.e-7);
}
