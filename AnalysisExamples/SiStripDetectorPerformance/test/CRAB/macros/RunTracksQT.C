RunTracksQT(char* input, char* output)
{
  gROOT->ProcessLine(".L /analysis/sw/CRAB/macros/TracksQT.C+");  
  gSystem->Load("/analysis/sw/CRAB/macros/TracksQT_C.so");
  TracksQT(input,output);
}
