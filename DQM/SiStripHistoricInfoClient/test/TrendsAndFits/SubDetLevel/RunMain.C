RunMain(string filename, char* outputFile)
{
  gROOT->ProcessLine(".L SubDetTree.cpp++");
  gSystem->Load("SubDetTree_cpp.so");
  SubDetTree(filename,outputFile);
}
