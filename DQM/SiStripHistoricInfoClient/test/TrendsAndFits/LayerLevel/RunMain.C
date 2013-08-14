RunMain(string filename, char* outputFile)
{
  gROOT->ProcessLine(".L LayerTree.cpp++");
  gSystem->Load("LayerTree_cpp.so");
  LayerTree(filename,outputFile);
}
