RunPlotMacro(char* inputFile, char* outputFile)
{
  gROOT->ProcessLine(".L PlotMacro.cpp++");
  gSystem->Load("PlotMacro_cpp.so");
  PlotMacro(inputFile,outputFile);
}
