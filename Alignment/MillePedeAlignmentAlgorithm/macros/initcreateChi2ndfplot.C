void initcreateChi2ndfplot(const char *txtFile)
{
  gROOT->ProcessLine(".L createChi2ndfplot.C+g");
  createChi2ndfplot(txtFile);
}
