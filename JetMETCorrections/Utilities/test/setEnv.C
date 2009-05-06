void setEnv()
{
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
  gROOT->ProcessLine(".X setDefaultStyle.C");
}
