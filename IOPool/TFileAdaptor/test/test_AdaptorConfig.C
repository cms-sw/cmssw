void test_AdaptorConfig(const char * fname) {
  gSystem->Load("pluginIOPoolTFileAdaptor");
  TFileAdaptorUI aui;
  gROOT->GetPluginManager()->Print();
  TFile * p = TFile::Open(fname);
  p->Map();
  aui.stats();
}
