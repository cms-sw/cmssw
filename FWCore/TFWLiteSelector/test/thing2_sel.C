{
  //Need this to allow ROOT to be able to use a ThingsTSelector
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();
  gSystem->Load("libFWCoreTFWLiteSelectorTest");
  
  TSelector* sel = new tfwliteselectortest::ThingsTSelector2();
  
  //This holds the list of files and 'TTree' to process
  TChain c("Events");
  c.Add("testTFWLiteSelector.root");
  
  //This actually processes the data
  c.Process(sel);
}
