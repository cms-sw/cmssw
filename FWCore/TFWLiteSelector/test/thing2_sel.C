{
gSystem->Load("libFWCoreFWLite");
AutoLibraryLoader::enable();
gSystem->Load("libFWCoreTFWLiteSelectorTest");
TSelector* sel = new tfwliteselectortest::ThingsTSelector2();
TChain c("Events");
c.Add("test.root");
c.Process(sel);
}
