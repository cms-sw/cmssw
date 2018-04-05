{
  //This make the TSelectors in the library available to the remote proof session
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();
  gSystem->Load("libFWCoreTFWLiteSelectorTest");
}
