{
  //This make the TSelectors in the library available to the remote proof session
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  gSystem->Load("libFWCoreTFWLiteSelectorTest");
}
