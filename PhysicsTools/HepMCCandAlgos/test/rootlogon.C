{
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  cout << "Opening Gen+Candidate file" << endl;
  TFile f("gen.root");
}
