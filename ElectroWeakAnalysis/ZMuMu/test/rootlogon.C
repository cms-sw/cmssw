{
  gSystem->CompileMacro("setTDRStyle.C", "k");
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  setTDRStyle();
}
