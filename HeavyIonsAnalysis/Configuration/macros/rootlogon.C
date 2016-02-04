
{
  gSystem->Load( "libFWCoreFWLite" );
  gSystem->Load("libDataFormatsFWLite");
  AutoLibraryLoader::enable();
 
  //open dummy file for automatic loading of necessary libraries
  new TFile("./hiCommonSkimAOD.root");

}




