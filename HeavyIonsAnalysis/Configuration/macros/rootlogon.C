
{
  gSystem->Load( "libFWCoreFWLite" );
  gSystem->Load("libDataFormatsFWLite");
  FWLiteEnabler::enable();
 
  //open dummy file for automatic loading of necessary libraries
  new TFile("./hiCommonSkimAOD.root");

}




