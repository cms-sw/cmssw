{
  //Setup the connect to the proof server
  gROOT->Proof( "lnx7108.lns.cornell.edu" );
  
  //Need this to allow ROOT to be able to use a ThingsTSelector
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  //Have to load library manually since Proof does not use the 
  // mechanism used by TFile to find class dictionaries and therefore
  // the AutoLibraryLoader can not help
  gSystem->Load("libFWCoreTFWLiteSelectorTest");

  //This makes sure the TSelector library and dictionary are properly
  // installed in the remote PROOF servers
  gProof->Exec( ".x Pthing_sel_Remote.C" );

  //This creates the 'data set' which defines what files we need to process
  // NOTE: the files given must be accessible by the remote systems
  TDSet c( "TTree", "Events");
  c.Add("/home/gregor/cms/CMSSW_0_7_2/src/FWCore/TFWLiteSelector/test/test.root");
  
  //This makes the actual processing happen
  c.Process( "ThingsTSelector" );
}
