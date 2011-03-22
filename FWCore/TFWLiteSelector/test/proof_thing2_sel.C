{
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();

  if (gSystem->Getenv("tmpdir")) {
    gEnv->SetValue("Proof.Sandbox", "$tmpdir/proof");
  }

  //Setup the proof server
  TProof *gProof=TProof::Open( "" );
  
  // This makes sure the TSelector library and dictionary are properly
  // installed in the remote PROOF servers

  // This works, but results in an annoying error message from 'cp',
  // something not right with the how the macro is sent?
  //gProof->Exec( ".x proof_remote.C" );

  // So inline it...
  gProof->Exec("gSystem->Load(\"libFWCoreFWLite\"); "
               "AutoLibraryLoader::enable(); "
  // Have to load library manually since Proof does not use the 
  // mechanism used by TFile to find class dictionaries and therefore
  // the AutoLibraryLoader can not help
               "gSystem->Load(\"libFWCoreTFWLiteSelectorTest\");");
  
  //This creates the 'data set' which defines what files we need to process
  // NOTE: the files given must be accessible by the remote systems
  TDSet c( "TTree", "Events");
  c.Add("$CMSSW_BASE/test.root");
  
  //This makes the actual processing happen
  c.Process( "tfwliteselectortest::ThingsTSelector2" );
}
