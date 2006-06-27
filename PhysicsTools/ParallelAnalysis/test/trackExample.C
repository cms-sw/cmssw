{
   gSystem->Load("libFWCoreFWLite");
   AutoLibraryLoader::enable();
   TChain events("Events");
   events.Add("file:/afs/cern.ch/cms/Releases/CMSSW/CMSSW_0_7_2/src/Configuration/Applications/data/reco-application-tracking-finaltrackfits-ctffinalfitanalytical.root");
   gSystem->Load( "libPhysicsToolsParallelAnalysis" );
   TrackTSelector * selector = new TrackTSelector;
   events.Process( selector );
}
