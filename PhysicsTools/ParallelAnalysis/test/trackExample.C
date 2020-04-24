{
/// enable automatic data formats librarly loading
   gSystem->Load("libFWCoreFWLite");
   FWLiteEnabler::enable();
/// set up events chain
   TChain events("Events");
   events.Add("aod.root");
/// load TSelector library
   gSystem->Load( "libPhysicsToolsParallelAnalysis" );
/// create actual selector object
   TSelector * selector = new examples::TrackTSelector;
/// process chain
   events.Process( selector );
}
