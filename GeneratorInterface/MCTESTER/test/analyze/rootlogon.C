{
  gSystem.Load("libHist.so");
  gSystem.Load("libGpad.so");
  
  gSystem.Load("$MCTESTERLOCATION/libHEPEvent.so");
  gSystem.Load("$MCTESTERLOCATION/libHepMCEvent.so");
  gSystem.Load("$MCTESTERLOCATION/libMCTester.so");
  
  gROOT->SetStyle("Plain");
}
