{

// initialize the core of the framework, and load the PFRootEvent 
// library, which contains the ROOT interface
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libRecoParticleFlowPFRootEvent.so");
// AutoLibraryLoader::enable();
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

}
