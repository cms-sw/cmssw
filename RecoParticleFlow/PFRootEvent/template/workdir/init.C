{
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libRecoParticleFlowMyPFRootEvent.so");
AutoLibraryLoader::enable();
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();
}
