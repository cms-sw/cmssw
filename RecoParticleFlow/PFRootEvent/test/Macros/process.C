{
// initialize the core of the framework, and load the PFRootEvent 
// library, which contains the ROOT interface

gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libRecoParticleFlowPFRootEvent.so");
AutoLibraryLoader::enable();
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

// create a PFRootEventManager
PFRootEventManagerColin em("pfRootEvent.opt");

if(em.tree_) {
  int n =  em.tree_->GetEntries();
  // int n = 2;
  for(unsigned i=0; i<n; i++) {
    em.processEntry(i);
    // em.print();
  }
  //em.processEntry(1126);
  em.write();
}
gApplication->Terminate(); 
}
