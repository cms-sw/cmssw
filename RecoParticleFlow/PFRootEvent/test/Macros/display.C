{
// initialize the core of the framework, and load the PFRootEvent 
// library, which contains the ROOT interface
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libRecoParticleFlowPFRootEvent.so");
// // AutoLibraryLoader::enable();
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

// create a PFRootEventManager
PFRootEventManager em("pfRootEvent.opt");
// create a JetPFRootEventManager to make FWLiteJets
//JetPFRootEventManager em("pfRootEvent.opt");
// display first entry
int i=0;
em.display(i++);

// look for ECAL rechit with maximum energy
em.lookForMaxRecHit(true);
}
