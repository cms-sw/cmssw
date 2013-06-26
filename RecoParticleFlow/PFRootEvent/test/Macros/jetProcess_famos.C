{
// initialize the core of the framework, and load the PFRootEvent 
// library, which contains the ROOT interface
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libRecoParticleFlowPFRootEvent.so");
// AutoLibraryLoader::enable();
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

// create a PFRootEventManager
PFRootEventManager em("jet_famos.opt");

if(em.tree() ) {
	int n =  em.tree()->GetEntries();
	for(unsigned i=0; i<n; i++) {
		em.processEntry(i);
		//em.print();
	}
	em.write();
}
gApplication->Terminate(); 
}
