{

gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libDemoPFRootEvent.so");
AutoLibraryLoader::enable();
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

// PFRootEventManager em("pfRootEvent.opt");
// em.ProcessEntry(3);
}
