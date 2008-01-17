{
// initialize the core of the framework, and load the PFRootEvent 
// library, which contains the ROOT interface
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libRecoParticleFlowPFRootEvent.so");
// // AutoLibraryLoader::enable();
gSystem->Load("libCintex.so");
ROOT::Cintex::Cintex::Enable();

// create a PFRootEventManager
// PFRootEventManager em("pfRootEvent.opt");
// create a JetPFRootEventManager to make FWLiteJets
string opt = "pfRootEvent.opt";
PFRootEventManager em( opt.c_str() );

//create a DisplayManager
DisplayManager dm(&em, opt.c_str() );

// display first entry
int i=0;
dm.display(i++);

// look for ECAL rechit with maximum energy
dm.lookForMaxRecHit(true);

// create a dialogFrame:
// DialogFrame *win = new DialogFrame(&em,&dm,gClient->GetRoot(), 200,220);

}
