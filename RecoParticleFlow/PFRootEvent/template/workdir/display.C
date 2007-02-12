{
// initialize the core of the framework, and load the PFRootEvent 
// library, which contains the ROOT interface

gROOT->Macro("init.C");

// create a PFRootEventManager
// PFRootEventManager em("pfRootEvent.opt");
MyPFRootEventManager em("pfRootEvent.opt");

// display first entry
int i=0;
em.display(i++);

// look for ECAL rechit with maximum energy
em.lookForMaxRecHit(true);
}
