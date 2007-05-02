{
// initialize the core of the framework, and load the PFRootEvent 
// library, which contains the ROOT interface

gROOT->Macro("init.C");

// create a PFRootEventManager
MyPFRootEventManager em("pfRootEvent.opt");

if(em.tree() ) {
  int n =  em.tree()->GetEntries();

  for(unsigned i=0; i<n; i++) {
    em.processEntry(i);
    // em.print();
  }
  em.write();
}
gApplication->Terminate(); 
}
