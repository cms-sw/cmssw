// -*- C++ -*-

#include "FWCore/FWLite/interface/TH1StoreNamespace.h"

using namespace std;

TH1Store th1store::ns_store;

void 
th1store::add (TH1 *histPtr) 
{ 
   ns_store.add (histPtr); 
}

TH1 *
th1store::hist (const string &name)
{ 
   return ns_store.hist (name); 
}

void 
th1store::write (const string &name)
{ 
   ns_store.write (name); 
}

void 
th1store::write (TFile *filePtr)
{ 
   ns_store.write (filePtr); 
}
