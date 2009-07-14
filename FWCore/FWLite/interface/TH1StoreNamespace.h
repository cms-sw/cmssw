// -*- C++ -*-

#if !defined(TH1StoreNamespace_H)
#define TH1StoreNamespace_H

#include "FWCore/FWLite/interface/TH1Store.h"

namespace th1store
{

   // variable declaration
   extern TH1Store ns_store;

   void  add   (TH1 *histPtr);
   TH1  *hist  (const std::string &name);
   void  write (const std::string &name);
   void  write (TFile *filePtr);

}

#endif // TH1StoreNamespace_H
