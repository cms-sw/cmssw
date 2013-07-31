#ifndef __subsys_____pkgname_____class___h
#define __subsys_____pkgname_____class___h
// -*- C++ -*-
//
// Package:    __subsys__/__pkgname__
// Class:      __class__
// 
/**\class __class__ __class__.h __subsys__/__pkgname__/plugins/__class__.h

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  __author__
//         Created:  __date__
//
//
#include <TH1.h>
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelector.h"

//A worker processes the events.  When using PROOF there is one Worker per PROOF CPU Node.
struct __class__Worker {
  __class__Worker(const TList*, TList&);
  ~__class__Worker();
  void process( const edm::Event& iEvent );
  void postProcess(TList&);
  //Place histograms, etc that you want to fill here
  //TH1F* h_a;
@example_track  TH1F* h_pt;
};

//Only one Selector is made per job. It gets all the results from each worker.
class __class__ : public TFWLiteSelector<__class__Worker> {
public :
  __class__();
  ~__class__();
  void begin(TList*&);
  void terminate(TList&);
    
private:
    
  __class__(__class__ const&);
  __class__ operator=(__class__ const&);
  
  ClassDef(__class__,2)
};
#endif
