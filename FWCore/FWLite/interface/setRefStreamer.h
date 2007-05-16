#ifndef FWCore_FWLite_setRefStreamer_h
#define FWCore_FWLite_setRefStreamer_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     setRefStreamer
// 
/**\class setRefStreamer setRefStreamer.h FWCore/FWLite/interface/setRefStreamer.h

 Description: Allows one to set the EDProductGetter used by the Ref's and returns the old getter

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue May 15 11:25:51 EDT 2007
// $Id$
//

// system include files

// user include files

// forward declarations
namespace edm {
  class EDProductGetter;
}

namespace fwlite {
  edm::EDProductGetter const* setRefStreamer(edm::EDProductGetter const* ep);
  
  class GetterOperate {
public:
    GetterOperate( edm::EDProductGetter const* iEP): old_(0) {
      old_ = setRefStreamer(iEP);
    }
    ~GetterOperate() {
      setRefStreamer(old_);
    }
private:
    edm::EDProductGetter const* old_;
  };
}


#endif
