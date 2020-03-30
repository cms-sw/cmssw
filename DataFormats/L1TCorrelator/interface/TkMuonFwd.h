#ifndef TkTrigger_L1MuonFwd_h
#define TkTrigger_L1MuonFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkMuonFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"


namespace l1t {

   class TkMuon ;

   typedef std::vector< TkMuon > TkMuonCollection ;

   typedef edm::Ref< TkMuonCollection > TkMuonRef ;
   typedef edm::RefVector< TkMuonCollection > TkMuonRefVector ;
   typedef std::vector< TkMuonRef > TkMuonVectorRef ;
}

#endif


