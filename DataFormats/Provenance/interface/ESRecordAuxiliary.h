#ifndef DataFormats_Provenance_ESRecordAuxiliary_h
#define DataFormats_Provenance_ESRecordAuxiliary_h
// -*- C++ -*-
//
// Package:     Provenance
// Class  :     ESRecordAuxiliary
// 
/**\class ESRecordAuxiliary ESRecordAuxiliary.h DataFormats/Provenance/interface/ESRecordAuxiliary.h

 Description: Holds information pertinent to a particular stored EventSetup Record

 Usage:
    For internal use during storage only

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Dec  3 16:10:46 CST 2009
// $Id: ESRecordAuxiliary.h,v 1.1 2009/12/16 17:40:01 chrjones Exp $
//

// system include files

// user include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// forward declarations
namespace edm {
  class ESRecordAuxiliary
  {
    
  public:
    ESRecordAuxiliary();
    ESRecordAuxiliary(const edm::EventID&, const edm::Timestamp& );
    //~ESRecordAuxiliary();
    
    // ---------- const member functions ---------------------
    const edm::EventID& eventID() const { return eventID_;}
    const edm::Timestamp& timestamp() const { return timestamp_;}
    

    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    
  private:
    //ESRecordAuxiliary(const ESRecordAuxiliary&); // stop default
    
    //const ESRecordAuxiliary& operator=(const ESRecordAuxiliary&); // stop default
    
    // ---------- member data --------------------------------
    edm::EventID eventID_;
    edm::Timestamp timestamp_;    
  };
}

#endif
