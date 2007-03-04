#ifndef Framework_ProvenanceAccess_h
#define Framework_ProvenanceAccess_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProvenanceAccess
// 
/**\class ProvenanceAccess ProvenanceAccess.h FWCore/Framework/interface/ProvenanceAccess.h

 Description: Helper interface used to access Provenance

 Usage:
    Provides a 'functional' access to the Provenance.  This is used to allow 'delayed' the calculation of the
provenance, which is needed for the unscheduled case.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Feb 15 11:37:49 IST 2006
// $Id: ProvenanceAccess.h,v 1.2 2006/07/06 19:11:42 wmtan Exp $
//

// system include files

// user include files
#include "DataFormats/Provenance/interface/Provenance.h"

// forward declarations
namespace edm {
   class EventProvenanceFiller;
class ProvenanceAccess
{

   public:
      ProvenanceAccess(Provenance*,
                       const EventProvenanceFiller*);
      //virtual ~ProvenanceAccess();

      // ---------- const member functions ---------------------
      ///the routine matches the data of the Provenance class
      const Provenance& provenance() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      //ProvenanceAccess(const ProvenanceAccess&); // stop default

      //const ProvenanceAccess& operator=(const ProvenanceAccess&); // stop default

      // ---------- member data --------------------------------
      Provenance* prov_;
      const EventProvenanceFiller* filler_;
      bool haveFilled_;
};

}

#endif
