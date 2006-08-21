#ifndef Framework_EventProvenanceFiller_h
#define Framework_EventProvenanceFiller_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventProvenanceFiller
// 
/**\class EventProvenanceFiller EventProvenanceFiller.h FWCore/Framework/interface/EventProvenanceFiller.h

 Description: Base class used internal to the Framework for filling Event based Provenance

 Usage:
    This class is used during 'unscheduled' execution for the case where a Selector uses the Provenance information
to decide to 'get' a datum from the Event. Under that case, the 'unscheduled' case must run the Producer who makes
such a datum and then see if the provenance information is a match.
    We use an abstract base class in order to avoid unnecessary physical coupling in the EventPrincipal class.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Feb 15 14:54:43 IST 2006
// $Id: EventProvenanceFiller.h,v 1.1 2006/03/05 21:40:25 chrjones Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
class Provenance;
class EventProvenanceFiller
{

   public:
   EventProvenanceFiller() {}
   virtual ~EventProvenanceFiller() {}

      // ---------- const member functions ---------------------
      virtual bool fill(Provenance&) = 0; 
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      EventProvenanceFiller(const EventProvenanceFiller&); // stop default

      const EventProvenanceFiller& operator=(const EventProvenanceFiller&); // stop default

      // ---------- member data --------------------------------

};

}
#endif
