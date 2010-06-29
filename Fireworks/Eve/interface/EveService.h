#ifndef Fireworks_Geometry_EveService_h
#define Fireworks_Geometry_EveService_h
// -*- C++ -*-
//
// Package:     Fireworks/Eve
// Class  :     EveService
// 
/**\class EveService EveService.h Fireworks/Geometry/interface/EveService.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Matevz Tadel
//         Created:  Fri Jun 25 18:56:52 CEST 2010
// $Id$
//

#include <string>

namespace edm
{
   class ParameterSet;
   class ActivityRegistry;
   class Event;
   class EventSetup;
}

class TEveManager;
class TRint;

class EveService
{
public:
   EveService(const edm::ParameterSet&, edm::ActivityRegistry&);
   virtual ~EveService();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   void postBeginJob();
   void postEndJob();

   void postProcessEvent(const edm::Event&, const edm::EventSetup&);

   TEveManager* getManager();

private:
   EveService(const EveService&);                  // stop default
   const EveService& operator=(const EveService&); // stop default

   // ---------- member data --------------------------------

   TEveManager  *m_EveManager;
   TRint        *m_Rint;
};

#endif
