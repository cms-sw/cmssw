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
// $Id: EveService.h,v 1.3 2010/07/08 19:43:44 matevz Exp $
//

#include <string>
#include <Rtypes.h>

namespace edm
{
   class ParameterSet;
   class ActivityRegistry;
   class Run;
   class Event;
   class EventSetup;
}

class TEveManager;
class TEveMagField;
class TEveTrackPropagator;
class TRint;

class TGTextButton;

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

   void postBeginRun(const edm::Run&, const edm::EventSetup&);

   void postProcessEvent(const edm::Event&, const edm::EventSetup&);

   void display();

   TEveManager*  getManager();
   TEveMagField* getMagField();
   void          setupFieldForPropagator(TEveTrackPropagator* prop);

   // GUI slots -- must be public so that ROOT can call them via CINT.

   void slotNextEvent();
   void slotExit();

protected:
   void createEventNavigationGUI();

private:
   EveService(const EveService&);                  // stop default
   const EveService& operator=(const EveService&); // stop default

   // ---------- member data --------------------------------

   TEveManager  *m_EveManager;
   TRint        *m_Rint;

   TEveMagField *m_MagField;

   TGTextButton *m_NextButton;

   ClassDef(EveService, 0);
};

#endif
