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
// $Id: EveService.h,v 1.5 2010/07/15 13:02:02 matevz Exp $
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
class TEveElement;
class TEveMagField;
class TEveTrackPropagator;
class TRint;

class TGTextButton;
class TGLabel;

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

   void display(const std::string& info="");

   TEveManager*  getManager();
   TEveMagField* getMagField();
   void          setupFieldForPropagator(TEveTrackPropagator* prop);

   // Shortcuts for adding top level event and geometry elements.
   void AddElement(TEveElement* el);
   void AddGlobalElement(TEveElement* el);

   // GUI slots -- must be public so that ROOT can call them via CINT.

   void slotExit();
   void slotNextEvent();
   void slotStep();
   void slotContinue();

protected:
   void createEventNavigationGUI();

private:
   EveService(const EveService&);                  // stop default
   const EveService& operator=(const EveService&); // stop default

   // ---------- member data --------------------------------

   TEveManager  *m_EveManager;
   TRint        *m_Rint;

   TEveMagField *m_MagField;

   bool          m_AllowStep;
   bool          m_ShowEvent;

   TGTextButton *m_ContinueButton;
   TGTextButton *m_StepButton;
   TGLabel      *m_StepLabel;

   ClassDef(EveService, 0);
};

#endif
