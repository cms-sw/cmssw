#ifndef Fireworks_Core_CSGContinuousAction_h
#define Fireworks_Core_CSGContinuousAction_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGContinuousAction
//
/**\class CSGContinuousAction CSGContinuousAction.h Fireworks/Core/interface/CSGContinuousAction.h

   Description: An action which continues over time (e.g. playing events)

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 29 10:19:42 EDT 2008
// $Id: CSGContinuousAction.h,v 1.7 2009/08/26 18:59:20 amraktad Exp $
//

// system include files
#include <string>

// user include files
#include "Fireworks/Core/interface/CSGAction.h"

// forward declarations

class CSGContinuousAction : public CSGAction {

public:
   CSGContinuousAction(CSGActionSupervisor *sup, const char *name);
   //virtual ~CSGContinuousAction();

   // ---------- const member functions ---------------------
   bool isRunning() const { return m_isRunning; }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void createCustomIconsButton(TGCompositeFrame* p,
                                const TGPicture* upPic,
                                const TGPicture* downPic,
                                const TGPicture* disabledPic,
                                const TGPicture* upRunningPic,
                                const TGPicture* downRunningPic,
                                TGLayoutHints* l = 0,
                                Int_t id = -1,
                                GContext_t norm = TGButton::GetDefaultGC() (),
                                UInt_t option = 0);
   void stop();

   sigc::signal<void> started_;
   sigc::signal<void> stopped_;

   //override
   virtual void globalEnable();
   virtual void globalDisable();

   void switchMode();

private:
   CSGContinuousAction(const CSGContinuousAction&); // stop default

   const CSGContinuousAction& operator=(const CSGContinuousAction&); // stop default

   // ---------- member data --------------------------------
   std::string m_imageFileName;
   std::string m_runningImageFileName;
   //const TGPicture* m_runningImage;
   const TGPicture* m_upPic;
   const TGPicture* m_downPic;
   const TGPicture* m_disabledPic;
   const TGPicture* m_runningUpPic;
   const TGPicture* m_runningDownPic;

   FWCustomIconsButton* m_button;

   bool m_isRunning;


};


#endif
