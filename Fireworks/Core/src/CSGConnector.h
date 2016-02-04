#ifndef Fireworks_Core_CSGConnector_h
#define Fireworks_Core_CSGConnector_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGConnector
//
/**\class CSGConnector CSGConnector.h Fireworks/Core/interface/CSGConnector.h

   Description: An adapter classes used to connect ROOT signals to a CSGAction

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu May 29 18:16:04 CDT 2008
// $Id: CSGConnector.h,v 1.5 2009/08/26 18:59:20 amraktad Exp $
//

// system include files
#include "TQObject.h"

// user include files

// forward declarations
class CSGAction;
class CSGActionSupervisor;

class CSGConnector : public TQObject {

public:
   CSGConnector(CSGAction *action, CSGActionSupervisor *supervisor) : m_action(action), m_supervisor(supervisor) {
   };
   //virtual ~CSGConnector();

   // ---------- member functions ---------------------------
   void handleMenu(Int_t entry);
   void handleToolBar(Int_t entry);
   ClassDef(CSGConnector,0);

private:
   CSGConnector(const CSGConnector&); // stop default

   const CSGConnector& operator=(const CSGConnector&); // stop default

   // ---------- member data --------------------------------
   CSGAction *m_action;
   CSGActionSupervisor *m_supervisor;

};


#endif
