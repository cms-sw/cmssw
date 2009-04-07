// -*- C++ -*-
#ifndef Fireworks_Core_FWDisplayEvent_h
#define Fireworks_Core_FWDisplayEvent_h
//
// Package:     Core
// Class  :     FWDisplayEvent
//
/**\class FWDisplayEvent FWDisplayEvent.h Fireworks/Core/interface/FWDisplayEvent.h

   Description: Displays an fwlite::Event in ROOT

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Mon Dec  3 08:34:30 PST 2007
// $Id: FWDisplayEvent.h,v 1.29 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <vector>
#include <string>
#include <memory>
#include <boost/shared_ptr.hpp>
#include "Rtypes.h"

// user include files
#include "DetIdToMatrix.h"

// forward declarations
class TGPictureButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class FWEventItemsManager;
class FWViewManagerManager;
class FWModelChangeManager;
class FWSelectionManager;
class FWGUIManager;
class FWEventItem;
class FWPhysicsObjectDesc;
class FWConfigurationManager;
class FWTextView;
class FWColorManager;

namespace fwlite {
   class Event;
}

class FWDisplayEvent
{

public:
   FWDisplayEvent(const std::string& iConfigFileName = std::string(),
                  bool iEnableDebug = false, bool iNewLego = true);
   virtual ~FWDisplayEvent();

   // ---------- const member functions ---------------------
   int draw(const fwlite::Event& ) const;

   const DetIdToMatrix& getIdToGeo() const {
      return m_detIdToGeo;
   }

   void writeConfigurationFile(const std::string& iFileName) const;
   // ---------- static member functions --------------------
   static double getMagneticField() {
      return m_magneticField;
   }
   static void   setMagneticField(double var) {
      m_magneticField = var;
   }
   static double getCaloScale() {
      return m_caloScale;
   }
   static void   setCaloScale(double var) {
      m_caloScale = var;
   }

   // ---------- member functions ---------------------------
   int draw(const fwlite::Event& );

   void registerPhysicsObject(const FWPhysicsObjectDesc&);
private:
   FWDisplayEvent(const FWDisplayEvent&);    // stop default

   const FWDisplayEvent& operator=(const FWDisplayEvent&);    // stop default

   // ---------- member data --------------------------------
   std::auto_ptr<FWConfigurationManager> m_configurationManager;
   std::auto_ptr<FWModelChangeManager> m_changeManager;
   std::auto_ptr<FWColorManager> m_colorManager;
   std::auto_ptr<FWSelectionManager> m_selectionManager;
   std::auto_ptr<FWEventItemsManager> m_eiManager;
   std::auto_ptr<FWViewManagerManager> m_viewManager;
   std::auto_ptr<FWGUIManager> m_guiManager;
   std::auto_ptr<FWTextView> m_textView;

   DetIdToMatrix m_detIdToGeo;
   std::string m_configFileName;
   static double m_magneticField;
   static double m_caloScale;
};


#endif
