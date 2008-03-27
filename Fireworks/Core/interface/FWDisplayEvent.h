#ifndef Fireworks_Core_FWDisplayEvent_h
#define Fireworks_Core_FWDisplayEvent_h
// -*- C++ -*-
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
// $Id: FWDisplayEvent.h,v 1.19 2008/03/20 09:39:26 dmytro Exp $
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

namespace fwlite {
  class Event;
}

class FWDetailView;

class FWDisplayEvent
{

   public:
   FWDisplayEvent(const std::string& iConfigFileName = std::string(),
                  bool iEnableDebug = false, bool iNewLego = true);
      virtual ~FWDisplayEvent();

      // ---------- const member functions ---------------------
      int draw(const fwlite::Event& ) const;

      const DetIdToMatrix& getIdToGeo() const { return m_detIdToGeo; }
   
      void writeConfigurationFile(const std::string& iFileName) const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      int draw(const fwlite::Event& );
      void registerProxyBuilder(const std::string&, 
				const std::string&);
      
      void registerPhysicsObject(const FWPhysicsObjectDesc&);
     void registerDetailView (const std::string &item_name, FWDetailView *view);
   private:
      FWDisplayEvent(const FWDisplayEvent&); // stop default

      const FWDisplayEvent& operator=(const FWDisplayEvent&); // stop default

      // ---------- member data --------------------------------
      std::auto_ptr<FWConfigurationManager> m_configurationManager;
      std::auto_ptr<FWModelChangeManager> m_changeManager;
      std::auto_ptr<FWSelectionManager> m_selectionManager;
      std::auto_ptr<FWEventItemsManager> m_eiManager;
      std::auto_ptr<FWGUIManager> m_guiManager;
      std::auto_ptr<FWViewManagerManager> m_viewManager;

      DetIdToMatrix    m_detIdToGeo;
};


#endif
