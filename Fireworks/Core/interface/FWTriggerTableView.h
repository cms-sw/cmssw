// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableView_h
#define Fireworks_Core_FWTriggerTableView_h
//
// Package:     Core
// Class  :     FWTriggerTableView
// $Id: FWTriggerTableView.h,v 1.7 2009/09/23 20:30:57 chrjones Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"

// forward declarations
class TGFrame;
class TGLEmbeddedViewer;
class TGCompositeFrame;
class TEvePad;
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveGeoShape;
class TGLMatrix;
class TGTextEntry;
class FWEventItem;
class FWTriggerTableViewManager;
class FWTableWidget;
class FWEveValueScaler;
class TEveWindowFrame;
class TEveWindowSlot;
class FWTriggerTableViewManager;
class FWTriggerTableViewTableManager;
class FWCustomIconsButton;
class FWGUIValidatingTextEntry;
class FWExpressionValidator;

namespace fwlite {
   class Event;
}

class FWTriggerTableView : public FWViewBase {
   friend class FWTriggerTableViewTableManager;

public:
   struct Column {
      std::string title;
      std::vector<std::string> values;
      Column(const std::string& s) : title(s){
      }
   };

   FWTriggerTableView(TEveWindowSlot *, FWTriggerTableViewManager *);
   virtual ~FWTriggerTableView();

   // ---------- const member functions ---------------------
   TGFrame* frame() const;
   const std::string& typeName() const;
   virtual void addTo(FWConfiguration&) const;

   virtual void saveImageTo(const std::string& iName) const;

   // ---------- static member functions --------------------
   static const std::string& staticTypeName();

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);
   void setBackgroundColor(Color_t);
   void resetColors (const class FWColorManager &);
   void dataChanged ();
   void columnSelected (Int_t iCol, Int_t iButton, Int_t iKeyMod);

private:
   FWTriggerTableView(const FWTriggerTableView&);      // stop default
   const FWTriggerTableView& operator=(const FWTriggerTableView&);      // stop default

   void fillAverageAcceptFractions();
protected:
   TEveWindowFrame*                m_eveWindow;
   TGCompositeFrame*               m_vert;
   FWTriggerTableViewManager*      m_manager;
   FWTriggerTableViewTableManager* m_tableManager;
   FWTableWidget*                  m_tableWidget;
   int m_currentColumn;
   std::vector<Column>             m_columns;
   fwlite::Event*                  m_event;
   std::vector<double>             m_averageAccept;
};


#endif
