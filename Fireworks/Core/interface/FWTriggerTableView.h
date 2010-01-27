// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableView_h
#define Fireworks_Core_FWTriggerTableView_h
//
// Package:     Core
// Class  :     FWTriggerTableView
// $Id: FWTriggerTableView.h,v 1.2 2009/10/07 12:47:52 dmytro Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWStringParameter.h"


#define BOOST_NO_INITIALIZER_LISTS
// without this #define, genreflex chokes on std::initializer_list
// at least when buildig from tarball in SLC5 with the default gcc4.1.2
#include <boost/unordered_map.hpp>
#undef BOOST_NO_INITIALIZER_LISTS

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
   virtual void saveImageTo(const std::string& iName) const;

   // ---------- static member functions --------------------
   static const std::string& staticTypeName();

   // ---------- member functions ---------------------------
   void setBackgroundColor(Color_t);
   void resetColors (const class FWColorManager &);
   void dataChanged ();
   void columnSelected (Int_t iCol, Int_t iButton, Int_t iKeyMod);
   void updateFilter();

private:
   FWTriggerTableView(const FWTriggerTableView&);      // stop default
   const FWTriggerTableView& operator=(const FWTriggerTableView&);      // stop default

   void fillAverageAcceptFractions();
protected:
   typedef boost::unordered_map<std::string,double> acceptmap_t;
   TEveWindowFrame*                m_eveWindow;
   TGCompositeFrame*               m_vert;
   FWTriggerTableViewManager*      m_manager;
   FWTriggerTableViewTableManager* m_tableManager;
   FWTableWidget*                  m_tableWidget;
   int m_currentColumn;
   std::vector<Column>             m_columns;
   fwlite::Event*                  m_event;
   acceptmap_t                     m_averageAccept;
   FWStringParameter               m_regex;
   FWStringParameter               m_process;
};


#endif
