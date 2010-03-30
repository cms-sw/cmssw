// -*- C++ -*-
#ifndef Fireworks_Core_FWL1TriggerTableView_h
#define Fireworks_Core_FWL1TriggerTableView_h
//
// Package:     Core
// Class  :     FWL1TriggerTableView

#include "Fireworks/Core/interface/FWViewBase.h"

class TGFrame;
class TGCompositeFrame;
class FWTableWidget;
class FWEveValueScaler;
class TEveWindowFrame;
class TEveWindowSlot;
class FWL1TriggerTableViewManager;
class FWL1TriggerTableViewTableManager;

class FWL1TriggerTableView : public FWViewBase
{
   friend class FWL1TriggerTableViewTableManager;

public:
   struct Column {
      std::string title;
      std::vector<std::string> values;
      Column(const std::string& s) : title(s){
      }
   };

   FWL1TriggerTableView(TEveWindowSlot *, FWL1TriggerTableViewManager *);
   virtual ~FWL1TriggerTableView(void);

   TGFrame* 		frame(void) const;
   const std::string& 	typeName(void) const;
   virtual void 	saveImageTo(const std::string& iName) const;

   static const std::string& staticTypeName(void);

   void 		setBackgroundColor(Color_t);
   void 		resetColors(const class FWColorManager &);
   void 		dataChanged(void);
   void 		columnSelected(Int_t iCol, Int_t iButton, Int_t iKeyMod);

private:
   FWL1TriggerTableView(const FWL1TriggerTableView&);      // stop default
   const FWL1TriggerTableView& operator=(const FWL1TriggerTableView&);      // stop default

protected:
   TEveWindowFrame*                m_eveWindow;
   TGCompositeFrame*               m_vert;
   FWL1TriggerTableViewManager*      m_manager;
   FWL1TriggerTableViewTableManager* m_tableManager;
   FWTableWidget*                  m_tableWidget;
   int 				   m_currentColumn;
   std::vector<Column>             m_columns;
};

#endif // Fireworks_Core_FWL1TriggerTableView_h
