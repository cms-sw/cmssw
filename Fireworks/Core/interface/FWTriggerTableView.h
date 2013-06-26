// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableView_h
#define Fireworks_Core_FWTriggerTableView_h
//
// Package:     Core
// Class  :     FWTriggerTableView
// $Id: FWTriggerTableView.h,v 1.9 2011/02/16 18:38:36 amraktad Exp $
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
class TGCompositeFrame;
class FWTableWidget;
class TGComboBox;
class TEveWindowFrame;
class TEveWindowSlot;
class FWTriggerTableViewManager;
class FWTriggerTableViewTableManager;
class FWJobMetadataManager;

namespace fwlite {
   class Event;
}

class FWTriggerTableView : public FWViewBase 
{
   friend class FWTriggerTableViewTableManager;
public:
   struct Column {
      std::string title;
      std::vector<std::string> values;
      Column( const std::string& s ) : title( s )
      {}
   };

   FWTriggerTableView(TEveWindowSlot *, FWViewType::EType );
   virtual ~FWTriggerTableView( void );

   // ---------- const member functions ---------------------
   virtual void   addTo( FWConfiguration& ) const;
   virtual void   saveImageTo( const std::string& iName ) const;
   Color_t backgroundColor() const { return m_backgroundColor; }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom( const FWConfiguration& );
   void setBackgroundColor( Color_t );
   //void resetColors( const class FWColorManager& );
   void dataChanged( void );
   void columnSelected( Int_t iCol, Int_t iButton, Int_t iKeyMod );

   void setProcessList( std::vector<std::string>* x) { m_processList = x; }
   void resetCombo() const;
   //  void processChanged(Int_t);
  void processChanged(const char*);
protected:
   FWStringParameter               m_regex;
   FWStringParameter               m_process;

   std::vector<Column>             m_columns;
   FWTriggerTableViewTableManager* m_tableManager;

   virtual void fillTable(fwlite::Event* event) = 0;

private:
   FWTriggerTableView( const FWTriggerTableView& );      // stop default
   const FWTriggerTableView& operator=( const FWTriggerTableView& );      // stop default

   bool isProcessValid()const;
   virtual void populateController(ViewerParameterGUI&) const;

   mutable TGComboBox*             m_combo;

   // destruction
   TEveWindowFrame*                m_eveWindow;
   TGCompositeFrame*               m_vert;


   FWTableWidget*                  m_tableWidget;

   Color_t                         m_backgroundColor;

   std::vector<std::string>*       m_processList;

};


#endif
