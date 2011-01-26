// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableView_h
#define Fireworks_Core_FWTriggerTableView_h
//
// Package:     Core
// Class  :     FWTriggerTableView
// $Id: FWTriggerTableView.h,v 1.7 2011/01/26 11:47:06 amraktad Exp $
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
class TEveWindowFrame;
class TEveWindowSlot;
class FWTriggerTableViewManager;
class FWTriggerTableViewTableManager;

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

   FWTriggerTableView( TEveWindowSlot *, FWViewType::EType );
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

private:
   FWTriggerTableView( const FWTriggerTableView& );      // stop default
   const FWTriggerTableView& operator=( const FWTriggerTableView& );      // stop default
	
protected:
   virtual void fillTable(fwlite::Event* event) = 0;

   typedef boost::unordered_map<std::string,double> acceptmap_t;
   TEveWindowFrame*                m_eveWindow;
   TGCompositeFrame*               m_vert;

   FWTriggerTableViewTableManager* m_tableManager;
   FWTableWidget*                  m_tableWidget;
   std::vector<Column>             m_columns;

   Color_t                         m_backgroundColor;
};


#endif
