// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableView_h
#define Fireworks_Core_FWTriggerTableView_h
//
// Package:     Core
// Class  :     FWTriggerTableView
// $Id: FWTriggerTableView.h,v 1.5 2010/09/02 18:10:10 amraktad Exp $
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

   FWTriggerTableView( TEveWindowSlot *, FWTriggerTableViewManager * );
   virtual ~FWTriggerTableView( void );

   // ---------- const member functions ---------------------
   TGFrame*       frame( void ) const;
   virtual void   addTo( FWConfiguration& ) const;
   virtual void   saveImageTo( const std::string& iName ) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom( const FWConfiguration& );
   void setBackgroundColor( Color_t );
   void resetColors( const class FWColorManager& );
   void dataChanged( void );
   void columnSelected( Int_t iCol, Int_t iButton, Int_t iKeyMod );
   void updateFilter( void );

private:
   FWTriggerTableView( const FWTriggerTableView& );      // stop default
   const FWTriggerTableView& operator=( const FWTriggerTableView& );      // stop default

   void fillAverageAcceptFractions( void );
	
protected:
   typedef boost::unordered_map<std::string,double> acceptmap_t;
   TEveWindowFrame*                m_eveWindow;
   TGCompositeFrame*               m_vert;
   FWTriggerTableViewManager*      m_manager;
   FWTriggerTableViewTableManager* m_tableManager;
   FWTableWidget*                  m_tableWidget;
   int                             m_currentColumn;
   std::vector<Column>             m_columns;
   fwlite::Event*                  m_event;
   acceptmap_t                     m_averageAccept;
   FWStringParameter               m_regex;
   FWStringParameter               m_process;
};


#endif
