// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTriggerTableView
// $Id: FWTriggerTableView.cc,v 1.13 2010/11/04 22:38:55 amraktad Exp $
//

// system include files
#include <fstream>

#include "TEveWindow.h"

// user include files
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWTriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWTriggerTableViewTableManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"


#include "DataFormats/FWLite/interface/Event.h"

// configuration keys
static const std::string kColumns = "columns";
static const std::string kSortColumn = "sortColumn";
static const std::string kDescendingSort = "descendingSort";

//
//
// constructors and destructor
//
FWTriggerTableView::FWTriggerTableView (TEveWindowSlot* iParent, FWViewType::EType id)
   : FWViewBase(id, 2),
     m_tableManager(new FWTriggerTableViewTableManager(this)),
     m_tableWidget(0)
{  
   m_eveWindow = iParent->MakeFrame(0);
   TGCompositeFrame *frame = m_eveWindow->GetGUICompositeFrame();

   m_vert = new TGVerticalFrame(frame);
   frame->AddFrame(m_vert, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // have to have at least one column when call  FWTableWidget constructor
   m_columns.push_back( Column( "Name" ));

   m_tableWidget = new FWTableWidget(m_tableManager, frame); 
   m_tableWidget->SetHeaderBackgroundColor( gVirtualX->GetPixel( kWhite ));

   m_tableWidget->Connect("columnClicked(Int_t,Int_t,Int_t)", "FWTriggerTableView",
                          this, "columnSelected(Int_t,Int_t,Int_t)");
   m_vert->AddFrame(m_tableWidget, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));


   frame->MapSubwindows();
   frame->Layout();
   frame->MapWindow();
}

FWTriggerTableView::~FWTriggerTableView()
{
   // take out composite frame and delete it directly (without the timeout)
   TGCompositeFrame *frame = m_eveWindow->GetGUICompositeFrame();
   frame->RemoveFrame( m_vert );
   delete m_vert;

   m_eveWindow->DestroyWindowAndSlot();
   delete m_tableManager;
}

void
FWTriggerTableView::setBackgroundColor(Color_t iColor)
{
   m_backgroundColor = iColor;
   m_tableWidget->SetBackgroundColor(gVirtualX->GetPixel(iColor));
   m_tableWidget->SetLineSeparatorColor( gVirtualX->GetPixel(iColor == kWhite ?  kBlack : kWhite));
}

//
// const member functions
//

void
FWTriggerTableView::addTo( FWConfiguration& iTo ) const
{
   FWConfigurableParameterizable::addTo(iTo);
   FWConfiguration sortColumn( m_tableWidget->sortedColumn());
   iTo.addKeyValue( kSortColumn, sortColumn );
   FWConfiguration descendingSort( m_tableWidget->descendingSort());
   iTo.addKeyValue( kDescendingSort, descendingSort );
}

void
FWTriggerTableView::saveImageTo( const std::string& iName ) const
{
   std::string fileName = iName + ".txt";
   std::ofstream triggers( fileName.c_str() );

   triggers << m_columns[2].title << " " << m_columns[0].title << "\n";
   for( unsigned int i = 0, vend = m_columns[0].values.size(); i != vend; ++i )
      if( m_columns[1].values[i] == "1" )
         triggers << m_columns[2].values[i] << "\t" << m_columns[0].values[i] << "\n";
   triggers.close();
}


void FWTriggerTableView::dataChanged ()
{
   for(std::vector<Column>::iterator i = m_columns.begin(); i!= m_columns.end(); ++i)
      (*i).values.clear();

   edm::EventBase *base = const_cast<edm::EventBase*>(FWGUIManager::getGUIManager()->getCurrentEvent());
   if (fwlite::Event* event = dynamic_cast<fwlite::Event*>(base))
      fillTable(event);

   m_tableManager->dataChanged();
}

void
FWTriggerTableView::columnSelected (Int_t iCol, Int_t iButton, Int_t iKeyMod)
{
}


void 
FWTriggerTableView::updateFilter( void )
{
   dataChanged();
}

//
// static member functions
//

void
FWTriggerTableView::setFrom( const FWConfiguration& iFrom )
{
   const FWConfiguration *main = &iFrom;

   // unnecessary nesting for old version
   if (version() < 2)
   {
      if (typeId() == FWViewType::kTableHLT)
         main = iFrom.valueForKey( "HLTTriggerTableView" );
      else
         main = iFrom.valueForKey( "L1TriggerTableView" );
   }

   const FWConfiguration *sortColumn = main->valueForKey( kSortColumn );
   const FWConfiguration *descendingSort = main->valueForKey( kDescendingSort );
   if( sortColumn != 0 && descendingSort != 0 ) 
   {
      unsigned int sort = sortColumn->version();
      bool descending = descendingSort->version();
      if( sort < (( unsigned int ) m_tableManager->numberOfColumns()))
         m_tableWidget->sort( sort, descending );
   }

   // FWViewBase parameters
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->setFrom(iFrom);      
   }  
}


