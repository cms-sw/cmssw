// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTriggerTableView
// $Id: FWTriggerTableView.cc,v 1.9 2010/06/16 17:08:39 amraktad Exp $
//

// system include files
#include <boost/regex.hpp>

#include "TEveWindow.h"

// user include files
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWTriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWTriggerTableViewTableManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// constants, enums and typedefs
//
static const std::string kTableView = "TriggerTableView";
static const std::string kColumns = "columns";
static const std::string kSortColumn = "sortColumn";
static const std::string kDescendingSort = "descendingSort";

//
// constructors and destructor
//
FWTriggerTableView::FWTriggerTableView (TEveWindowSlot* iParent, FWTriggerTableViewManager *manager)
   : m_manager(manager),
     m_tableManager(new FWTriggerTableViewTableManager(this)),
     m_tableWidget(0),
     m_currentColumn(-1),
     m_event(0),
     m_regex(this,"Filter",std::string()),
     m_process(this,"Process",std::string("HLT"))
{
   m_columns.push_back(Column("Filter Name"));
   m_columns.push_back(Column("Accept"));
   m_columns.push_back(Column("Average Accept"));
   m_eveWindow = iParent->MakeFrame(0);
   m_regex.changed_.connect(boost::bind(&FWTriggerTableView::updateFilter,this));
   m_process.changed_.connect(boost::bind(&FWTriggerTableView::updateFilter,this));
   TGCompositeFrame *frame = m_eveWindow->GetGUICompositeFrame();

   m_vert = new TGVerticalFrame(frame);
   frame->AddFrame(m_vert, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   m_tableWidget = new FWTableWidget(m_tableManager, m_vert);
   resetColors(m_manager->colorManager());
   m_tableWidget->SetHeaderBackgroundColor(gVirtualX->GetPixel(kWhite));
   m_tableWidget->Connect("columnClicked(Int_t,Int_t,Int_t)", "FWTriggerTableView",
                          this, "columnSelected(Int_t,Int_t,Int_t)");
   m_vert->AddFrame(m_tableWidget, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   dataChanged();
   frame->MapSubwindows();
   frame->Layout();
   frame->MapWindow();
}

FWTriggerTableView::~FWTriggerTableView()
{
   // take out composite frame and delete it directly ( without the timeout)
   TGCompositeFrame *frame = m_eveWindow->GetGUICompositeFrame();
   frame->RemoveFrame(m_vert);
   delete m_vert;

   m_eveWindow->DestroyWindowAndSlot();
   delete m_tableManager;
}

void
FWTriggerTableView::setBackgroundColor(Color_t iColor)
{
   m_tableWidget->SetBackgroundColor(gVirtualX->GetPixel(iColor));
}

void FWTriggerTableView::resetColors (const FWColorManager &manager)
{
   m_tableWidget->SetBackgroundColor(gVirtualX->GetPixel(manager.background()));
   m_tableWidget->SetLineSeparatorColor(gVirtualX->GetPixel(manager.foreground()));
}

//
// const member functions
//
TGFrame*
FWTriggerTableView::frame() const
{
   return 0;
}

const std::string&
FWTriggerTableView::typeName() const
{
   return staticTypeName();
}

void
FWTriggerTableView::addTo( FWConfiguration& iTo ) const
{
	// are we the first FWTriggerTableView to go into the configuration?  If
	// we are, then we are responsible for writing out the list of
	// types (which we do by letting FWTriggerTableViewManager::addToImpl
	// write into our configuration)
	if( this == m_manager->m_views.front().get())
		m_manager->addToImpl( iTo );
	
	// then there is the stuff we have to do anyway: remember which is
	// a sorted column
	FWConfiguration main( 1 );
	FWConfiguration sortColumn( m_tableWidget->sortedColumn());
	main.addKeyValue( kSortColumn, sortColumn );
	FWConfiguration descendingSort( m_tableWidget->descendingSort());
	main.addKeyValue( kDescendingSort, descendingSort );
	iTo.addKeyValue( kTableView, main );
	
	// take care of parameters
	FWConfigurableParameterizable::addTo( iTo );
}

void
FWTriggerTableView::saveImageTo(const std::string& iName) const {
}


void FWTriggerTableView::dataChanged ()
{
   m_columns.at(0).values.clear();
   m_columns.at(1).values.clear();
   m_columns.at(2).values.clear();
   boost::regex filter(m_regex.value());
   if ( !m_manager->items().empty() && m_manager->items().front() != 0) {
      if ( fwlite::Event* event = const_cast<fwlite::Event*>(m_manager->items().front()->getEvent()) ) {
         if ( event != m_event ) {
            m_event = event;
            fillAverageAcceptFractions();
         }
         fwlite::Handle<edm::TriggerResults> hTriggerResults;
         edm::TriggerNames const* triggerNames(0);
         try{
            hTriggerResults.getByLabel(*event,"TriggerResults","",m_process.value().c_str());
            triggerNames = &event->triggerNames(*hTriggerResults);
         } catch (cms::Exception&) {
            fwLog(fwlog::kWarning) << " no trigger results with process name HLT is available" << std::endl;
            m_tableManager->dataChanged();
            return;
         }
         for(unsigned int i=0; i<triggerNames->size(); ++i) {
            if ( !boost::regex_search(triggerNames->triggerName(i),filter) ) continue;
            m_columns.at(0).values.push_back(triggerNames->triggerName(i));
            m_columns.at(1).values.push_back(Form("%d",hTriggerResults->accept(i)));
            m_columns.at(2).values.push_back(Form("%6.1f%%",m_averageAccept[triggerNames->triggerName(i)]*100));
         }
      }
   }
   m_tableManager->dataChanged();
}

void
FWTriggerTableView::columnSelected (Int_t iCol, Int_t iButton, Int_t iKeyMod)
{
   if (iButton == 1 || iButton == 3)
      m_currentColumn = iCol;
}

void
FWTriggerTableView::fillAverageAcceptFractions()
{
   edm::EventID currentEvent = m_event->id();
   // better to keep the keys and just set to zero the values
   for (acceptmap_t::iterator it = m_averageAccept.begin(), ed = m_averageAccept.end(); it != ed; ++it) {
      it->second = 0;
   }

   // loop over events
   fwlite::Handle<edm::TriggerResults> hTriggerResults;
   for (m_event->toBegin(); !m_event->atEnd(); ++(*m_event)) {
      hTriggerResults.getByLabel(*m_event,"TriggerResults","","HLT");
      edm::TriggerNames const* triggerNames(0);
      try{
         triggerNames = &m_event->triggerNames(*hTriggerResults);
      } catch (cms::Exception&) {
         fwLog(fwlog::kError) <<" exception caught while trying to get trigger info"<<std::endl;
         break;
      }

      for(unsigned int i=0; i<triggerNames->size(); ++i) {
         if ( hTriggerResults->accept(i) ) { 
            m_averageAccept[triggerNames->triggerName(i)]++;
         }
      }
   }
   m_event->to(currentEvent);

   double denominator = 1.0/m_event->size();
   for (acceptmap_t::iterator it = m_averageAccept.begin(), ed = m_averageAccept.end(); it != ed; ++it) {
      it->second *= denominator;
   }
}

void 
FWTriggerTableView::updateFilter( void )
{
	dataChanged();
}

//
// static member functions
//
const std::string&
FWTriggerTableView::staticTypeName( void )
{
   static std::string s_name( "TriggerTable" );
   return s_name;
}

void
FWTriggerTableView::setFrom( const FWConfiguration& iFrom )
{
	if( this == m_manager->m_views.front().get())
		m_manager->setFrom( iFrom );

	const FWConfiguration *main = iFrom.valueForKey( kTableView );
	if( main != 0 )
	{
		const FWConfiguration *sortColumn = main->valueForKey( kSortColumn );
		const FWConfiguration *descendingSort = main->valueForKey( kDescendingSort );
		if( sortColumn != 0 && descendingSort != 0 ) 
		{
			unsigned int sort = sortColumn->version();
			bool descending = descendingSort->version();
			if( sort < (( unsigned int ) m_tableManager->numberOfColumns()))
				m_tableWidget->sort( sort, descending );
		}
	} 
	else
	{
		// configuration doesn't contain info for the table.  Be forgiving.
		fwLog( fwlog::kError ) 
		<< "This configuration file contains trigger tables, but no column information.  "
		<< "(It is probably old.)  Using defaults." << std::endl;
	}
	
	// take care of parameters
	FWConfigurableParameterizable::setFrom( iFrom );
}

