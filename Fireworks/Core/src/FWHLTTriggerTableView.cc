// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHLTTriggerTableView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Tue Jan 25 16:02:03 CET 2011
// $Id: FWHLTTriggerTableView.cc,v 1.3 2011/02/16 18:38:36 amraktad Exp $
//
#include <boost/regex.hpp>

#include "Fireworks/Core/interface/FWHLTTriggerTableView.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Fireworks/Core/interface/FWTriggerTableViewTableManager.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"


FWHLTTriggerTableView::FWHLTTriggerTableView(TEveWindowSlot* iParent)
   : FWTriggerTableView(iParent, FWViewType::kTableHLT),
     m_event(0)
{  

   m_columns[0].title = "Filter Name";
   m_columns.push_back(Column("Accept"));
   m_columns.push_back(Column("Average Accept"));
   dataChanged();
}


void FWHLTTriggerTableView::fillTable(fwlite::Event* event)
{
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
   boost::regex filter(m_regex.value());
   for(unsigned int i=0; i<triggerNames->size(); ++i) {
      if ( !boost::regex_search(triggerNames->triggerName(i),filter) ) continue;
      m_columns.at(0).values.push_back(triggerNames->triggerName(i));
      m_columns.at(1).values.push_back(Form("%d",hTriggerResults->accept(i)));
      m_columns.at(2).values.push_back(Form("%6.1f%%",m_averageAccept[triggerNames->triggerName(i)]*100));
   }
}

void
FWHLTTriggerTableView::fillAverageAcceptFractions()
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
