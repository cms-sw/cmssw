// -*- C++ -*-
//
// Package:     Core
// Class  :     FWL1TriggerTableView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Tue Jan 25 16:02:11 CET 2011
// $Id: FWL1TriggerTableView.cc,v 1.19 2011/10/19 07:13:35 yana Exp $
//

#include <boost/regex.hpp>
#include "Fireworks/Core/interface/FWL1TriggerTableView.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"


FWL1TriggerTableView::FWL1TriggerTableView(TEveWindowSlot* iParent)
   : FWTriggerTableView(iParent, FWViewType::kTableL1)
{ 
   m_columns[0].title = "Algorithm Name";
   m_columns.push_back( Column( "Result" ) );
   m_columns.push_back( Column( "Bit Number" ) );
   m_columns.push_back( Column( "Prescale" ) );

   dataChanged();
}

void
FWL1TriggerTableView::fillTable( fwlite::Event* event )
{
   fwlite::Handle<L1GtTriggerMenuLite> triggerMenuLite;
   fwlite::Handle<L1GlobalTriggerReadoutRecord> triggerRecord;

   try
   {
      // FIXME: Replace magic strings with configurable ones
      triggerMenuLite.getByLabel( event->getRun(), "l1GtTriggerMenuLite", "", "" );
      triggerRecord.getByLabel( *event, "gtDigis", "", "" );
   }
   catch( cms::Exception& )
   {
      fwLog( fwlog::kWarning ) << "FWL1TriggerTableView: no L1Trigger menu is available." << std::endl;
      return;
   }
	  
   if( triggerMenuLite.isValid() && triggerRecord.isValid() )
   {
     const L1GtTriggerMenuLite::L1TriggerMap& algorithmMap = triggerMenuLite->gtAlgorithmMap();
				
     int pfIndexTechTrig = -1;
     int pfIndexAlgoTrig = -1;

     boost::regex filter(m_regex.value());

     /// prescale factors
     std::vector<std::vector<int> > prescaleFactorsAlgoTrig = triggerMenuLite->gtPrescaleFactorsAlgoTrig();
     std::vector<std::vector<int> > prescaleFactorsTechTrig = triggerMenuLite->gtPrescaleFactorsTechTrig();
     pfIndexAlgoTrig = ( triggerRecord->gtFdlWord()).gtPrescaleFactorIndexAlgo();
     pfIndexTechTrig = ( triggerRecord->gtFdlWord()).gtPrescaleFactorIndexTech();

     int pfIndexTechTrigValidSize = static_cast<int>(prescaleFactorsTechTrig.size());
     if( pfIndexTechTrigValidSize <=  pfIndexTechTrig )
       fwLog( fwlog::kError) << Form( "FWL1TriggerTableView: Can't get Technical Trigger pre-scale factors. Index [%d] larger that table size [%d]\n",
				      pfIndexTechTrig, (int)prescaleFactorsTechTrig.size());
     int pfIndexAlgoTrigValidSize = static_cast<int>(prescaleFactorsAlgoTrig.size());
     if( pfIndexAlgoTrigValidSize <=  pfIndexAlgoTrig )
       fwLog( fwlog::kError) << Form( "FWL1TriggerTableView: Can't get L1 Algo pre-scale factors. Index [%d] larger that table size [%d]\n",
				      pfIndexAlgoTrig, (int)prescaleFactorsAlgoTrig.size());

     const DecisionWord dWord = triggerRecord->decisionWord();
     for( L1GtTriggerMenuLite::CItL1Trig itTrig = algorithmMap.begin(), itTrigEnd = algorithmMap.end();
	  itTrig != itTrigEnd; ++itTrig )
     {
       const unsigned int bitNumber = itTrig->first;
       const std::string& aName = itTrig->second;
       int errorCode = 0;
       const bool result = triggerMenuLite->gtTriggerResult( aName, dWord, errorCode );

       if ( !boost::regex_search(aName, filter) ) continue;

       m_columns.at(0).values.push_back( aName );
       m_columns.at(1).values.push_back( Form( "%d", result ));
       m_columns.at(2).values.push_back( Form( "%d", bitNumber ));

       if(( pfIndexAlgoTrig < pfIndexAlgoTrigValidSize )
	  && static_cast<unsigned int>(prescaleFactorsAlgoTrig.at(pfIndexAlgoTrig).size()) > bitNumber )
       {	 
	 m_columns.at(3).values.push_back( Form( "%d", prescaleFactorsAlgoTrig.at( pfIndexAlgoTrig ).at( bitNumber )));
       }
       else
	 m_columns.at(3).values.push_back( "invalid");
     }
     
     const static std::string kTechTriggerName = "TechTrigger";
     const TechnicalTriggerWord ttWord = triggerRecord->technicalTriggerWord();

     int tBitNumber = 0;
     int tBitResult = 0;
     if(boost::regex_search(kTechTriggerName, filter))
     {
       for( TechnicalTriggerWord::const_iterator tBitIt = ttWord.begin(), tBitEnd = ttWord.end(); 
	    tBitIt != tBitEnd; ++tBitIt, ++tBitNumber )
       {
	 if( *tBitIt )
	   tBitResult = 1;
	 else
	   tBitResult = 0;

	 m_columns.at(0).values.push_back( kTechTriggerName );
	 m_columns.at(1).values.push_back( Form( "%d", tBitResult ));
	 m_columns.at(2).values.push_back( Form( "%d", tBitNumber ));

	 if (( pfIndexTechTrig < pfIndexTechTrigValidSize )
	     && static_cast<int>(prescaleFactorsTechTrig.at(pfIndexTechTrig).size()) > tBitNumber )
	 {	      
	   m_columns.at(3).values.push_back( Form( "%d", prescaleFactorsTechTrig.at( pfIndexTechTrig ).at( tBitNumber )));
	 }	    
	 else
	   m_columns.at(3).values.push_back( Form( "invalid" ));
       }
     }
   } // trigger valid
   else
   {
     m_columns.at(0).values.push_back( "No L1Trigger menu available." );
     m_columns.at(1).values.push_back( " " );
     m_columns.at(2).values.push_back( " " );
     m_columns.at(3).values.push_back( " " );
   }
}
