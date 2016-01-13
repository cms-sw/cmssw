// -*- C++ -*-
//
// Package:    EventFilter/Phase2TrackerRawToDigi/Phase2TrackerHeaderProducer
// Class:      Phase2TrackerHeaderProducer
// 
/**\class Phase2TrackerHeaderProducer Phase2TrackerHeaderProducer.cc EventFilter/Phase2TrackerRawToDigi/plugins/Phase2TrackerHeaderProducer.cc

 Description: Producer for the phase 2 tracker header digi

*/
//
// Original Author:  Jerome De Favereau De Jeneret
//         Created:  Mon, 01 Sep 2014 08:42:31 GMT
//

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerHeaderDigi.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/plugins/Phase2TrackerHeaderProducer.h"   

using namespace std;

namespace Phase2Tracker {

  typedef std::vector<Phase2TrackerHeaderDigi> header_map;  

  Phase2Tracker::Phase2TrackerHeaderProducer::Phase2TrackerHeaderProducer(const edm::ParameterSet& pset)
  {
     produces<header_map>("TrackerHeader");
     token_ = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("ProductLabel"));
  }
  
  Phase2Tracker::Phase2TrackerHeaderProducer::~Phase2TrackerHeaderProducer() 
  {
  }
  
  void Phase2Tracker::Phase2TrackerHeaderProducer::produce( edm::Event& event, const edm::EventSetup& es)    
  {
     // Retrieve FEDRawData collection
     edm::Handle<FEDRawDataCollection> buffers;
     event.getByToken( token_, buffers );

     // fill collection
     std::auto_ptr<header_map> hdigis( new header_map ); /* switch to unique_ptr in CMSSW 7 */

     size_t fedIndex;
     for( fedIndex = Phase2Tracker::FED_ID_MIN; fedIndex < Phase2Tracker::CMS_FED_ID_MAX; ++fedIndex )
     {
       const FEDRawData& fed = buffers->FEDData(fedIndex);
       if(fed.size()==0) continue;
       // construct buffer
       Phase2Tracker::Phase2TrackerFEDBuffer buffer(fed.data(),fed.size());
       Phase2TrackerHeaderDigi head_digi = Phase2TrackerHeaderDigi(buffer.trackerHeader());
       // store digis
       hdigis->push_back(head_digi);
     }
     event.put(hdigis, "TrackerHeader" );
  }
} // end Phase2tracker Namespace  
