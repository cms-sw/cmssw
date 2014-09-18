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
  
  Phase2Tracker::Phase2TrackerHeaderProducer::Phase2TrackerHeaderProducer(const edm::ParameterSet& pset)
  {
     produces<edm::DetSet<Phase2TrackerHeaderDigi>>("TrackerHeader");
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
     size_t fedIndex;
     for( fedIndex = Phase2Tracker::FED_ID_MIN; fedIndex < Phase2Tracker::CMS_FED_ID_MAX; ++fedIndex )
     {
       const FEDRawData& fed = buffers->FEDData(fedIndex);
       if(fed.size()!=0)
       {
         // construct buffer
         Phase2Tracker:: Phase2TrackerFEDBuffer * buffer = new Phase2Tracker::Phase2TrackerFEDBuffer(fed.data(),fed.size());
         Phase2TrackerHeaderDigi head_digi = Phase2TrackerHeaderDigi(buffer->trackerHeader());
         // store digis
         edm::DetSet<Phase2TrackerHeaderDigi> *header_digi = new edm::DetSet<Phase2TrackerHeaderDigi>(fedIndex);
         header_digi->push_back(head_digi);
         std::auto_ptr< edm::DetSet<Phase2TrackerHeaderDigi> > hdd(header_digi);
         event.put( hdd, "TrackerHeader" );
       }
     }
  }
} // end Phase2tracker Namespace  
