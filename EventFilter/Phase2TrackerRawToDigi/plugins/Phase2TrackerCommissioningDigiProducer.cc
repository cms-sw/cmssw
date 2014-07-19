#include "EventFilter/Phase2TrackerRawToDigi/plugins/Phase2TrackerCommissioningDigiProducer.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ext/algorithm>

using namespace std;

namespace Phase2Tracker {

  Phase2TrackerCommissioningDigiProducer::Phase2TrackerCommissioningDigiProducer( const edm::ParameterSet& pset ) :
    runNumber_(0),
    productLabel_(pset.getParameter<edm::InputTag>("ProductLabel")),
    cabling_(0),
  {
    produces< edm::DetSet<Phase2TrackerCommissioningDigi> >("ConditionData");
  }
  
  Phase2TrackerCommissioningDigiProducer::~Phase2TrackerCommissioningDigiProducer()
  {
  }
  
  void Phase2TrackerCommissioningDigiProducer::produce( edm::Event& event, const edm::EventSetup& es)
  {
    // Retrieve FEDRawData collection
    edm::Handle<FEDRawDataCollection> buffers;
    event.getByLabel( "rawDataCollector", buffers );

    // Analyze strip tracker FED buffers in data
    size_t fedIndex;
    for( fedIndex=0; fedIndex<Phase2Tracker::CMS_FED_ID_MAX; ++fedIndex )
    {
      const FEDRawData& fed = buffers->FEDData(fedIndex);
      if(fed.size()!=0 && fedIndex >= Phase2Tracker::FED_ID_MIN && fedIndex <= Phase2Tracker::FED_ID_MAX)
      {
	// construct buffer
	Phase2Tracker:: Phase2TrackerFEDBuffer* buffer = 0;
	buffer = new Phase2Tracker::Phase2TrackerFEDBuffer(fed.data(),fed.size());

        // fetch condition data
        std::map<uint32_t,uint32_t> cond_data = buffer->conditionData();
        delete buffer;

        // print cond data for debug
        LogTrace("Phase2TrackerCommissioningDigiProducer") << "--- Condition data debug ---" << std::endl;
        std::map<uint32_t,uint32_t>::const_iterator it;
        for(it = cond_data.begin(); it != cond_data.end(); it++)
        {
          LogTrace("Phase2TrackerCommissioningDigiProducer") << std::hex << "key: " << it->first
                                                    << std::hex << " value: " << it->second << " (hex) "
                                                    << std::dec               << it->second << " (dec) " << std::endl;
        }
        LogTrace("Phase2TrackerCommissioningDigiProducer") << "----------------------------" << std::endl;
        // store it into digis
        edm::DetSet<Phase2TrackerCommissioningDigi> *cond_data_digi = new edm::DetSet<Phase2TrackerCommissioningDigi>(fedIndex);
        for(it = cond_data.begin(); it != cond_data.end(); it++)
        {
          cond_data_digi->push_back(Phase2TrackerCommissioningDigi(it->first,it->second));
        }
        std::auto_ptr< edm::DetSet<Phase2TrackerCommissioningDigi> > cdd(cond_data_digi);
        event.put( cdd, "ConditionData" );
      }
    }
  }
}
