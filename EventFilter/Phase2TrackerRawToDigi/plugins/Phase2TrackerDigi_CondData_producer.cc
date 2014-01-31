#include "EventFilter/Phase2TrackerRawToDigi/plugins/Phase2TrackerDigi_CondData_producer.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ext/algorithm>

using namespace std;

namespace sistrip {

  Phase2TrackerDigi_CondData_producer::Phase2TrackerDigi_CondData_producer( const edm::ParameterSet& pset ) :
    runNumber_(0),
    productLabel_(pset.getParameter<edm::InputTag>("ProductLabel")),
    cabling_(0),
    cacheId_(0)
  {
    produces< edm::DetSet<SiStripCommissioningDigi> >("ConditionData");
  }
  
  Phase2TrackerDigi_CondData_producer::~Phase2TrackerDigi_CondData_producer()
  {
  }
  
  void Phase2TrackerDigi_CondData_producer::beginJob( const edm::EventSetup & )
  {
  }
  
  void Phase2TrackerDigi_CondData_producer::beginRun( edm::Run & run, const edm::EventSetup & es)
  {
  }
  
  void Phase2TrackerDigi_CondData_producer::endJob()
  {
  }
  
  void Phase2TrackerDigi_CondData_producer::produce( edm::Event& event, const edm::EventSetup& es)
  {
    // Retrieve FEDRawData collection
    edm::Handle<FEDRawDataCollection> buffers;
    event.getByLabel( "rawDataCollector", buffers );

    // Analyze strip tracker FED buffers in data
    size_t fedIndex;
    for( fedIndex=0; fedIndex<sistrip::CMS_FED_ID_MAX; ++fedIndex )
    {
      const FEDRawData& fed = buffers->FEDData(fedIndex);
      if(fed.size()!=0 && fedIndex >= sistrip::FED_ID_MIN && fedIndex <= sistrip::FED_ID_MAX)
      {
	// construct buffer
	sistrip:: Phase2TrackerFEDBuffer* buffer = 0;
	buffer = new sistrip::Phase2TrackerFEDBuffer(fed.data(),fed.size());

        // fetch condition data
        std::map<uint32_t,uint32_t> cond_data = buffer->conditionData();
        delete buffer;

        // print cond data for debug
        LogTrace("Phase2TrackerDigi_CondData_producer") << "--- Condition data debug ---" << std::endl;
        std::map<uint32_t,uint32_t>::const_iterator it;
        for(it = cond_data.begin(); it != cond_data.end(); it++)
        {
          LogTrace("Phase2TrackerDigi_CondData_producer") << std::hex << "key: " << it->first
                                                    << std::hex << " value: " << it->second << " (hex) "
                                                    << std::dec               << it->second << " (dec) " << std::endl;
        }
        LogTrace("Phase2TrackerDigi_CondData_producer") << "----------------------------" << std::endl;
        // store it into digis
        edm::DetSet<SiStripCommissioningDigi> *cond_data_digi = new edm::DetSet<SiStripCommissioningDigi>(fedIndex);
        for(it = cond_data.begin(); it != cond_data.end(); it++)
        {
          cond_data_digi->push_back(SiStripCommissioningDigi(it->first,it->second));
        }
        std::auto_ptr< edm::DetSet<SiStripCommissioningDigi> > cdd(cond_data_digi);
        event.put( cdd, "ConditionData" );
      }
    }
  }
}
