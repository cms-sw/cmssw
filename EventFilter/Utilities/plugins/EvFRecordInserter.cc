#include "EvFRecordInserter.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


namespace evf{

  EvFRecordInserter::EvFRecordInserter( const edm::ParameterSet& pset) 
    : evc_(0)
    , ehi_(0LL)
    , last_(0)
    , label_(pset.getParameter<edm::InputTag>("inputTag"))
  {}
    void EvFRecordInserter::analyze(const edm::Event & e, const edm::EventSetup& c)
    {
      edm::Handle<FEDRawDataCollection> rawdata;
      e.getByLabel(label_,rawdata);
      unsigned int id = fedinterface::EVFFED_ID;
      const FEDRawData& data = rawdata->FEDData(id);
      size_t size=data.size();
      unsigned char * cdata = const_cast<unsigned char*>(data.data());
      timeval now;
      gettimeofday(&now,0);
      evc_++;
      uint32_t ldiff = 0;
      if(e.id().event()-last_ < 0xff)
	ldiff = e.id().event()-last_;
      else
	ldiff = 0xff;
      ehi_ = (ehi_<<8)+(uint64_t)ldiff;
      last_ = e.id().event();
      if(size>0){
	ef_.setEPTimeStamp(((uint64_t)(now.tv_sec) << 32) 
			   + (uint64_t)(now.tv_usec),cdata);
	
	ef_.setEPProcessId(getpid(),cdata);
	ef_.setEPEventId(e.id().event(), cdata);
	ef_.setEPEventCount(evc_, cdata);
	ef_.setEPEventHisto(ehi_, cdata);
      }
    }
    

} // end namespace evf
