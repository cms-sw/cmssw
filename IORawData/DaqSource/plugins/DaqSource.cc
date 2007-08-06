/** \file 
 *
 *  $Date: 2007/04/19 13:28:45 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - S. Argiro'
 */

#include "DaqSource.h"
#include <DataFormats/Provenance/interface/EventID.h>
#include <DataFormats/Provenance/interface/Timestamp.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>

#include "SealBase/Error.h"

#include <iostream>
#include <string>
#include <sys/time.h>


using namespace edm;
using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
DaqSource::DaqSource(const edm::ParameterSet& pset, 
		     const InputSourceDescription& desc) 
  : RawInputSource(pset,desc)
  , reader_(0)
  , lumiSegmentSizeInEvents_(pset.getUntrackedParameter<unsigned int>("evtsPerLS",0))
  , lsid_(0)
  , fakeLSid_(lumiSegmentSizeInEvents_ != 0)
{
  produces<FEDRawDataCollection>();
  
  // Instantiate the requested data source
  string reader = pset.getUntrackedParameter<string>("readerPluginName");
  
  try{
    reader_=
      DaqReaderPluginFactory::get()->create(reader,
					    pset.getUntrackedParameter<ParameterSet>("readerPset"));
  }
  catch(edm::Exception &e){
    if(e.category() == "Configuration" && reader_ == 0)
      {
	reader_=
	  DaqReaderPluginFactoryU::get()->create(reader);
	if(reader_==0) throw;
      }  
    else
      throw;
  }
}

//______________________________________________________________________________
DaqSource::~DaqSource()
{
  delete reader_;
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
std::auto_ptr<Event> DaqSource::readOneEvent()
{
  if(reader_==0)
    throw cms::Exception("LogicError")
      << "DaqSource is used without a reader. Check your configuration !";
  EventID eventId;
  edm::TimeValue_t time = 0LL;
  gettimeofday((timeval *)(&time),0);
  
  Timestamp tstamp(time);
  
  // pass a 0 pointer to fillRawData()!
  FEDRawDataCollection* fedCollection(0);

  // let reader_ fill the fedCollection 
  if (!reader_->fillRawData(eventId,tstamp,fedCollection)) {
    // fillRawData() failed, clean up the fedCollection in case it was allocated!
    if (0!=fedCollection) delete fedCollection;
    return std::auto_ptr<Event>(0);
  }
  if(fakeLSid_ && lsid_ != (eventId.event()/lumiSegmentSizeInEvents_ + 1))
      {
	lsid_ = eventId.event()/lumiSegmentSizeInEvents_ + 1;
	setLuminosityBlockNumber_t(lsid_);
      }
  // make a brand new event
  std::auto_ptr<Event> e=makeEvent(eventId,tstamp);
  
  // have fedCollection managed by a std::auto_ptr<>
  std::auto_ptr<FEDRawDataCollection> bare_product(fedCollection);
  
  // put the fed collection into the transient event store
  e->put(bare_product);

  return e;
}
