/** \file 
 *
 *  $Date: 2006/05/31 14:12:25 $
 *  $Revision: 1.8 $
 *  \author N. Amapane - S. Argiro'
 */

#include "DaqSource.h"
#include <DataFormats/Common/interface/EventID.h>
#include <DataFormats/Common/interface/Timestamp.h>
//#include <FWCore/Framework/interface/InputSourceDescription.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <IORawData/DaqSource/interface/DaqBaseReader.h>
#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>

#include <iostream>
#include <string>

using namespace edm;
using namespace std;


DaqSource::DaqSource(const ParameterSet& pset, 
		     const InputSourceDescription& desc) : 
  RawInputSource(pset, desc),
  reader_(0)
{
  produces<FEDRawDataCollection>();
  
  // Instantiate the requested data source
  string reader = pset.getParameter<string>("reader");
  reader_ = DaqReaderPluginFactory::get()->create(reader,pset.getParameter<ParameterSet>("pset"));
}

DaqSource::~DaqSource(){
  delete reader_;
}

#include <sys/time.h>

std::auto_ptr<Event>
DaqSource::readOneEvent() {

  EventID eventId;
  edm::TimeValue_t time = 0LL;
  gettimeofday((timeval *)time,0);
  
  Timestamp tstamp(time);

  std::auto_ptr<FEDRawDataCollection> bare_product(new FEDRawDataCollection);

  if (!reader_->fillRawData(eventId, tstamp, *bare_product)) {
    return std::auto_ptr<Event>(0);
  }
  //  std::cout << " making run " << eventId.run() << " event " << eventId.event()
  //	    << std::endl;
  std::auto_ptr<Event> e = makeEvent(eventId, tstamp);

  e->put(bare_product);

  return e;
}
