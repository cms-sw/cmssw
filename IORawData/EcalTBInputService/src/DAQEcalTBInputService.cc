/*
 *  $Date: 2008/03/05 19:06:42 $
 *  $Revision: 1.18 $
 *  \author N. Amapane - S. Argiro'
 *  \author G. Franzoni
 */

#include "DAQEcalTBInputService.h"

#include "IORawData/EcalTBInputService/src/EcalTBDaqFileReader.h"

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <iostream>

using namespace edm;
using namespace std;

DAQEcalTBInputService::DAQEcalTBInputService(const ParameterSet& pset, 
					     const InputSourceDescription& desc) : 
  edm::ExternalInputSource(pset,desc), fileCounter_(0), eventRead_(0)
{    
  isBinary_= pset.getUntrackedParameter<bool>("isBinary",true);
  if ( isBinary_ ) {
    LogInfo("EcalTBInputService") << "@SUB=DAQEcalTBInputService" << "BINARY input data file";
  } else {
    LogInfo("EcalTBInputService") << "@SUB=DAQEcalTBInputService" << "ASCII input data file";
  }
  runNumber_ = pset.getUntrackedParameter<unsigned int>("runNumber", 1);
  reader_ = new EcalTBDaqFileReader();
  produces<FEDRawDataCollection>();
}


DAQEcalTBInputService::~DAQEcalTBInputService(){
  if (reader_)
    delete reader_;
  //  clear();
}


// void DAQEcalTBInputService::clear() {
//   for(map<int, DaqFEDRawData *>::iterator it = daqevdata_.begin();
//       it != daqevdata_.end(); ++it) {
//     delete (*it).second;
//   }
//   daqevdata_.clear();
// }

void DAQEcalTBInputService::setRunAndEventInfo()
{
  
  eventRead_=false;

  if ( !reader_->isInitialized() || reader_->checkEndOfFile() )
    {
      if (fileCounter_>=(unsigned int)(fileNames().size())) return; // nothing good
      reader_->initialize(fileNames()[fileCounter_],isBinary_);
      fileCounter_++;
    }
  
  eventRead_=reader_->fillDaqEventData();
  
  if (eventRead_)
    {
      if ( reader_->getRunNumber() != 0 ) {
        setRunNumber(reader_->getRunNumber());
      } else {
        setRunNumber( runNumber_ );
      }
      //For the moment adding 1 by hand (CMS has event number starting from 1)
      setEventNumber(reader_->getEventNumber()+1);
      // time is a hack
      edm::TimeValue_t present_time = presentTime();
      unsigned long time_between_events = timeBetweenEvents();
      setTime(present_time + time_between_events);
    }
  else
    return;
}

bool DAQEcalTBInputService::produce(edm::Event& e) 
{
  if (! eventRead_ )
    return false;
  
  std::auto_ptr<FEDRawDataCollection> bare_product(new  FEDRawDataCollection());

  FEDRawData& eventfeddata = (*bare_product).FEDData(reader_->getFedId());
  eventfeddata.resize(reader_->getFedData().len);
  copy(reader_->getFedData().fedData, reader_->getFedData().fedData + reader_->getFedData().len , eventfeddata.data());

  LogInfo("EcalTBInputService") << "@SUB=DAQEcalTBInputService::produce" << "read run " << reader_->getRunNumber() << " ev " << reader_->getEventNumber();

  e.put(bare_product);

  return true;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

DEFINE_FWK_INPUT_SOURCE(DAQEcalTBInputService);
