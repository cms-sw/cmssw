/** \file 
 *
 *  $Date: 2009/07/02 15:22:03 $
 *  $Revision: 1.31 $
 *  \author N. Amapane - S. Argiro'
 */

#include "DaqSource.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"

#include "IORawData/DaqSource/interface/DaqBaseReader.h"
#include "IORawData/DaqSource/interface/DaqReaderPluginFactory.h"

#include "DataFormats/Provenance/interface/Timestamp.h" 
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

#include <string>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <linux/unistd.h>


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////


namespace daqsource{
  static unsigned int gtpEvmId_ =  FEDNumbering::getTriggerGTPFEDIds().first;
  static unsigned int gtpeId_ =  FEDNumbering::getTriggerEGTPFEDIds().first;
}

namespace edm {
 namespace daqsource{
  static unsigned int gtpEvmId_ =  FEDNumbering::getTriggerGTPFEDIds().first;
  static unsigned int gtpeId_ =  FEDNumbering::getTriggerEGTPFEDIds().first;
 }

  //______________________________________________________________________________
  DaqSource::DaqSource(const ParameterSet& pset, 
		     const InputSourceDescription& desc) 
    : InputSource(pset,desc)
    , evf::ModuleWeb("DaqSource")
    , reader_(0)
    , lumiSegmentSizeInEvents_(pset.getUntrackedParameter<unsigned int>("evtsPerLS",0))
    , fakeLSid_(lumiSegmentSizeInEvents_ != 0)
    , runNumber_(RunID::firstValidRun().run())
    , luminosityBlockNumber_(LuminosityBlockID::firstValidLuminosityBlock().luminosityBlock())
    , noMoreEvents_(false)
    , newRun_(true)
    , newLumi_(true)
    , ep_()
    , lumiSectionIndex_(1)
    , prescaleSetIndex_(0)
    , lsTimedOut_(false)
    , is_(0)
    , mis_(0)
  {
    count = 0;
    pthread_mutex_init(&mutex_,0);
    pthread_cond_init(&cond_,0);
    produces<FEDRawDataCollection>();
    setTimestamp(Timestamp::beginOfTime());
    
    // Instantiate the requested data source
    std::string reader = pset.getUntrackedParameter<std::string>("readerPluginName");
    
    try{
      reader_=
        DaqReaderPluginFactory::get()->create(reader,
  					    pset.getUntrackedParameter<ParameterSet>("readerPset"));
      reader_->setRunNumber(runNumber_);
    }
    catch(edm::Exception &e) {
      if(e.category() == "Configuration" && reader_ == 0) {
  	reader_ = DaqReaderPluginFactoryU::get()->create(reader);
  	if(reader_ == 0) throw;
	else reader_->setRunNumber(runNumber_);
      }
      else {
        throw;
      }
    }
  }
  
  //______________________________________________________________________________
  DaqSource::~DaqSource() {
    if(is_)
      {
	is_->fireItemRevoked("lumiSectionIndex");
	is_->fireItemRevoked("prescaleSetIndex");
	is_->fireItemRevoked("lsTimedOut");
      }
    if(mis_)
      {
	mis_->fireItemRevoked("lumiSectionIndex");
	mis_->fireItemRevoked("prescaleSetIndex");
	mis_->fireItemRevoked("lsTimedOut");
      }
    delete reader_;
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  // implementation of member functions
  ////////////////////////////////////////////////////////////////////////////////
  
  //______________________________________________________________________________
  InputSource::ItemType 
  DaqSource::getNextItemType() {
    if (noMoreEvents_) {
      pthread_mutex_lock(&mutex_);
      pthread_cond_signal(&cond_);
      pthread_mutex_unlock(&mutex_);
      return IsStop;
    }
    if (newRun_) {
      return IsRun;
    }
    if (newLumi_ && luminosityBlockPrincipal()) {
      return IsLumi;
    }
    if (ep_.get() != 0) {
      return IsEvent;
    }
    if(reader_ == 0) {
      throw edm::Exception(errors::LogicError)
        << "DaqSource is used without a reader. Check your configuration !";
    }
    EventID eventId;
    TimeValue_t time = 0LL;
    timeval stv;
    gettimeofday(&stv,0);
    time = stv.tv_sec;
    time = (time << 32) + stv.tv_usec;
    Timestamp tstamp(time);

    int bunchCrossing = EventAuxiliary::invalidBunchXing;
    int orbitNumber   = EventAuxiliary::invalidBunchXing;
    
    // pass a 0 pointer to fillRawData()!
    FEDRawDataCollection* fedCollection(0);
  
    // let reader_ fill the fedCollection 
    if(!reader_->fillRawData(eventId, tstamp, fedCollection)) {
      // fillRawData() failed, clean up the fedCollection in case it was allocated!
      if (0 != fedCollection) delete fedCollection;
      noMoreEvents_ = true;
      pthread_mutex_lock(&mutex_);
      pthread_cond_signal(&cond_);
      pthread_mutex_unlock(&mutex_);
      return IsStop;
    }
    if (eventId.event() == 0) {
      throw edm::Exception(errors::LogicError)
        << "The reader used with DaqSource has returned an invalid (zero) event number!\n"
        << "Event numbers must begin at 1, not 0.";
    }
    EventSourceSentry(*this);
    setTimestamp(tstamp);
    
    unsigned char *gtpFedAddr = fedCollection->FEDData(daqsource::gtpEvmId_).size()!=0 ? fedCollection->FEDData(daqsource::gtpEvmId_).data() : 0;
    uint32_t gtpsize = 0;
    edm::EventAuxiliary::ExperimentType evttype = EventAuxiliary::Undefined;
    if(gtpFedAddr !=0) gtpsize = fedCollection->FEDData(daqsource::gtpEvmId_).size();
    unsigned char *gtpeFedAddr = fedCollection->FEDData(daqsource::gtpeId_).size()!=0 ? fedCollection->FEDData(daqsource::gtpeId_).data() : 0; 

    if(fakeLSid_ && luminosityBlockNumber_ != ((eventId.event() - 1)/lumiSegmentSizeInEvents_ + 1)) {
	luminosityBlockNumber_ = (eventId.event() - 1)/lumiSegmentSizeInEvents_ + 1;
	pthread_mutex_lock(&mutex_);
	pthread_cond_signal(&cond_);
	pthread_mutex_unlock(&mutex_);
	::usleep(1000);

	pthread_mutex_lock(&mutex_);
	pthread_mutex_unlock(&mutex_);
        newLumi_ = true;
	lumiSectionIndex_.value_ = luminosityBlockNumber_;
	resetLuminosityBlockPrincipal();
    }
    else if(!fakeLSid_){ 

      if(gtpFedAddr!=0 && evf::evtn::evm_board_sense(gtpFedAddr,gtpsize)){
	unsigned int thisEventLSid = evf::evtn::getlbn(gtpFedAddr);
	prescaleSetIndex_.value_ = (evf::evtn::getfdlpsc(gtpFedAddr) & 0xffff);
	evttype =  edm::EventAuxiliary::ExperimentType(evf::evtn::getevtyp(gtpFedAddr));
	if(luminosityBlockNumber_ != (thisEventLSid + 1)){
	  luminosityBlockNumber_ = thisEventLSid + 1;
	  pthread_mutex_lock(&mutex_);
	  pthread_cond_signal(&cond_);
	  pthread_mutex_unlock(&mutex_);
	  ::usleep(1000);

	  pthread_mutex_lock(&mutex_);
	  pthread_mutex_unlock(&mutex_);
	  newLumi_ = true;
	  lumiSectionIndex_.value_ = luminosityBlockNumber_;
	  resetLuminosityBlockPrincipal();
	}
      }
      else if(gtpeFedAddr!=0 && evf::evtn::gtpe_board_sense(gtpeFedAddr)){
	unsigned int thisEventLSid = evf::evtn::gtpe_getlbn(gtpeFedAddr);
	evttype =  edm::EventAuxiliary::PhysicsTrigger; 
	if(luminosityBlockNumber_ != (thisEventLSid + 1)){
	  luminosityBlockNumber_ = thisEventLSid + 1;
	  pthread_mutex_lock(&mutex_);
	  pthread_cond_signal(&cond_);
	  pthread_mutex_unlock(&mutex_);
	  ::usleep(1000);

	  pthread_mutex_lock(&mutex_);
	  pthread_mutex_unlock(&mutex_);
	  newLumi_ = true;
	  lumiSectionIndex_.value_ = luminosityBlockNumber_;
	  resetLuminosityBlockPrincipal();
	}
      }
    }
    if(gtpFedAddr!=0 && evf::evtn::evm_board_sense(gtpFedAddr,gtpsize)){
      bunchCrossing =  int(evf::evtn::getfdlbx(gtpFedAddr));
      orbitNumber =  int(evf::evtn::getorbit(gtpFedAddr));
      TimeValue_t time = evf::evtn::getgpshigh(gtpFedAddr);
      time = (time << 32) + evf::evtn::getgpslow(gtpFedAddr);
      Timestamp tstamp(time);
    }
    else if(gtpeFedAddr!=0 && evf::evtn::gtpe_board_sense(gtpeFedAddr)){
      bunchCrossing =  int(evf::evtn::gtpe_getbx(gtpeFedAddr));
      orbitNumber =  int(evf::evtn::gtpe_getorbit(gtpeFedAddr));
    }
    eventId = EventID(runNumber_, eventId.event());
    
    // If there is no luminosity block principal, make one.
    if (luminosityBlockPrincipal().get() == 0 || luminosityBlockPrincipal()->luminosityBlock() != luminosityBlockNumber_) {
      newLumi_ = true;
      LuminosityBlockAuxiliary lumiAux(runPrincipal()->run(),
        luminosityBlockNumber_, timestamp(), Timestamp::invalidTimestamp());
      setLuminosityBlockPrincipal(boost::shared_ptr<LuminosityBlockPrincipal>(
        new LuminosityBlockPrincipal(lumiAux,
                                     productRegistry(),
                                     processConfiguration())));

    }
    // make a brand new event
    EventAuxiliary eventAux(eventId,
      processGUID(), timestamp(), luminosityBlockPrincipal()->luminosityBlock(), true, evttype, bunchCrossing, EventAuxiliary::invalidStoreNumber,
			    orbitNumber);
    ep_ = std::auto_ptr<EventPrincipal>(
	new EventPrincipal(eventAux, productRegistry(), processConfiguration()));
    
    // have fedCollection managed by a std::auto_ptr<>
    std::auto_ptr<FEDRawDataCollection> bare_product(fedCollection);

    std::auto_ptr<Event> e(new Event(*ep_, moduleDescription()));
    // put the fed collection into the transient event store
    e->put(bare_product);
    // The commit is needed to complete the "put" transaction.
    e->commit_();
    if (newLumi_) {
      return IsLumi;
    }
    return IsEvent;
  }

  void
  DaqSource::setRun(RunNumber_t r) {
    assert(ep_.get() == 0);
    reset();
    newRun_ = newLumi_ = true;
    runNumber_ = r;
    if (reader_) reader_->setRunNumber(runNumber_);
    noMoreEvents_ = false;
    resetLuminosityBlockPrincipal();
    resetRunPrincipal();
  }

  boost::shared_ptr<RunPrincipal>
  DaqSource::readRun_() {
    assert(newRun_);
    assert(!noMoreEvents_);
    newRun_ = false;
    RunAuxiliary runAux(runNumber_, timestamp(), Timestamp::invalidTimestamp());
    return boost::shared_ptr<RunPrincipal>(
	new RunPrincipal(runAux,
			 productRegistry(),
			 processConfiguration()));
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  DaqSource::readLuminosityBlock_() {
    assert(!newRun_);
    assert(newLumi_);
    assert(!noMoreEvents_);
    assert(luminosityBlockPrincipal());
    assert(ep_.get() != 0);
    newLumi_ = false;
    return luminosityBlockPrincipal();
  }

  std::auto_ptr<EventPrincipal>
  DaqSource::readEvent_() {
    assert(!newRun_);
    assert(!newLumi_);
    assert(!noMoreEvents_);
    assert(ep_.get() != 0);
    std::auto_ptr<EventPrincipal> result = ep_;
    return result;
  }

  void
  DaqSource::setLumi(LuminosityBlockNumber_t) {
      throw edm::Exception(errors::LogicError,"DaqSource::setLumi(LuminosityBlockNumber_t lumiNumber)")
        << "The luminosity block number cannot be set externally for DaqSource.\n"
        << "Contact a Framework developer.\n";
  }

  std::auto_ptr<EventPrincipal>
  DaqSource::readIt(EventID const&) {
      throw edm::Exception(errors::LogicError,"DaqSource::readIt(EventID const& eventID)")
        << "Random access read cannot be used for DaqSource.\n"
        << "Contact a Framework developer.\n";
  }

  void
  DaqSource::skip(int) {
      throw edm::Exception(errors::LogicError,"DaqSource::skip(int offset)")
        << "Random access skip cannot be used for DaqSource\n"
        << "Contact a Framework developer.\n";
  }

  void DaqSource::publish(xdata::InfoSpace *is)
  {
    is_ = is;
    is->fireItemAvailable("lumiSectionIndex", &lumiSectionIndex_);
    is->fireItemAvailable("prescaleSetIndex", &prescaleSetIndex_);
    is->fireItemAvailable("lsTimedOut",       &lsTimedOut_);
  }
  void DaqSource::publishToXmas(xdata::InfoSpace *is)
  {
    mis_ = is;
    is->fireItemAvailable("lumiSectionIndex", &lumiSectionIndex_);
    is->fireItemAvailable("prescaleSetIndex", &prescaleSetIndex_);
    is->fireItemAvailable("lsTimedOut",       &lsTimedOut_);
  }

  void DaqSource::openBackDoor(unsigned int timeout_sec)
  {
    count++;
    if(count==2) throw;
    pthread_mutex_lock(&mutex_);
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_sec;
    int rc = pthread_cond_timedwait(&cond_, &mutex_, &ts);
    if(rc == ETIMEDOUT) lsTimedOut_.value_ = true; 
  }
  
  void DaqSource::closeBackDoor()
  {
    count--;
    pthread_mutex_unlock(&mutex_);
    lsTimedOut_.value_ = false; 
    ::usleep(1000);
  }
}
