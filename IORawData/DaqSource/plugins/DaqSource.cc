/** \file 
 *
 *  $Date: 2012/11/28 19:17:36 $
 *  $Revision: 1.65 $
 *  \author N. Amapane - S. Argiro'
 */

#include "DaqSource.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"

#include "IORawData/DaqSource/interface/DaqBaseReader.h"
#include "IORawData/DaqSource/interface/DaqReaderPluginFactory.h"

#include "DataFormats/Provenance/interface/Timestamp.h" 
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include "boost/tokenizer.hpp"


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////



namespace edm {
 namespace daqsource{
  constexpr unsigned int gtpEvmId_ =  FEDNumbering::MINTriggerGTPFEDID;
  constexpr unsigned int gtpeId_ =  FEDNumbering::MINTriggerEGTPFEDID;
 }

  //______________________________________________________________________________
  DaqSource::DaqSource(const ParameterSet& pset, 
		     const InputSourceDescription& desc) 
    : InputSource(pset,desc)
    , evf::ModuleWeb("DaqSource")
    , reader_(0)
    , lumiSegmentSizeInEvents_(pset.getUntrackedParameter<unsigned int>("evtsPerLS",0))
    , lumiSegmentSizeInSeconds_(pset.getUntrackedParameter<unsigned int>("secondsPerLS",0))
    , useEventCounter_(pset.getUntrackedParameter<bool>("useEventCounter",false))
    , useTimer_(lumiSegmentSizeInSeconds_!=0)
    , eventCounter_(0)
    , keepUsingPsidFromTrigger_(pset.getUntrackedParameter<bool>("keepUsingPsidFromTrigger",false))
    , fakeLSid_(lumiSegmentSizeInEvents_ != 0 || lumiSegmentSizeInSeconds_ != 0)
    , runNumber_(pset.getUntrackedParameter<unsigned int>("runNumber",RunID::firstValidRun().run()))
    , luminosityBlockNumber_(LuminosityBlockID::firstValidLuminosityBlock().luminosityBlock())
    , daqProvenanceHelper_(TypeID(typeid(FEDRawDataCollection)))
    , noMoreEvents_(false)
    , alignLsToLast_(false)
    , lumiSectionIndex_(0)
    , prescaleSetIndex_(0)
    , lastLumiPrescaleIndex_(0)
    , lastLumiUsingEol_(0)
    , lsTimedOut_(0)
    , lsToBeRecovered_(0)
    , is_(0)
    , mis_(0)
    , thisEventLSid(0)
    , goToStopping(false)
    , immediateStop(false)
    , forkInfo_(nullptr)
    , runFork_(false)
    , beginRunTiming_(false)
    , bunchCrossing_(EventAuxiliary::invalidBunchXing)
    , orbitNumber_(EventAuxiliary::invalidBunchXing)
    , evttype_(EventAuxiliary::Undefined)
    , eventID_(0,0,0)
    , fedCollection_(nullptr)
  {
    count = 0;
    pthread_mutex_init(&mutex_,0);
    pthread_mutex_init(&signal_lock_,0);
    pthread_cond_init(&cond_,0);

    std::cout << " DaqSourceFake will use fake ls id ? " << fakeLSid_ << std::endl; 
    if(fakeLSid_) std::cout << " DaqSourceFake set to make a new ls every " << lumiSegmentSizeInEvents_ << " events " << std::endl;
    if(fakeLSid_) std::cout << " DaqSourceFake set to make a new ls every " << lumiSegmentSizeInSeconds_ << " seconds " << std::endl;

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

    // Initialize LS timer if requested
    if(lumiSegmentSizeInSeconds_ != 0) gettimeofday(&startOfLastLumi,0);

    // Initialize metadata, and save the process history ID for use every event.
    phid_ = daqProvenanceHelper_.daqInit(productRegistryUpdate());

  }
  
  //______________________________________________________________________________
  DaqSource::~DaqSource() {
    delete reader_;
  }
  
  void DaqSource::publishForkInfo(evf::moduleweb::ForkInfoObj * forkInfoObj) {
    forkInfo_ = forkInfoObj;
    runFork_=true;
    immediateStop=false;
    noMoreEvents_=false;
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // implementation of member functions
  ////////////////////////////////////////////////////////////////////////////////


  //______________________________________________________________________________
  int DaqSource::doMyBeginRun() {

    if (forkInfo_) {
      while (!immediateStop) {
	//queue new run to Framework (causes EP beginRun to be executed)
	if (newRun()) {
	  beginRunTiming_=true;
	  gettimeofday(&tvStat_, NULL);
	  return 2;
	}
	//measure time in fwk beginRun
	if (beginRunTiming_) {
	  timeval tsTmp;
	  gettimeofday(&tsTmp,NULL);
	  long tusecs = (tsTmp.tv_sec-tvStat_.tv_sec)*1000000 + tsTmp.tv_usec - tvStat_.tv_usec;
	  double tsecs = ((double)(tusecs/10000))/100.;
	  std::cout << "DaqSource: FWK beginRun elapsed time: " << tsecs << " seconds in master EP"<< std::endl;
	  edm::LogInfo("DaqSource") << "FWK beginRun elapsed time: " << tsecs << " seconds in master EP";
	  beginRunTiming_=false;
	  usleep(10000);//short sleep before fork
	}
	//first or new run init
	if (forkInfo_->forkParams.isMaster==-1) {
	  forkInfo_->lock();//keeping it locked during init!
	  forkInfo_->forkHandler(forkInfo_->fuAddr); //fork all slaves
        }
	if (forkInfo_->forkParams.isMaster==-1) {
	  forkInfo_->unlock();
	  std::cout << "ERROR (DaqSource): not notified to be either in  master or slave process after fork" << std::endl;
	  return -2;
	}

	//slave process after fork: exit all this
	if (forkInfo_->forkParams.isMaster==0) {
	  forkInfo_->unlock();
	  return 1;
	}

	//master process after fork:
	if (forkInfo_->forkParams.isMaster==1) {
	    forkInfo_->unlock();
	    int slotToRestart=-1;
	    sem_wait(forkInfo_->control_sem_);
	    forkInfo_->lock();

	    //got unblocked due to next run
	    if (forkInfo_->forkParams.isMaster==-1) {
	            forkInfo_->unlock();
		    continue; // check here for newRun()?
            }
	    //check if asked to stop
	    immediateStop=forkInfo_->stopCondition;
	    if (immediateStop) {
	      forkInfo_->receivedStop_=true;
	      break;
	    }
	    
	    //check if asked to restart
	    slotToRestart = forkInfo_->forkParams.slotId;

	  if (slotToRestart==-1 && forkInfo_->forkParams.restart==0) {
	    //this will deal with spurious semaphore signals when slave is killed
	    forkInfo_->unlock();
	    continue;
	  }
	  //restart single slave
	  forkInfo_->forkHandler(forkInfo_->fuAddr);
	}
      }
      //loop exit
      forkInfo_->unlock();
      return 0;
    }
    return -1; //no forkInfo_
  }


  //______________________________________________________________________________
  InputSource::ItemType 
  DaqSource::getNextItemType() {
    //    std::cout << getpid() << " enter getNextItemType " << std::endl;
    if (runFork_) {
      runFork_=false;
      int queueNext = doMyBeginRun();
      //check if new run (requires returning IsRun once)
      if (queueNext == 2) runFork_=true;
    }

    //get initial time before beginRun (used with old forking)
    if (!forkInfo_ && newRun()) {
      beginRunTiming_=true;
      gettimeofday(&tvStat_, NULL);
    }

    if (immediateStop) return IsStop;

    // --------------
    if(goToStopping){noMoreEvents_ = true; goToStopping=false;}
    if (noMoreEvents_) {
      pthread_mutex_lock(&mutex_);
      pthread_cond_signal(&cond_);
      pthread_mutex_unlock(&mutex_);
      return IsStop;
    }
    if (newRun()) {
      return IsRun;
    }

    //calculate and print the beginRun the timing
    if (beginRunTiming_) {
      timeval tsTmp;
      gettimeofday(&tsTmp,NULL);
      long tusecs = (tsTmp.tv_sec-tvStat_.tv_sec)*1000000 + tsTmp.tv_usec - tvStat_.tv_usec;
      double tsecs = ((double)(tusecs/10000))/100.;
      std::cout << "DaqSource (slave pid "<< getpid() << " ): FWK beginRun elapsed time: " 
		<< tsecs << " seconds "<< std::endl;
      edm::LogInfo("DaqSource") << "DaqSource (slave pid "<< getpid() << " ): FWK beginRun elapsed time: " 
		<< tsecs << " seconds ";
      beginRunTiming_=false;
    }

    if (newLumi() && luminosityBlockAuxiliary()) {
      //      std::cout << "newLumi & lumiblock valid " << std::endl;
      return IsLumi;
    }
    if (alignLsToLast_) { //here we are recovering from a gap in Ls number so an event may already be cached but 
      // we hold onto it until we have issued all the necessary endLumi/beginLumi
      //       std::cout << getpid() << "alignLsToLast was set and ls number is " 
      // 		<< luminosityBlockNumber_ << " before signaling" << std::endl;
      signalWaitingThreadAndBlock();
      luminosityBlockNumber_++;
      //       std::cout << getpid() << "alignLsToLast signaled and incremented " 
      // 		<< luminosityBlockNumber_ << " eventcached " 
      // 		<< eventCached() << std::endl;
      setNewLumi();
      if (lumiSectionIndex_!=0)
        lumiSectionIndex_->value_ = luminosityBlockNumber_;
      resetLuminosityBlockAuxiliary();
      if(luminosityBlockNumber_ == thisEventLSid+1) 
      {
        alignLsToLast_ = false;
      }
      if (!luminosityBlockAuxiliary() || luminosityBlockAuxiliary()->luminosityBlock() != luminosityBlockNumber_) {
	setLuminosityBlockAuxiliary(new LuminosityBlockAuxiliary(
	      runNumber_, luminosityBlockNumber_, timestamp(), Timestamp::invalidTimestamp()));
	luminosityBlockAuxiliary()->setProcessHistoryID(phid_);

	//      std::cout << "nextItemType: dealt with new lumi block principal, retval is " << retval << std::endl;
      }
      return IsLumi;
    }
    if (eventCached()) {
      //      std::cout << "read event already cached " << std::endl;
      return IsEvent;
    }

    if(reader_ == 0) {
      throw edm::Exception(errors::LogicError)
	  << "DaqSource is used without a reader. Check your configuration !";
    }
    TimeValue_t time = 0LL;
    timeval stv;
    gettimeofday(&stv,0);
    time = stv.tv_sec;
    time = (time << 32) + stv.tv_usec;
    Timestamp tstamp(time);

    bunchCrossing_ = EventAuxiliary::invalidBunchXing;
    orbitNumber_   = EventAuxiliary::invalidBunchXing;
    evttype_ = EventAuxiliary::Undefined;
    
    // pass a null pointer to fillRawData()!
    fedCollection_ = nullptr ;

  
    // let reader_ fill the fedCollection 
    int retval = reader_->fillRawData(eventID_, tstamp, fedCollection_);
    if(retval==0) {
      // fillRawData() failed, clean up the fedCollection in case it was allocated!
      if (nullptr != fedCollection_) delete fedCollection_;
      noMoreEvents_ = true;
      pthread_mutex_lock(&mutex_);
      pthread_cond_signal(&cond_);
      pthread_mutex_unlock(&mutex_);
      return IsStop;
    }
    else if(retval<0)
      {
 
	unsigned int nextLsFromSignal = (-1)*retval+1;
// 	std::cout << getpid() << "::got end-of-lumi for " << (-1)*retval
// 		  << " was " << luminosityBlockNumber_ << std::endl;
	if(luminosityBlockNumber_ == (nextLsFromSignal-1) )
	  {
	    if (lastLumiUsingEol_ != 0)
	      lastLumiUsingEol_->value_ = nextLsFromSignal;
	    if(lsToBeRecovered_ != 0 && lsToBeRecovered_->value_){
// 	      std::cout << getpid() << "eol::recover ls::for " << (-1)*retval << std::endl;
	      signalWaitingThreadAndBlock();
	      luminosityBlockNumber_++;
	      setNewLumi();
	      if (lumiSectionIndex_ != 0)
	        lumiSectionIndex_->value_ = luminosityBlockNumber_;
	      resetLuminosityBlockAuxiliary();
	      thisEventLSid = nextLsFromSignal - 1;
	      if(luminosityBlockNumber_ != thisEventLSid+1) 
		alignLsToLast_ = true;
	      //	      std::cout << getpid() << "eol::::alignLsToLast_ " << alignLsToLast_ << std::endl;
	    }
	    else{
	      //	      std::cout << getpid() << "eol::realign ls::for " << (-1)*retval << std::endl;
	      luminosityBlockNumber_ = nextLsFromSignal;
	      setNewLumi();
	      if (lumiSectionIndex_ != 0)
	        lumiSectionIndex_->value_ = luminosityBlockNumber_;
	      resetLuminosityBlockAuxiliary();
	    }
	  }
	else {
	  if(nextLsFromSignal >(luminosityBlockNumber_+100) ) {
	    edm::LogError("DaqSource") << "Got EOL event with value " << retval 
				       << " nextLS would be " << nextLsFromSignal 
				       << " while we expected " << luminosityBlockNumber_+1 << " - disregarding... ";
	  }
	  if (nextLsFromSignal > luminosityBlockNumber_+2) //recover on delta > 2
	  {
	      if (lastLumiUsingEol_ != 0)
	        lastLumiUsingEol_->value_ = nextLsFromSignal;
              thisEventLSid=nextLsFromSignal-1;//set new LS
	      signalWaitingThreadAndBlock();
	      luminosityBlockNumber_++;
	      setNewLumi();
	      if (lumiSectionIndex_ != 0)
	      lumiSectionIndex_->value_ = luminosityBlockNumber_;
	      alignLsToLast_ = true;

	      //set new lumi block
	      resetLuminosityBlockAuxiliary();
	      setLuminosityBlockAuxiliary(new LuminosityBlockAuxiliary(
	        runNumber_, luminosityBlockNumber_, timestamp(), Timestamp::invalidTimestamp()));
	      luminosityBlockAuxiliary()->setProcessHistoryID(phid_);
	  }

	}
	//	else
	//	  std::cout << getpid() << "::skipping end-of-lumi for " << (-1)*retval << std::endl;
      }
    else
      {
	if (eventID_.event() == 0) {
	  throw edm::Exception(errors::LogicError)
	    << "The reader used with DaqSource has returned an invalid (zero) event number!\n"
	    << "Event numbers must begin at 1, not 0.";
	}
	EventSourceSentry(*this);
	setTimestamp(tstamp);
    
	unsigned char *gtpFedAddr = fedCollection_->FEDData(daqsource::gtpEvmId_).size()!=0 ? fedCollection_->FEDData(daqsource::gtpEvmId_).data() : 0;
	uint32_t gtpsize = 0;
	if(gtpFedAddr !=0) gtpsize = fedCollection_->FEDData(daqsource::gtpEvmId_).size();
	unsigned char *gtpeFedAddr = fedCollection_->FEDData(daqsource::gtpeId_).size()!=0 ? fedCollection_->FEDData(daqsource::gtpeId_).data() : 0; 

	unsigned int nextFakeLs	= 0;
	eventCounter_++;
	if (fakeLSid_)
	    evttype_ =  edm::EventAuxiliary::PhysicsTrigger;
	bool fakeLSFromEventCount = fakeLSid_ && (lumiSegmentSizeInEvents_ != 0);
	bool fakeLSFromTimer = fakeLSid_ && (lumiSegmentSizeInSeconds_ != 0);
	if(fakeLSFromEventCount && luminosityBlockNumber_ != 
	   (nextFakeLs = useEventCounter_ ? ((eventCounter_-1)/lumiSegmentSizeInEvents_ + 1) :
	    ((eventID_.event() - 1)/lumiSegmentSizeInEvents_ + 1))) {
	    if (prescaleSetIndex_ != 0 && lastLumiPrescaleIndex_!= 0) {
	      lastLumiPrescaleIndex_->value_ = prescaleSetIndex_->value_;
	      prescaleSetIndex_->value_ = 0; // since we do not know better but we want to be able to run
	    }
	  if(luminosityBlockNumber_ == nextFakeLs-1)
	    signalWaitingThreadAndBlock();
	  luminosityBlockNumber_ = nextFakeLs;
	  thisEventLSid = nextFakeLs-1;
	  setNewLumi();
	  if (lumiSectionIndex_ != 0)
	    lumiSectionIndex_->value_ = luminosityBlockNumber_;
	  resetLuminosityBlockAuxiliary();
	  if(keepUsingPsidFromTrigger_ && 
	     gtpFedAddr!=0 && evf::evtn::evm_board_sense(gtpFedAddr,gtpsize)){
	    if (prescaleSetIndex_ != 0)
	      prescaleSetIndex_->value_  = (evf::evtn::getfdlpsc(gtpFedAddr) & 0xffff);
	  }	  
	}
	else if(fakeLSFromTimer){
	  struct timeval tv;
	  gettimeofday(&tv,0);
	  unsigned int elapsed_time = tv.tv_sec - startOfLastLumi.tv_sec;
	  //	  std::cout << "daqsource elapsed time " << elapsed_time << std::endl;
	  if(luminosityBlockNumber_ != 
	     (nextFakeLs = elapsed_time/lumiSegmentSizeInSeconds_ + 1)) {
	    if(prescaleSetIndex_ != 0){
	      lastLumiPrescaleIndex_->value_ = prescaleSetIndex_->value_;
	      prescaleSetIndex_->value_ = 0; // since we do not know better but we want to be able to run
	    }
	    if(luminosityBlockNumber_ == (nextFakeLs-1))
	      signalWaitingThreadAndBlock();
	    luminosityBlockNumber_ = nextFakeLs;
	    thisEventLSid = nextFakeLs-1;
	    setNewLumi();
	    if(lumiSectionIndex_!=0)
	      lumiSectionIndex_->value_ = luminosityBlockNumber_;
	    resetLuminosityBlockAuxiliary();
	    if(keepUsingPsidFromTrigger_ && 
	       gtpFedAddr!=0 && evf::evtn::evm_board_sense(gtpFedAddr,gtpsize)){
	      if(prescaleSetIndex_ != 0)
		prescaleSetIndex_->value_  = (evf::evtn::getfdlpsc(gtpFedAddr) & 0xffff);
	    }	  
	  }
	} 
	else if(!fakeLSid_){ 

	  if(gtpFedAddr!=0 && evf::evtn::evm_board_sense(gtpFedAddr,gtpsize)){
	    if (lastLumiPrescaleIndex_ != 0 && prescaleSetIndex_ != 0)
	      lastLumiPrescaleIndex_->value_ = prescaleSetIndex_->value_;
	    thisEventLSid = evf::evtn::getlbn(gtpFedAddr);
	    if (prescaleSetIndex_ != 0)
	      prescaleSetIndex_->value_  = (evf::evtn::getfdlpsc(gtpFedAddr) & 0xffff);
	    evttype_ =  edm::EventAuxiliary::ExperimentType(evf::evtn::getevtyp(gtpFedAddr));
	    if(luminosityBlockNumber_ > (thisEventLSid + 1))
	    {
	      //late event,throw fwk exception
	      std::ostringstream excptmsg;
	      excptmsg << "DaqSource::event with late LS (" << thisEventLSid + 1 << ")received.";
              throw edm::Exception(errors::LogicError,excptmsg.str());
	    }
	    if(luminosityBlockNumber_ != (thisEventLSid + 1)){
	      // we got here in a running process and some Ls might have been skipped so set the flag, 
	      // increase by one, check and if appropriate set the flag then continue
	      if(lsToBeRecovered_!=0 && lsToBeRecovered_->value_){
		//		std::cout << getpid() << "eve::recover ls::for " << thisEventLSid << std::endl;
		signalWaitingThreadAndBlock();
		luminosityBlockNumber_++;
		setNewLumi();
		if (lumiSectionIndex_ !=0 )
		  lumiSectionIndex_->value_ = luminosityBlockNumber_;
		resetLuminosityBlockAuxiliary();
		if(luminosityBlockNumber_ != thisEventLSid+1) alignLsToLast_ = true;
		//		std::cout << getpid() << "eve::::alignLsToLast_ " << alignLsToLast_ << std::endl;
	      }
	      else{ // we got here because the process was restarted. just realign the ls id and proceed with this event
		//		std::cout << getpid() << "eve::realign ls::for " << thisEventLSid << std::endl;
		luminosityBlockNumber_ = thisEventLSid + 1;
		setNewLumi();
		if (lumiSectionIndex_ !=0)
		  lumiSectionIndex_->value_ = luminosityBlockNumber_;
		resetLuminosityBlockAuxiliary();
		if (lsToBeRecovered_ !=0)
		  lsToBeRecovered_->value_ = true;
	      }
	    }
	  }
	  else if(gtpeFedAddr!=0 && evf::evtn::gtpe_board_sense(gtpeFedAddr)){
	    if (lastLumiPrescaleIndex_ != 0 && prescaleSetIndex_ != 0)
	      lastLumiPrescaleIndex_->value_ = prescaleSetIndex_->value_;
	    thisEventLSid = evf::evtn::gtpe_getlbn(gtpeFedAddr);
	    if (prescaleSetIndex_ != 0)
	      prescaleSetIndex_->value_ = 0; //waiting to get a PS index from gtpe
	    evttype_ =  edm::EventAuxiliary::PhysicsTrigger; 
	    if(luminosityBlockNumber_ != (thisEventLSid + 1)){
	      if(luminosityBlockNumber_ == thisEventLSid)
		signalWaitingThreadAndBlock();
	      luminosityBlockNumber_ = thisEventLSid + 1;
	      setNewLumi();
	      if (lumiSectionIndex_ !=0 )
	        lumiSectionIndex_->value_ = luminosityBlockNumber_;
	      resetLuminosityBlockAuxiliary();
	    }
	  }
	}
	if(gtpFedAddr!=0 && evf::evtn::evm_board_sense(gtpFedAddr,gtpsize)){
	  bunchCrossing_ =  int(evf::evtn::getfdlbx(gtpFedAddr));
	  orbitNumber_ =  int(evf::evtn::getorbit(gtpFedAddr));
	  TimeValue_t time = evf::evtn::getgpshigh(gtpFedAddr);
	  time = (time << 32) + evf::evtn::getgpslow(gtpFedAddr);
	  Timestamp tstamp(time);
	  setTimestamp(tstamp);      
	}
	else if(gtpeFedAddr!=0 && evf::evtn::gtpe_board_sense(gtpeFedAddr)){
	  bunchCrossing_ =  int(evf::evtn::gtpe_getbx(gtpeFedAddr));
	  orbitNumber_ =  int(evf::evtn::gtpe_getorbit(gtpeFedAddr));
	}
      }    
          
    //    std::cout << "lumiblockaux = " << luminosityBlockAuxiliary() << std::endl;
    // If there is no luminosity block principal, make one.
    if (!luminosityBlockAuxiliary() || luminosityBlockAuxiliary()->luminosityBlock() != luminosityBlockNumber_) {
      setNewLumi();
      setLuminosityBlockAuxiliary(new LuminosityBlockAuxiliary(
	runNumber_, luminosityBlockNumber_, timestamp(), Timestamp::invalidTimestamp()));
      luminosityBlockAuxiliary()->setProcessHistoryID(phid_);

      //      std::cout << "nextItemType: dealt with new lumi block principal, retval is " << retval << std::endl;
    }
    //    std::cout << "here retval = " << retval << std::endl;
    if(retval<0){
      //      std::cout << getpid() << " returning from getnextitem because retval < 0 - IsLumi "
      //		<< IsLumi << std::endl;
      if(newLumi()) return IsLumi; else return getNextItemType();
    }
    if (newLumi()) {
      return IsLumi;
    }
    setEventCached();
    return IsEvent;
  }

  void
  DaqSource::setRun(RunNumber_t r) {
    assert(!eventCached());
    reset();
    setNewRun();
    runNumber_ = r;
    if (reader_) reader_->setRunNumber(runNumber_);
    noMoreEvents_ = false;
    resetLuminosityBlockAuxiliary();
  }

  boost::shared_ptr<RunAuxiliary>
  DaqSource::readRunAuxiliary_() {
    assert(newRun());
    assert(!noMoreEvents_);
    resetNewRun();
    boost::shared_ptr<RunAuxiliary> ra(new RunAuxiliary(runNumber_, timestamp(), Timestamp::invalidTimestamp()));
    ra->setProcessHistoryID(phid_);
    return ra;
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  DaqSource::readLuminosityBlockAuxiliary_() {
    assert(!newRun());
    assert(newLumi());
    assert(!noMoreEvents_);
    assert(luminosityBlockAuxiliary());
    resetNewLumi();
    return luminosityBlockAuxiliary();
  }

  EventPrincipal*
  DaqSource::readEvent_(EventPrincipal& eventPrincipal) {
    //    std::cout << "assert not newRun " << std::endl;
    assert(!newRun());
    //    std::cout << "assert not newLumi " << std::endl;
    assert(!newLumi());
    //    std::cout << "assert not noMoreEvents " << std::endl;
    assert(!noMoreEvents_);
    //    std::cout << "assert eventCached " << std::endl;
    assert(eventCached());
    //    std::cout << "asserts done " << std::endl;

    resetEventCached();

    // make a brand new event principal
    eventID_ = EventID(runNumber_,thisEventLSid+1, eventID_.event());
    EventAuxiliary eventAux(eventID_, processGUID(),
			    timestamp(),
			    true,
			    evttype_,
			    bunchCrossing_,
			    EventAuxiliary::invalidStoreNumber,
			    orbitNumber_);
    eventAux.setProcessHistoryID(phid_);
    eventPrincipal.fillEventPrincipal(eventAux, boost::shared_ptr<LuminosityBlockPrincipal>());
    
    // have fedCollection managed by a std::auto_ptr<>
    std::auto_ptr<FEDRawDataCollection> bare_product(fedCollection_);

    WrapperOwningHolder edp(new Wrapper<FEDRawDataCollection>(bare_product), Wrapper<FEDRawDataCollection>::getInterface());
    eventPrincipalCache()->put(daqProvenanceHelper_.constBranchDescription_, edp, daqProvenanceHelper_.dummyProvenance_);

/*
    Event e(*eventPrincipalCache(), md_);
    // put the fed collection into the transient event store
    e.put(bare_product);
    // The commit is needed to complete the "put" transaction.
    e.commit_();
*/

    eventPrincipal.setLuminosityBlockPrincipal(luminosityBlockPrincipal());
    return &eventPrincipal;
  }

  void
  DaqSource::setLumi(LuminosityBlockNumber_t) {
      throw edm::Exception(errors::LogicError,"DaqSource::setLumi(LuminosityBlockNumber_t lumiNumber)")
        << "The luminosity block number cannot be set externally for DaqSource.\n"
        << "Contact a Framework developer.\n";
  }

  EventPrincipal*
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
    lumiSectionIndex_      = (xdata::UnsignedInteger32*)is_->find("lumiSectionIndex");
    prescaleSetIndex_      = (xdata::UnsignedInteger32*)is_->find("prescaleSetIndex");
    lastLumiPrescaleIndex_ = (xdata::UnsignedInteger32*)is_->find("lastLumiPrescaleIndex");
    lastLumiUsingEol_ = (xdata::UnsignedInteger32*)is_->find("lastLumiUsingEol");
    lsTimedOut_            = (xdata::Boolean*)is_->find("lsTimedOut");
    lsToBeRecovered_       = (xdata::Boolean*)is_->find("lsToBeRecovered");
  }
  void DaqSource::publishToXmas(xdata::InfoSpace *is)
  {
    mis_ = is;
  }

  void DaqSource::openBackDoor(unsigned int timeout_sec, bool *running)
  {
    count++;
    if(count==2) throw;
    pthread_mutex_lock(&mutex_);
    if (running) *running=true;
    pthread_mutex_unlock(&signal_lock_);
    timespec ts;
#if _POSIX_TIMERS > 0
    clock_gettime(CLOCK_REALTIME, &ts);
#else
    struct timeval tv; 
    gettimeofday(&tv, NULL);
    ts.tv_sec = tv.tv_sec + 0;
    ts.tv_nsec = 0;
#endif
    ts.tv_sec += timeout_sec;

    int rc = pthread_cond_timedwait(&cond_, &mutex_, &ts);
    if(rc == ETIMEDOUT && lsTimedOut_ != 0) lsTimedOut_->value_ = true; 
  }
  
  void DaqSource::closeBackDoor()
  {
    count--;
    pthread_cond_signal(&cond_);
    pthread_mutex_unlock(&mutex_);
    pthread_mutex_lock(&signal_lock_);
    if (lsTimedOut_ != 0)
      lsTimedOut_->value_ = false; 
  }

  void DaqSource::signalWaitingThreadAndBlock()
  {
    //check if we are running offline, in which case return immediately
    if(is_ == 0) return;
    pthread_mutex_lock(&signal_lock_);
    pthread_mutex_lock(&mutex_);
    pthread_mutex_unlock(&signal_lock_);
    //    std::cout << getpid() << " DS::signal from evloop " << std::endl;
    pthread_cond_signal(&cond_);
    //    std::cout << getpid() << " DS::go to wait for scalers wl " << std::endl;
    pthread_cond_wait(&cond_, &mutex_);
    pthread_mutex_unlock(&mutex_);
    ::usleep(1000);//allow other thread to lock
  }  

  void DaqSource::defaultWebPage(xgi::Input *in, xgi::Output *out)
  {
      std::string path;
      std::string urn;
      std::string mname;
      std::string query;
      std::string original_referrer_;
      try 
	{
	  cgicc::Cgicc cgi(in);
	  if ( xgi::Utils::hasFormElement(cgi,"gotostopping") )
	    {
	      goToStopping=true;
	    }
	  if ( xgi::Utils::hasFormElement(cgi,"module") )
	    mname = xgi::Utils::getFormElement(cgi, "module")->getValue();
	  cgicc::CgiEnvironment cgie(in);
	  if(original_referrer_ == "")
	    original_referrer_ = cgie.getReferrer();
	  path = cgie.getPathInfo();
	  query = cgie.getQueryString();
	}
      catch (const std::exception & e) 
	{
	  // don't care if it did not work
	}

      using std::endl;
      *out << "<html>"                                                   << endl;
      *out << "<head>"                                                   << endl;


      *out << "<STYLE type=\"text/css\"> #T1 {border-width: 2px; border: solid blue; text-align: center} </STYLE> "                                      << endl; 
      *out << "<link type=\"text/css\" rel=\"stylesheet\"";
      *out << " href=\"/" <<  urn
	   << "/styles.css\"/>"                   << endl;

      *out << "<title>" << moduleName_
	   << " MAIN</title>"                                            << endl;

      *out << "</head>"                                                  << endl;
      *out << "<body onload=\"loadXMLDoc()\">"                           << endl;
      *out << "<table border=\"0\" width=\"100%\">"                      << endl;
      *out << "<tr>"                                                     << endl;
      *out << "  <td align=\"left\">"                                    << endl;
      *out << "    <img"                                                 << endl;
      *out << "     align=\"middle\""                                    << endl;
      *out << "     src=\"/evf/images/bugicon.jpg\""	                 << endl;
      *out << "     alt=\"main\""                                        << endl;
      *out << "     width=\"90\""                                        << endl;
      *out << "     height=\"64\""                                       << endl;
      *out << "     border=\"\"/>"                                       << endl;
      *out << "    <b>"                                                  << endl;
      *out <<             moduleName_                                    << endl;
      *out << "    </b>"                                                 << endl;
      *out << "  </td>"                                                  << endl;
      *out << "  <td width=\"32\">"                                      << endl;
      *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
      *out << "      <img"                                               << endl;
      *out << "       align=\"middle\""                                  << endl;
      *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""             << endl;
      *out << "       alt=\"HyperDAQ\""                                  << endl;
      *out << "       width=\"32\""                                      << endl;
      *out << "       height=\"32\""                                     << endl;
      *out << "       border=\"\"/>"                                     << endl;
      *out << "    </a>"                                                 << endl;
      *out << "  </td>"                                                  << endl;
      *out << "  <td width=\"32\">"                                      << endl;
      *out << "  </td>"                                                  << endl;
      *out << "  <td width=\"32\">"                                      << endl;
      *out << "    <a href=\"" << original_referrer_  << "\">"           << endl;
      *out << "      <img"                                               << endl;
      *out << "       align=\"middle\""                                  << endl;
      *out << "       src=\"/evf/images/spoticon.jpg\""			 << endl;
      *out << "       alt=\"main\""                                      << endl;
      *out << "       width=\"32\""                                      << endl;
      *out << "       height=\"32\""                                     << endl;
      *out << "       border=\"\"/>"                                     << endl;
      *out << "    </a>"                                                 << endl;
      *out << "  </td>"                                                  << endl;
      *out << "</tr>"                                                    << endl;
      *out << "</table>"                                                 << endl;

      *out << "<hr/>"                                                    << endl;
  
      *out << cgicc::form().set("method","GET").set("action", path ) 
	   << std::endl;
      boost::char_separator<char> sep("&");
      boost::tokenizer<boost::char_separator<char> > tokens(query, sep);
      for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
	   tok_iter != tokens.end(); ++tok_iter){
	size_t pos = (*tok_iter).find_first_of("=");
	if(pos != std::string::npos){
	  std::string first  = (*tok_iter).substr(0    ,                        pos);
	  std::string second = (*tok_iter).substr(pos+1, (*tok_iter).length()-pos-1);
	  *out << cgicc::input().set("type","hidden").set("name",first).set("value", second) 
	       << std::endl;
	}
      }

      *out << cgicc::input().set("type","hidden").set("name","gotostopping").set("value","true")
	   << std::endl;
      *out << cgicc::input().set("type","submit").set("value","Go To Stopping")  	     << std::endl;
      *out << cgicc::form()						   << std::endl;  

      *out << "</body>"                                                  << endl;
      *out << "</html>"                                                  << endl;
  }
}
