/** \file 
 *
 *  $Date: 2007/08/14 19:19:14 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - S. Argiro'
 */

#include "DaqSource.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "IORawData/DaqSource/interface/DaqBaseReader.h"
#include "IORawData/DaqSource/interface/DaqReaderPluginFactory.h"

#include "DataFormats/Provenance/interface/Timestamp.h" 
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

#include <string>
#include <sys/time.h>


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

namespace edm {

  //______________________________________________________________________________
  DaqSource::DaqSource(const ParameterSet& pset, 
		     const InputSourceDescription& desc) 
    : InputSource(pset,desc)
    , reader_(0)
    , lumiSegmentSizeInEvents_(pset.getUntrackedParameter<unsigned int>("evtsPerLS",0))
    , fakeLSid_(lumiSegmentSizeInEvents_ != 0)
    , remainingEvents_(maxEvents())
    , runNumber_(RunNumber_t())
    , luminosityBlockNumber_(LuminosityBlockNumber_t())
    , lbp_()
    , ep_() {
    produces<FEDRawDataCollection>();
    setTimestamp(Timestamp::beginOfTime());
    
    // Instantiate the requested data source
    std::string reader = pset.getUntrackedParameter<std::string>("readerPluginName");
    
    try{
      reader_=
        DaqReaderPluginFactory::get()->create(reader,
  					    pset.getUntrackedParameter<ParameterSet>("readerPset"));
    }
    catch(edm::Exception &e) {
      if(e.category() == "Configuration" && reader_ == 0) {
  	reader_ = DaqReaderPluginFactoryU::get()->create(reader);
  	if(reader_ == 0) throw;
      } else {
        throw;
      }
    }
  }
  
  //______________________________________________________________________________
  DaqSource::~DaqSource() {
    delete reader_;
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  // implementation of member functions
  ////////////////////////////////////////////////////////////////////////////////
  
  //______________________________________________________________________________
  std::auto_ptr<EventPrincipal> DaqSource::readOneEvent(boost::shared_ptr<RunPrincipal> rp) {
    if(reader_==0) {
      throw cms::Exception("LogicError")
        << "DaqSource is used without a reader. Check your configuration !";
    }
    EventID eventId;
    TimeValue_t time = 0LL;
    gettimeofday((timeval *)(&time),0);
    Timestamp tstamp(time);
    
    // pass a 0 pointer to fillRawData()!
    FEDRawDataCollection* fedCollection(0);
  
    // let reader_ fill the fedCollection 
    if(!reader_->fillRawData(eventId, tstamp, fedCollection)) {
      // fillRawData() failed, clean up the fedCollection in case it was allocated!
      if (0 != fedCollection) delete fedCollection;
      return std::auto_ptr<EventPrincipal>(0);
    }
    setTimestamp(tstamp);
    eventId = EventID(runNumber_, eventId.event());
    LuminosityBlockNumber_t newLsid = eventId.event()/lumiSegmentSizeInEvents_ + 1;
    if(fakeLSid_ && luminosityBlockNumber_ != newLsid) {
      this->setLumi(newLsid);
    }
    
    // If there is no luminosity block principal, make one.
    if (lbp_.get() == 0 || lbp_->luminosityBlock() != luminosityBlockNumber_) {
     lbp_ = boost::shared_ptr<LuminosityBlockPrincipal>(
        new LuminosityBlockPrincipal(luminosityBlockNumber_,
                                     timestamp(),
                                     Timestamp::invalidTimestamp(),
                                     productRegistry(),
                                     rp,
                                     processConfiguration()));

    }
    // make a brand new event
    ep_ = std::auto_ptr<EventPrincipal>(
	new EventPrincipal(eventId, timestamp(),
	productRegistry(), lbp_, processConfiguration(), true));
    
    // have fedCollection managed by a std::auto_ptr<>
    std::auto_ptr<FEDRawDataCollection> bare_product(fedCollection);

    std::auto_ptr<Event> e(new Event(*ep_, moduleDescription()));
    // put the fed collection into the transient event store
    e->put(bare_product);
    e->commit_();
    return ep_;
  }

  void
  DaqSource::setRun(RunNumber_t r) {
    // Do nothing if the run is not changed.
    if (r != runNumber_) {
      runNumber_ = r;
      lbp_.reset();
      ep_.reset();
    }
  }

  void
  DaqSource::setLumi(LuminosityBlockNumber_t lb) {
    if (lb != luminosityBlockNumber_) {
      luminosityBlockNumber_ = lb;
      lbp_.reset();
    }
  }

  boost::shared_ptr<RunPrincipal>
  DaqSource::readRun_() {
    return boost::shared_ptr<RunPrincipal>(
	new RunPrincipal(runNumber_,
			 timestamp(),
			 Timestamp::invalidTimestamp(),
			 productRegistry(),
			 processConfiguration()));
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  DaqSource::readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp) {
    if (ep_.get() == 0) ep_ = readOneEvent(rp);
    return ep_->luminosityBlockPrincipalSharedPtr();
  }

  std::auto_ptr<EventPrincipal>
  DaqSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    if (remainingEvents_ == 0) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    if (ep_.get() == 0) ep_ = readOneEvent(lbp->runPrincipalSharedPtr());
    if (ep_.get() == 0) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    if (lbp->luminosityBlock() != ep_->luminosityBlock()) {
      // We have advanced to the next luminosity block.
      // Return a null pointer, indicating no more events
      // in the luminosity block.
      return std::auto_ptr<EventPrincipal>(0);
    }
    --remainingEvents_;
    // Since ep_ is an auto_ptr, this return resets ep_ to null.
    return ep_;
  }

  std::auto_ptr<EventPrincipal>
  DaqSource::readIt(EventID const&) {
      throw cms::Exception("LogicError","DaqSource::readEvent_(EventID const& eventID)")
        << "Random access read cannot be used for DaqSource.\n"
        << "Contact a Framework developer.\n";
  }

  void
  DaqSource::skip(int) {
      throw cms::Exception("LogicError","DaqSource::skip(int offset)")
        << "Random access skip cannot be used for DaqSource\n"
        << "Contact a Framework developer.\n";
  }

}
