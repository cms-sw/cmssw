#ifndef IOPool_Streamer_StreamerInputModule_h
#define IOPool_Streamer_StreamerInputModule_h

#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "IOPool/Streamer/interface/StreamerFileIO.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm> 
#include <iterator>

namespace edm
{
  template <typename Producer>
  class StreamerInputModule : public StreamerInputSource {
  /**
     Requires the Producer class to provide following functions
           const InitMsgView* getHeader();
           const EventMsgView* getNextEvent();
  */
  public:  
    explicit StreamerInputModule(ParameterSet const& pset,
                 InputSourceDescription const& desc);
    virtual ~StreamerInputModule();
    virtual EventPrincipal* read();

  private:
    //ProductRegistry const* prod_reg_;
    std::auto_ptr<Producer> pr_; 
  }; //end-of-class-def

  template <typename Producer>
  StreamerInputModule<Producer>::~StreamerInputModule() {}

  template <typename Producer>
  StreamerInputModule<Producer>::StreamerInputModule(
                    ParameterSet const& pset,
                    InputSourceDescription const& desc):
	StreamerInputSource(pset, desc),
	//prod_reg_(&productRegistry()), 
	pr_(new Producer(pset)) {
    //Get header/init from Producer
    InitMsgView const* header = pr_->getHeader();
    deserializeAndMergeWithRegistry(*header); 
  }

  template <typename Producer>
  EventPrincipal* StreamerInputModule<Producer>::read() {

    EventMsgView const* eview = pr_->getNextEvent();

    if (pr_->newHeader()) {   
        FDEBUG(6) << "A new file has been opened and we must compare Headers here !!" << std::endl;
        // A new file has been opened and we must compare Headers here !!
        //Get header/init from Producer
        InitMsgView const* header = pr_->getHeader();
        deserializeAndMergeWithRegistry(*header, true);
    } 
    if (eview == 0) {
        return  0;
    }
    return(deserializeEvent(*eview));
  }

} // end of namespace-edm
  
#endif
