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
    virtual std::auto_ptr<EventPrincipal> read();

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
    std::auto_ptr<SendJobHeader> p = deserializeAndMergeWithRegistry(*header); 
    SendDescs const& descs = p->descs();
    // jbk - the next line should not be needed
    declareStreamers(descs);
    buildClassCache(descs);
    loadExtraClasses();
    saveTriggerNames(header);
  }

  template <typename Producer>
  std::auto_ptr<EventPrincipal> StreamerInputModule<Producer>::read() {

    EventMsgView const* eview = pr_->getNextEvent();

    if (pr_->newHeader()) {   
        FDEBUG(6) << "A new file has been opened and we must compare Headers here !!" << std::endl;
        // A new file has been opened and we must compare Heraders here !!
        //Get header/init from Producer
        InitMsgView const* header = pr_->getHeader();
        std::auto_ptr<SendJobHeader> p = deserializeAndMergeWithRegistry(*header);
        saveTriggerNames(header);
        if (!registryIsSubset(*p, *productRegistry())) {
            std::cout << "\n\nUn matching Init Message Headers found.\n";
            throw cms::Exception("read","StreamerInputModule")
                 << "Un matching Headers found.\n";
        }
    } 
    if (eview == 0) {
        return  std::auto_ptr<EventPrincipal>();
    }
    std::auto_ptr<EventPrincipal> pEvent(deserializeEvent(*eview));
    return pEvent;
  }

} // end of namespace-edm
  
#endif
