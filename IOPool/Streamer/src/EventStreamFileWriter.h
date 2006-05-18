#ifndef _EventStreamFileWriter_h
#define _EventStreamFileWriter_h 

#include "IOPool/Streamer/interface/BufferArea.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/OutputModule.h"


#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>

class TClass;

namespace edm
{

  class Worker;

  class EventStreamFileWriter: public edm::OutputModule
  {

  public:

    typedef std::vector<char> ProdRegBuf;
    typedef OutputModule::Selections Selections;

    explicit EventStreamFileWriter(edm::ParameterSet const& ps);
    virtual ~EventStreamFileWriter();

    virtual void write(EventPrincipal const& e);
    virtual void beginJob(EventSetup const&);

    void bufferReady();
    void stop();
    void sendRegistry(void* buf, int len);

  private:

    void serialize(EventPrincipal const& e);
    void serializeRegistry(Selections const& prods);

    void* registryBuffer() const { return (void*)&prod_reg_buf_[0]; }
    int registryBufferSize() const { return prod_reg_len_; }

  private:

    EventBuffer* bufs_;
    Worker* worker_;
    TClass* tc_; // for SendEvent
    ProdRegBuf prod_reg_buf_;
    int prod_reg_len_;
  };
}
#endif

