#ifndef _StreamerFileWriter_h
#define _StreamerFileWriter_h 

#include "IOPool/Streamer/interface/BufferArea.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/OutputModule.h"

#include "IOPool/Streamer/interface/StreamerFileIO.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>


class TClass;

namespace edm
{

  class StreamerFileWriter: public edm::OutputModule
  {

  public:

    typedef std::vector<char> ProdRegBuf;
    typedef OutputModule::Selections Selections;

    explicit StreamerFileWriter(edm::ParameterSet const& ps);
    virtual ~StreamerFileWriter();

  private:

    virtual void write(EventPrincipal const& e);
    virtual void beginJob(EventSetup const&);

    void stop();

    void serialize(EventPrincipal const& e);
    void serializeRegistry(Selections const& prods);

    void* registryBuffer() const { return (void*)&prod_reg_buf_[0]; }
    int registryBufferSize() const { return prod_reg_len_; }

    std::vector<std::string> getTriggerNames(); 
    //void getTriggerMask(EventPrincipal& ep);

  private:

    Selections const* selections_;

    vector<char> bufs_; 
    std::auto_ptr<StreamerOutputFile> stream_writer_;
    //StreamerOutputFile* stream_writer_;

    std::auto_ptr<StreamerOutputIndexFile> index_writer_; 
    //StreamerOutputIndexFile* index_writer_;

    TClass* tc_; // for SendEvent
    ProdRegBuf prod_reg_buf_;
    int prod_reg_len_;
    int hltsize_;

  };
}
#endif

