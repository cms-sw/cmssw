#ifndef _StreamerFileReader_H
#define _StreamerFileReader_H


#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "IOPool/Streamer/interface/StreamerFileIO.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/InputSource.h"

#include "TBuffer.h"
#include "TClass.h"

#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>

#include <typeinfo>

class TClass;

namespace edmtestp
{
  class StreamerFileReader : public edm::InputSource
  {
  public:
    StreamerFileReader(edm::ParameterSet const& pset,
		 edm::InputSourceDescription const& desc);
    virtual ~StreamerFileReader();

    virtual std::auto_ptr<edm::EventPrincipal> read();

  private:  

    std::auto_ptr<edm::SendJobHeader>  readHeader();
    std::auto_ptr<edm::EventPrincipal> readEvent(const edm::ProductRegistry& pr);

    void mergeWithRegistry(const edm::SendDescs& descs,
                         edm::ProductRegistry& reg);
    void declareStreamers(const edm::SendDescs& descs);
    void buildClassCache(const edm::SendDescs& descs);

    std::string filename_;
    StreamerInputFile* stream_reader_;    

    TClass* tc_;
    TClass* desc_;
  };

}

#endif

