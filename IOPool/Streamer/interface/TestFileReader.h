#ifndef IOPool_Streamer_TestFileReader_h
#define IOPool_Streamer_TestFileReader_h

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "IOPool/Streamer/interface/StreamerInputFile.h"

#include "boost/shared_ptr.hpp"

#include <vector>
#include <memory>
#include <string>
#include <fstream>

namespace edmtestp {
  class TestFileReader {
  public:
    TestFileReader(std::string const& filename, edm::EventBuffer& to,
		   edm::ProductRegistry& prods);
    virtual ~TestFileReader();

    void start();
    void join();

  private:
    void readEvents();
    static void run(TestFileReader*);

    std::string filename_;
    boost::shared_ptr<edm::StreamerInputFile> streamReader_;
    //std::ifstream ist_;
    //edm::EventReader reader_;
    edm::EventBuffer& to_;
    boost::shared_ptr<boost::thread> me_;
  };

}

#endif

