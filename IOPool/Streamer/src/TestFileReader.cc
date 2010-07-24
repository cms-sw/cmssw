 
#include "IOPool/Streamer/interface/TestFileReader.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include "boost/bind.hpp"
#include "boost/shared_array.hpp"

#include <algorithm>
#include <cstdlib>
#include <iterator>

using namespace edm;

namespace edmtestp
{  

  namespace 
  {
    struct BufHelper
    {
      explicit BufHelper(int len): buf_(new char[len]) { }
      ~BufHelper() {}
      void release() {buf_.reset();}
      char* get() const { return buf_.get(); }

    private:
      BufHelper(const BufHelper& ) { }
      BufHelper& operator=(const BufHelper&) { return *this; }

      boost::shared_array<char> buf_;
    };
  }

  // ----------------------------------

  TestFileReader::TestFileReader(const std::string& filename,
				 edm::EventBuffer& to,
				 edm::ProductRegistry& prods):
    filename_(filename),
    //ist_(filename_.c_str(),ios_base::binary | ios_base::in),
    //reader_(ist_),
    streamReader_(new StreamerInputFile(filename)),
    to_(to) {

   const InitMsgView* init =  streamReader_->startMessage();
   std::auto_ptr<edm::SendJobHeader> p = StreamerInputSource::deserializeRegistry(*init);
    /**
    if(!ist_)
      {
	throw edm::Exception(errors::Configuration,"TestFileReader")
	  << "cannot open file " << filename_;
      }

    std::auto_ptr<SendJobHeader> p = readHeaderFromStream(ist_);
    // just get rid of the header

    **/

    if(edm::registryIsSubset(*p, prods) == false) {
	throw edm::Exception(errors::Configuration,"TestFileReader")
	  << "the header record in file " << filename_
	  << "is not consistent with the one for the program \n";
    }

    // 13-Oct-2008, KAB - Added the following code to put the 
    // INIT message on the input queue.
    EventBuffer::ProducerBuffer b(to_);
    int len = init->size();
    char* buf_ = new char[len];
    memcpy(buf_, init->startAddress(), len);
    new (b.buffer()) stor::FragEntry(buf_, buf_, len, 1, 1,
                                     init->code(), init->run(),
                                     0, init->outputModuleId(),
                                     getpid(), 0);
    b.commit(sizeof(stor::FragEntry));
  }

  TestFileReader::~TestFileReader() {
  }

  void TestFileReader::start() {
    me_.reset(new boost::thread(boost::bind(TestFileReader::run,this)));
  }

  void TestFileReader::join() {
    me_->join();
  }

  void TestFileReader::run(TestFileReader* t) {
    t->readEvents();
  }

  void TestFileReader::readEvents() {

   while(streamReader_->next()) { 
       EventBuffer::ProducerBuffer b(to_);
       const EventMsgView* eview = streamReader_->currentRecord();

       // 13-Oct-2008, KAB - we need to make a copy of the event message
       // for two reasons:  1) the processing of the events is often done
       // asynchronously in the code that uses this reader, so we can't
       // keep re-using the same buffer from the stream_reader, and
       // 2) the code that uses this reader often uses deleters to
       // free up the memory used by the FragEntry, so we want the
       // first argument to the FragEntry to be something that can 
       // be deleted successfully.
       int len = eview->size();
       char* buf_ = new char[len];
       memcpy(buf_, eview->startAddress(), len);

       //stor::FragEntry* msg =
       //   new (b.buffer()) stor::FragEntry(eview->startAddress(),
       //                                    eview->startAddress(),
       // the first arg should be startAddress() right?
          //new (b.buffer()) stor::FragEntry((void*)eview->eventData(),
       new (b.buffer()) stor::FragEntry(buf_, buf_, len, 1, 1,
                                        eview->code(), eview->run(),
                                        eview->event(), eview->outModId(),
                                        getpid(), 0);
       b.commit(sizeof(stor::FragEntry));
   }

   /***
    while(1)
      {
	int len=0;
	ist_.read((char*)&len,sizeof(int));

	if(!ist_ || len==0 || ist_.eof()) break;

	EventBuffer::ProducerBuffer b(to_);
	// Pay attention here.
	// This is a bit of a mess.
	// Here we allocate an array (on the heap), fill it with data
	// from the file, then pass the bare pointer off onto the queue.
	// The ownership is to be picked up by the code that pulls it
	// off the queue.  The current implementation of the queue
	// is primitive - it knows nothing about the types of things
	// that are on the queue.

	BufHelper data(len);
	//std::cout << "allocated frag " << len << std::endl;
	ist_.read((char*)data.get(),len);
	//std::cout << "read frag to " << (void*)data.get() << std::endl;
	if(!ist_ || ist_.eof()) std::cerr << "got end!!!!" << std::endl;
        //HEREHERE need a real event number here for id
	stor::FragEntry* msg = 
	  new (b.buffer()) stor::FragEntry(data.get(),data.get(),len,1,1,Header::EVENT,0,1,0);
        assert(msg); // Suppresses compiler warning about unused variable
	//new (b.buffer()) stor::FragEntry(0,0,len);
	//std::cout << "make entry for frag " << (void*)msg << " " << msg->buffer_address_ << std::endl;
	data.release();
	//std::cout << "release frag" << std::endl;
	b.commit(sizeof(stor::FragEntry));
	//std::cout << "commit frag " << sizeof(stor::FragEntry) << std::endl;
	//sleep(2);
      } **/
  }

}
