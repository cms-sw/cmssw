
#include "IOPool/Streamer/interface/TestFileReader.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include "boost/bind.hpp"

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
      ~BufHelper() { delete [] buf_; }
      char* release() { char* rc = buf_; buf_=0; return rc; }
      char* get() const { return buf_; }

    private:
      BufHelper(const BufHelper& ) { }
      BufHelper& operator=(const BufHelper&) { return *this; }

      char* buf_;
    };
  }

  // ----------------------------------

  TestFileReader::TestFileReader(const std::string& filename,
				 edm::EventBuffer& to,
				 edm::ProductRegistry& prods):
    filename_(filename),
    //ist_(filename_.c_str(),ios_base::binary | ios_base::in),
    //reader_(ist_),
    stream_reader_(new StreamerInputFile(filename)),
    to_(&to)
  {

   const InitMsgView* init =  stream_reader_->startMessage();
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

    if(edm::registryIsSubset(*p,prods)==false)
      {
	throw edm::Exception(errors::Configuration,"TestFileReader")
	  << "the header record in flie " << filename_
	  << "is not consistent with the one for the program \n";
      }
  }

  TestFileReader::~TestFileReader()
  {
     delete stream_reader_;
  }

  void TestFileReader::start()
  {
    me_.reset(new boost::thread(boost::bind(TestFileReader::run,this)));
  }

  void TestFileReader::join()
  {
    me_->join();
  }

  void TestFileReader::run(TestFileReader* t)
  {
    t->readEvents();
  }

  void TestFileReader::readEvents()
  {

   while( stream_reader_->next() ) { 
       EventBuffer::ProducerBuffer b(*to_);
       const EventMsgView* eview =  stream_reader_->currentRecord();
       stor::FragEntry* msg =
       //   new (b.buffer()) stor::FragEntry(eview->startAddress(),
       //                                    eview->startAddress(),
       // the first arg should be startAddress() right?
          //new (b.buffer()) stor::FragEntry((void*)eview->eventData(),
          new (b.buffer()) stor::FragEntry((void*)eview->startAddress(),
                                           (void*)eview->eventData(),
                                           eview->size(),1,1,
                                           eview->code(),0,1,0);
        assert(msg);
        b.commit(sizeof(stor::FragEntry));

   }

   /***
    while(1)
      {
	int len=0;
	ist_.read((char*)&len,sizeof(int));

	if(!ist_ || len==0 || ist_.eof()) break;

	EventBuffer::ProducerBuffer b(*to_);
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
