/*----------------------------------------------------------------------

----------------------------------------------------------------------*/  

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "IOPool/Streamer/interface/TestFileReader.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "boost/bind.hpp"
#include "boost/shared_ptr.hpp"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

class Drain
{
 public:
  Drain();
  ~Drain();

  edm::EventBuffer& getQueue() { return q_; }

  void start();
  void join() { me_->join(); }

 private:
  static void run(Drain*);
  void readData();

  int count_;
  edm::EventBuffer q_;

  boost::shared_ptr<boost::thread> me_;
};

Drain::Drain():count_(),q_(sizeof(stor::FragEntry),2*3)
{
}

Drain::~Drain()
{
}

void Drain::start()
{
  me_.reset(new boost::thread(boost::bind(Drain::run,this)));
}

void Drain::run(Drain* d)
{
  std::cout << "Drain::run " << (void*)d << std::endl;
  d->readData();
}

void Drain::readData()
{
  while(1)
    {
      edm::EventBuffer::ConsumerBuffer b(q_);
      if(b.size()==0) break;

	  // std::cout << "Drain: " << (void*)this << " " << (void*)&q_ << std::endl;
      // std::cout << "Drain: woke up " << b.size() << std::endl;

      stor::FragEntry* v = (stor::FragEntry*)b.buffer();
      // std::cout << "Drain: cast frag " << b.buffer() << " " << v->buffer_size_ << std::endl;
      char* p = (char*)v->buffer_object_;
      // std::cout << "Drain: get frag " << (void*)p << std::endl;
      delete [] p;
      // std::cout << "Drain: delete frag " << b.size() << std::endl;
      ++count_;
    }

  std::cout << "Drain: got " << count_ << " events" << std::endl;
}

// -----------------------------------------------

class Main
{
 public:
  Main(const std::vector<std::string>& file_names);
  ~Main();
  
  int run();

 private:

  // disallow the following
  Main(const Main&) {} 
  Main& operator=(const Main&) { return *this; }

  std::vector<std::string> names_;
  edm::ProductRegistry prods_;
  Drain drain_;
  typedef boost::shared_ptr<edmtestp::TestFileReader> ReaderPtr;
  typedef std::vector<ReaderPtr> Readers;
  Readers readers_;
};

// ----------- implementation --------------

Main::~Main() { }


Main::Main(const std::vector<std::string>& file_names):
  names_(file_names),
  prods_(edm::getRegFromFile(file_names[0])),
  drain_()
{
  std::cout << "ctor of Main" << std::endl;
  // jbk - the next line should not be needed
  // edm::declareStreamers(prods_);
  std::vector<std::string>::iterator it(names_.begin()),en(names_.end());
  for(; it != en; ++it) {
      ReaderPtr p(new edmtestp::TestFileReader(*it,
					       drain_.getQueue(),
					       prods_));
      readers_.push_back(p);
  }
  edm::loadExtraClasses();
}

int Main::run()
{
  std::cout << "starting the drain" << std::endl;
  drain_.start();

  std::cout << "started the drain" << std::endl;
  // sleep(10);

  // start file readers
  Readers::iterator it(readers_.begin()),en(readers_.end());
  for(; it != en; ++it) {
      (*it)->start();
  }
  // wait for all file readers to complete
  for(it = readers_.begin(); it != en; ++it) {
      (*it)->join();
  }

  // send done to the drain
  edm::EventBuffer::ProducerBuffer b(drain_.getQueue());
  b.commit();

  drain_.join();
  return 0;
}

int main(int argc, char* argv[])
{
  // pull options out of command line
  if(argc < 2) {
      std::cout << "Usage: " << argv[0] << " "
	   << "file1 file2 ... fileN"
	   << std::endl;
      return 0;
      //throw cms::Exception("config") << "Bad command line arguments\n";
  }

  edmplugin::PluginManager::configure(edmplugin::standard::config());
  
  std::vector<std::string> file_names;
  
  for(int i = 1; i < argc; ++i) {
      std::cout << argv[i] << std::endl;
      file_names.push_back(argv[i]);
  }
  
  try {
    edm::loadExtraClasses();
    std::cout << "Done loading extra classes" << std::endl;
    Main m(file_names);
    m.run();
  }
  catch(cms::Exception& e) {
      std::cerr << "Caught an exception:\n" << e.what() << std::endl;
      throw;
  }
  catch(...) {
      std::cerr << "Caught unknown exception\n" << std::endl;
  }

  std::cout << "Main is done!" << std::endl;
  return 0;
}

