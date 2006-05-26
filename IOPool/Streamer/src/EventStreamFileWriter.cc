#include "DataFormats/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/src/EventStreamFileWriter.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"

#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "boost/shared_ptr.hpp"

//#include "PluginManager/PluginCapabilities.h"

//#include "StorageSvc/IOODatabaseFactory.h"
#include "StorageSvc/IClassLoader.h"
#include "StorageSvc/DbType.h" 

#include "TBuffer.h"
#include "TClass.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <list> 
#include <cstring>
#include <fstream>
#include <sstream>

using namespace edm;
using namespace std;

namespace edm
{

  typedef boost::shared_ptr<ofstream> OutPtr;
  typedef vector<char> SaveArea;

  string makeFileName(const string& base, int num)
  {
    ostringstream ost;
    ost << base << num << ".dat";
    return ost.str();
  }

  OutPtr makeFile(const string name,int num)
  {
    OutPtr p(new ofstream(makeFileName(name,num).c_str(),
                          ios_base::binary | ios_base::out));

    if(!(*p))
      {
        throw cms::Exception("Configuration","TestConsumer")
          << "cannot open file " << name;
      }

    return p;
  }

  struct Worker
  {
    Worker(const string& s, int m);

    string filename_;
    int file_num_;
    int cnt_;
    int max_;
    OutPtr ost_;
    SaveArea reg_;

    void checkCount();
    void saveReg(void* buf, int len);
    void writeReg();
  };

  Worker::Worker(const string& s,int m):
    filename_(s),
    file_num_(),
    cnt_(0),
    max_(m),
    ost_(makeFile(filename_,file_num_))
  {
  }

  void Worker::checkCount()
  {
    if(cnt_!=0 && (cnt_%max_) == 0)
      {
        ++file_num_;
        ost_ = makeFile(filename_,file_num_);
        writeReg();
      }
    ++cnt_;

  }

  void Worker::writeReg()
  {
    if(!reg_.empty())
      {
        int len = reg_.size();
        ost_->write((const char*)(&len),sizeof(int));
        ost_->write((const char*)&reg_[0],len);
      }
  }

  void Worker::saveReg(void* buf, int len)
  {
    reg_.resize(len);
    memcpy(&reg_[0],buf,len);
  }


  // ----------------------------------

EventStreamFileWriter::EventStreamFileWriter(ParameterSet const& ps):
    OutputModule(ps),
    selections_(&descVec_),
    bufs_(getEventBuffer(ps.template getParameter<int>("max_event_size"),
                         ps.template getParameter<int>("max_queue_depth"))),
    worker_(new Worker(ps.template getParameter<string>("fileName"),
                       ps.template getUntrackedParameter<int>("numPerFile",1<<31))),
    tc_(),
    prod_reg_buf_(100 * 1000),
    prod_reg_len_()
     
  {
    FDEBUG(6) << "StreamOutput constructor" << endl;
    edm::loadExtraClasses();
    tc_ = getTClass(typeid(SendEvent));
  }


EventStreamFileWriter::~EventStreamFileWriter()
  {
    try {
      stop(); // should not throw !
      delete worker_;
    }
    catch(...)
      {
        std::cerr << "EventStreamingModule: stopping the consumer caused "
                  << "an exception!\n"
                  << "Igoring the exception." << std::endl;
      }

  }

void EventStreamFileWriter::stop()
  {
    EventBuffer::ProducerBuffer pb(*bufs_);
    pb.commit();
  }


std::vector<std::string> EventStreamFileWriter::getTriggerNames() {
  edm::Service<edm::service::TriggerNamesService> tns;
  std::vector<std::string> allTriggerNames = tns->getTrigPaths();
  //for (unsigned int i=0; i!=allTriggerNames.size() ;++i) 
        //cout<<"TriggerName: "<<allTriggerNames.at(i);
  return allTriggerNames;
  }


void EventStreamFileWriter::beginJob(EventSetup const&)
  {
    serializeRegistry(descVec_);
    sendRegistry(registryBuffer(), registryBufferSize());
  }

void EventStreamFileWriter::serializeRegistry(Selections const& prods)
  {
    FDEBUG(6) << "StreamOutput: serializeRegistry" << endl;
    TClass* prog_reg = getTClass(typeid(SendJobHeader));
    SendJobHeader sd;

    Selections::const_iterator i(prods.begin()),e(prods.end());

    FDEBUG(9) << "Product List: " << endl;
    for(;i!=e;++i) 
      {
	sd.descs_.push_back(**i);
	FDEBUG(9) << "StreamOutput got product = " << (*i)->fullClassName_
		  << endl;
      }

    /**********************************/

    uint32 run = 1;
    char psetid[] = "1234567890123456";
    Version v(2,(const uint8*)psetid);
    char release_tag[] = "CMSSW_DUMMY";
    Strings hlt_names = getTriggerNames();    
    Strings l1_names;
   
    InitMsgBuilder init_message(&prod_reg_buf_[0], prod_reg_buf_.size(),
                 run, v,
                 release_tag,
                 hlt_names,
                 l1_names);

    TBuffer rootbuf(TBuffer::kWrite,(int)prod_reg_buf_.size(),
                                    init_message.dataAddress(),kFALSE);

    //cout<<"\nInitMsgBuilder is being used here"<<endl;

    RootDebug tracer(10,10);

    int bres = rootbuf.WriteObjectAny((char*)&sd,prog_reg);
    
    switch(bres)
      {
      case 0: // failure
	{
	  throw cms::Exception("Output","SerializationReg")
	    << "EventStreamOutput module could not serialize event\n";
	  break;
	}
      case 1: // succcess
	break;
      case 2: // truncated result
	{
	  throw cms::Exception("Output","SerializationReg")
	    << "EventStreamOutput module attempted to serialize the registry\n"
	    << "that is to big for the allocated buffers\n";
	  break;
	}
      default: // unknown
	{
	  throw cms::Exception("Output","SerializationReg")
	    << "EventStreamOutput module got an unknown error code\n"
	    << " while attempting to serialize registry\n";
	  break;
	}
      }	    

    init_message.setDescLength(rootbuf.Length()); 
    //prod_reg_len_ = im.msgSize();
           //prod_reg_len_ = init_message.size();
    //cout<<"Verify if this is correct"<<endl; 
    prod_reg_len_ = prod_reg_buf_.size();
  }

void EventStreamFileWriter::sendRegistry(void* buf, int len)
  { 
    
    worker_->saveReg(buf,len);
    worker_->writeReg();
  }

void EventStreamFileWriter::write(EventPrincipal const& e)
  {
    serialize(e);
    bufferReady();
  }

void EventStreamFileWriter::serialize(EventPrincipal const& e)
  {
    std::list<Provenance> provenances; // Use list so push_back does not invalidate iterators.
    // all provenance data needs to be transferred, including the
    // indirect stuff referenced from the product provenance structure.
    SendEvent se(e.id(),e.time());

    Selections::const_iterator i(selections_->begin()),ie(selections_->end());
    // Loop over EDProducts, fill the provenance, and write.
    for(; i != ie; ++i) {
      BranchDescription const& desc = **i;
      ProductID const& id = desc.productID_;

      if (id == ProductID()) {
        throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
          << "EventStreamOutput::serialize: invalid ProductID supplied in productRegistry\n";
      }
      EventPrincipal::SharedGroupPtr const group = e.getGroup(id);
      if (group.get() == 0) {
        std::string const wrapperBegin("edm::Wrapper<");
        std::string const wrapperEnd1(">");
        std::string const wrapperEnd2(" >");
        std::string const& name = desc.fullClassName_;
        std::string const& wrapperEnd = (name[name.size()-1] == '>' ? wrapperEnd2 : wrapperEnd1);
        std::string const className = wrapperBegin + name + wrapperEnd;
        TClass *cp = gROOT->GetClass(className.c_str());
        EDProduct *p = static_cast<EDProduct *>(cp->New());

        Provenance prov(desc);
        prov.event.status = BranchEntryDescription::CreatorNotRun;
        prov.event.productID_ = id;
        provenances.push_back(prov);
        Provenance & provenance = *provenances.rbegin();
        se.prods_.push_back(ProdPair(p, &provenance));
      } else {
        se.prods_.push_back(ProdPair(group->product(), &group->provenance()));
      }
    }

#if 0
    FDEBUG(11) << "-----Dump start" << endl;
    for(SendProds::iterator pii=se.prods_.begin();pii!=se.prods_.end();++pii)
      std::cout << "Prov:"
	   << " " << pii->desc()->fullClassName_
	   << " " << pii->desc()->productID_
	   << endl;      
    FDEBUG(11) << "-----Dump end" << endl;
#endif

    EventBuffer::ProducerBuffer b(*bufs_);


    /*************************************/

    uint32 lumi=2;  //Dummy
    std::vector<bool> l1bits(16);
    uint8 hltbits[] = "4567";
    const int hltsize = (sizeof(hltbits)-1)*4;
    uint32 reserved=78; //dummy
    EventMsgBuilder msg(b.buffer(), b.size(),
                          e.id().run(), e.id().event(), lumi,
                          l1bits, hltbits, hltsize);
    msg.setReserved(reserved);  

    //cout<<"\nEventMsgBuilder is being used here"<<endl;

    TBuffer rootbuf(TBuffer::kWrite,b.size(),msg.eventAddr(),kFALSE);

    //EventMsg msg(b.buffer(),b.size(),e.id().event(),e.id().run(),1,1);
    //TBuffer rootbuf(TBuffer::kWrite,msg.getDataSize(),msg.data(),kFALSE);
    RootDebug tracer(10,10);

    int bres = rootbuf.WriteObjectAny(&se,tc_);
    
    switch(bres)
      {
      case 0: // failure
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module could not serialize event: "
	    << e.id();
	  break;
	}
      case 1: // succcess
	break;
      case 2: // truncated result
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module attempted to serialize an event\n"
	    << "that is to big for the allocated buffers: "
	    << e.id();
	  break;
	}
      default: // unknown
	{
	  throw cms::Exception("Output","Serialization")
	    << "EventStreamOutput module got an unknown error code\n"
	    << " while attempting to serialize event: "
	    << e.id();
	  break;
	}
      }
     
    msg.setEventLength(rootbuf.Length());
    b.commit(msg.size());

    //msg.setDataSize(rootbuf.Length());
    //b.commit(msg.msgSize());
  }

void EventStreamFileWriter::bufferReady()
  {
    worker_->checkCount();

    EventBuffer::ConsumerBuffer cb(*bufs_);

    int sz = cb.size();
    worker_->ost_->write((const char*)(&sz),sizeof(int));
    worker_->ost_->write((const char*)cb.buffer(),sz);
    //cout<<"Size of Buffer Just written: "<<sz<<endl;

  }

}
