#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Framework/interface/Event.h"

#include "IOPool/Streamer/src/StreamerFileWriter.h"

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

StreamerFileWriter::StreamerFileWriter(ParameterSet const& ps):
    OutputModule(ps),
    selections_(&descVec_),
    //bufs_(getEventBuffer(ps.template getParameter<int>("max_event_size"),
    //                     ps.template getParameter<int>("max_queue_depth"))),
    stream_writer_(new StreamerOutputFile(ps.template getParameter<string>("fileName"))),
    index_writer_(new StreamerOutputIndexFile(
                                ps.template getParameter<string>("indexFileName"))), 
    tc_(),
    prod_reg_buf_(100 * 1000),
    prod_reg_len_()
     
  {
    bufs_.resize(ps.template getParameter<int>("max_event_size"));
    FDEBUG(6) << "StreamOutput constructor" << endl;
    edm::loadExtraClasses();
    tc_ = getTClass(typeid(SendEvent));
  }


StreamerFileWriter::~StreamerFileWriter()
  {
    try {
      stop(); // should not throw !
    }
    catch(...)
      {
        std::cerr << "EventStreamingModule: stopping the consumer caused "
                  << "an exception!\n"
                  << "Igoring the exception." << std::endl;
      }

  }

void StreamerFileWriter::stop()
  {
    //EventBuffer::ProducerBuffer pb(*bufs_);
    //pb.commit();

    //Write the EOF Record Both at the end of Streamer file and Index file
    uint32 dummyStatusCode = 1234;
    std::vector<uint32> hltStats;

    hltStats.push_back(32);
    hltStats.push_back(33);
    hltStats.push_back(34);

    stream_writer_->writeEOF(dummyStatusCode, hltStats);
    index_writer_->writeEOF(dummyStatusCode, hltStats);

    //delete index_writer_; 
    //delete stream_writer_;
    cout<<"DONE STOP!"<<endl;
  }

// This functionality can actullay be provided by OutputModule
// Its moved here as it is not currently there.
std::vector<std::string> StreamerFileWriter::getTriggerNames() {
  edm::Service<edm::service::TriggerNamesService> tns;
  std::vector<std::string> allTriggerNames = tns->getTrigPaths();

  //for (unsigned int i=0; i!=allTriggerNames.size() ;++i) 
        //cout<<"TriggerName: "<<allTriggerNames.at(i);

  hltsize_ = allTriggerNames.size(); 
  return allTriggerNames;
  }

void StreamerFileWriter::beginJob(EventSetup const&)
  {
    serializeRegistry(descVec_);
  }

void StreamerFileWriter::serializeRegistry(Selections const& prods)
  {
    FDEBUG(6) << "StreamOutput: serializeRegistry" << endl;
    TClass* prog_reg = getTClass(typeid(SendJobHeader));
    SendJobHeader sd;

    Selections::const_iterator i(prods.begin()),e(prods.end());

    FDEBUG(9) << "Product List: " << endl;
    for(;i!=e;++i) 
      {
	sd.descs_.push_back(**i);
	FDEBUG(9) << "StreamOutput got product = " << (*i)->className()
		  << endl;
      }

    //Following values are strictly DUMMY and will be replaced
    // once available  
    uint32 run = 1;
    char psetid[] = "1234567890123456";
    Version v(2,(const uint8*)psetid);
    char release_tag[] = "CMSSW_DUMMY";
    Strings hlt_names; //9
    //Strings hlt_names = getTriggerNames();    
    hlt_names.push_back("a");  hlt_names.push_back("b");
    hlt_names.push_back("c");  hlt_names.push_back("d");
    hlt_names.push_back("e");  hlt_names.push_back("f");
    hlt_names.push_back("g");  hlt_names.push_back("h");
    hlt_names.push_back("i");
    Strings l1_names;  //11
    l1_names.push_back("t1");  l1_names.push_back("t10");
    l1_names.push_back("t2");  l1_names.push_back("t3");
    l1_names.push_back("t4");  l1_names.push_back("t5");
    l1_names.push_back("t6");  l1_names.push_back("t7");
    l1_names.push_back("t8");  l1_names.push_back("t9");
    l1_names.push_back("t11");

    InitMsgBuilder init_message(&prod_reg_buf_[0], prod_reg_buf_.size(),
                 run, v,
                 release_tag,
                 hlt_names,
                 l1_names);

    TBuffer rootbuf(TBuffer::kWrite,(int)prod_reg_buf_.size(),
                                    init_message.dataAddress(),kFALSE);

    cout<<"\nInitMsgBuilder is being used here"<<endl;

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
    prod_reg_len_ = prod_reg_buf_.size();

    cout<<"init_message.size: "<<init_message.size()<<endl;
    cout<<"init_message.run: "<<init_message.run()<<endl;

    //Write the Init Message to Streamer file
    stream_writer_->write(init_message); 
    cout<<"Just wrote init_message"<<endl;

    uint32 magic = 22;
    uint64 reserved = 666;
    index_writer_->writeIndexFileHeader(magic, reserved);
    index_writer_->write(init_message);
  }

void StreamerFileWriter::write(EventPrincipal const& e)
  {
    serialize(e);
  }

void StreamerFileWriter::serialize(EventPrincipal const& e)
  {
    std::list<Provenance> provenances; // Use list so push_back does not invalidate iterators.
    // all provenance data needs to be transferred, including the
    // indirect stuff referenced from the product provenance structure.
    SendEvent se(e.id(),e.time());

    Selections::const_iterator i(selections_->begin()),ie(selections_->end());
    // Loop over EDProducts, fill the provenance, and write.

    cout<<"Loop over EDProducts, fill the provenance, and write"<<endl;

    for(; i != ie; ++i) {
      BranchDescription const& desc = **i;
      ProductID const& id = desc.productID();

      if (id == ProductID()) {
        throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
          << "EventStreamOutput::serialize: invalid ProductID supplied in productRegistry\n";
      }
      EventPrincipal::SharedGroupPtr const group = e.getGroup(id);
      if (group.get() == 0) {
        std::string const& name = desc.className();
        std::string const className = wrappedClassName(name);
        TClass *cp = gROOT->GetClass(className.c_str());
	if (cp == 0) {
          throw edm::Exception(errors::ProductNotFound,"NoMatch")
            << "TypeID::className: No dictionary for class " << className << '\n'
            << "Add an entry for this class\n"
            << "to the appropriate 'classes_def.xml' and 'classes.h' files." << '\n';
	}
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
	   << " " << pii->desc()->className()
	   << " " << pii->desc()->productID_
	   << endl;      
    FDEBUG(11) << "-----Dump end" << endl;
#endif

    //EventBuffer::ProducerBuffer b(*bufs_);

    /*************************************/
    //Following is strictly DUMMY Data, and will be replaced with actual
    // once figured out.
    uint32 lumi=2;  
    std::vector<bool> l1bit(11);
    l1bit[0]=true;  l1bit[4]=true;  l1bit[8]=false;  
    l1bit[1]=true;  l1bit[5]=false;  l1bit[9]=false;
    l1bit[2]=false;  l1bit[6]=true;  l1bit[10]=true; 
    l1bit[3]=false;  l1bit[7]=false;  l1bit[11]=true;
    uint8 hltbits[] = "4567";
    const int hltsize = 9;//(sizeof(hltbits)-1)*4;
    uint32 reserved=78; 

    EventMsgBuilder msg(&bufs_[0], bufs_.size(),
                          e.id().run(), e.id().event(), lumi,
                          l1bit, hltbits, hltsize);
    msg.setReserved(reserved);  


    TBuffer rootbuf(TBuffer::kWrite,bufs_.size(),msg.eventAddr(),kFALSE);
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
    //b.commit(msg.size());

    cout<<"msg.size: "<<msg.size()<<endl;

    
    //Write the Event Message to Streamer file
    long long int event_offset = stream_writer_->write(msg);

    index_writer_->write(msg, event_offset);
  }

}
