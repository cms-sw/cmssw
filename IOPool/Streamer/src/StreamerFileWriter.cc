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

//#include "IOPool/Streamer/src/StreamerUtilities.h"

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
    //tc_(),
    prod_reg_buf_(100 * 1000),
    prod_reg_len_()
     
  {
    bufs_.resize(ps.template getParameter<int>("max_event_size"));
    FDEBUG(6) << "StreamOutput constructor" << endl;
    edm::loadExtraClasses();
    //tc_ = getTClass(typeid(SendEvent));
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

    int msg_length = serializeRegistry(descVec_, (InitMsgBuilder&) init_message);
    init_message.setDescLength(msg_length);
 
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

    int msg_length = serializeEvent(e, selections_, msg, bufs_.size() );
    msg.setEventLength(msg_length);
    //b.commit(msg.size());

    cout<<"msg.size: "<<msg.size()<<endl;

    
    //Write the Event Message to Streamer file
    long long int event_offset = stream_writer_->write(msg);

    index_writer_->write(msg, event_offset);
  }

}
