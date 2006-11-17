#ifndef StreamerOutputModule_h_
#define StreamerOutputModule_h_

#include "IOPool/Streamer/interface/Utilities.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/ProductRegistry.h"

#include "FWCore/Framework/interface/OutputModule.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/StreamTranslator.h"

#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"

#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/ParameterSetID.h"

#include "FWCore/Utilities/interface/Digest.h"

#include "TBuffer.h"

#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>

namespace 
{
   //A utility function that packs bits from source into bytes, with 
   // packInOneByte as the numeber of bytes that are packed from source to dest.
   void printBits(unsigned char c){
 
        for (int i=7; i>=0; i--) {
            int bit = ((c >> i) & 1);
            cout << " "<<bit; 
        } 
   }   
    
   void packIntoString(std::vector<unsigned char> const& source,
                    std::vector<unsigned char>& package)
   {   
   unsigned int packInOneByte = 4; 
   unsigned int sizeOfPackage = 1+((source.size()-1)/packInOneByte); //Two bits per HLT
    
   package.resize(sizeOfPackage); 
   memset(&package[0], 0x00, sizeOfPackage);
 
   for (std::vector<unsigned char>::size_type i=0; i != source.size() ; ++i)
   { 
      unsigned int whichByte = i/packInOneByte;
      unsigned int indxWithinByte = i % packInOneByte;
      package[whichByte] = package[whichByte] | (source[i] << (indxWithinByte*2));
   }
  //for (unsigned int i=0; i !=package.size() ; ++i)
  //   printBits(package[i]);
  // cout<<endl;

   }

}

namespace edm
{
  typedef edm::OutputModule::Selections Selections;
  typedef std::vector<char> SBuffer;
  
  template <class Consumer>
  class StreamerOutputModule : public edm::OutputModule
  {

  /** Consumers are suppose to provide
         void doOutputHeader(const InitMsgBuilder& init_message)
         void doOutputEvent(const EventMsgBuilder& msg)
         void stop()
  **/
        
  public:
    explicit StreamerOutputModule(edm::ParameterSet const& ps);  
    virtual ~StreamerOutputModule();

  private:
    virtual void write(EventPrincipal const& e);
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void endLuminosityBlock(LuminosityBlockPrincipal const&){}
    virtual void endRun(RunPrincipal const&){}

    std::auto_ptr<InitMsgBuilder> serializeRegistry();

    std::auto_ptr<EventMsgBuilder> serializeEvent(EventPrincipal const& e); 

    void setHltMask(EventPrincipal const& e);
     
  private:
    Selections const* selections_;

    SBuffer prod_reg_buf_;
    SBuffer bufs_;

    int maxEventSize_;
    bool useCompression_;
    int compressionLevel_;

    Consumer* c_;
    StreamTranslator* translator_;

    //Event variables, made class memebers to avoid re instatiation for each event.
    unsigned int hltsize_;
    uint32 lumi_;
    std::vector<bool> l1bit_;
    std::vector<unsigned char> hltbits_;
    uint32 reserved_;

  }; //end-of-class-def

 

template <class Consumer>
StreamerOutputModule<Consumer>::StreamerOutputModule(edm::ParameterSet const& ps):
  OutputModule(ps),
  selections_(&descVec_[InEvent]),
  prod_reg_buf_(1000 * 1000),
  maxEventSize_(ps.template getParameter<int>("max_event_size")),
  useCompression_(ps.template getParameter<bool>("use_compression")),
  compressionLevel_(ps.template getParameter<int>("compression_level")),
  c_(new Consumer(ps)),   //Try auto_ptr with this ?
  translator_(new StreamTranslator(selections_)),
  hltsize_(0),
  lumi_(2), //a dummy value of 2.
  l1bit_(0),
  hltbits_(0),
  reserved_(0) // no compression as default value - we need this!
  {
    if(useCompression_ == true)
    {
      if(compressionLevel_ <= 0) {
        FDEBUG(9) << "Compression Level = " << compressionLevel_ 
                  << " no compression" << std::endl;
        compressionLevel_ = 0;
        useCompression_ = false;
      } else if(compressionLevel_ > 9) {
        FDEBUG(9) << "Compression Level = " << compressionLevel_ 
                  << " using max compression level 9" << std::endl;
        compressionLevel_ = 9;
      }
    }
    bufs_.resize(maxEventSize_);
    edm::loadExtraClasses();
  }

template <class Consumer>
StreamerOutputModule<Consumer>::~StreamerOutputModule()
  {
    delete c_;
    delete translator_;
  }

template <class Consumer>
void StreamerOutputModule<Consumer>::beginJob(EventSetup const&)
  {
    std::auto_ptr<InitMsgBuilder>  init_message = serializeRegistry(); 
    c_->doOutputHeader(*init_message);  // You can't use init_message 
                                           // in StreamerOutputModule after this point
  }

template <class Consumer>
void StreamerOutputModule<Consumer>::endJob()
  {
    c_->stop();  // for closing of files, notify storage manager, etc.
  }

template <class Consumer>
void StreamerOutputModule<Consumer>::write(EventPrincipal const& e)
  {
    std::auto_ptr<EventMsgBuilder> msg = serializeEvent(e);
    c_->doOutputEvent(*msg); // You can't use msg
                              // in StreamerOutputModule after this point
  }


template <class Consumer>
std::auto_ptr<InitMsgBuilder> StreamerOutputModule<Consumer>::serializeRegistry()
  {
    //Build the INIT Message
    //Following values are strictly DUMMY and will be replaced
    // once available with Utility function etc.
    uint32 run = 1;
    
    //Get the Process PSet ID  
    edm::pset::Registry* reg = edm::pset::Registry::instance();
    edm::ParameterSetID toplevel = edm::pset::getProcessParameterSetID(reg);

    //In case we need to print it 
    //  cms::Digest dig(toplevel.compactForm());
    //  cms::MD5Result r1 = dig.digest();
    //  std::string hexy = r1.toString();
    //  cout <<"HEX Representation of Process PSetID: "<<hexy<<endl;  

    //Setting protocol version III
    Version v(3,(uint8*)toplevel.compactForm().c_str());

    Strings hlt_names = edm::getAllTriggerNames();
    hltsize_ = hlt_names.size();

    //L1 stays dummy as of today
    Strings l1_names;  //3
    l1_names.push_back("t1");  l1_names.push_back("t10");
    l1_names.push_back("t2");  

    std::auto_ptr<InitMsgBuilder> init_message(
                                new InitMsgBuilder(&prod_reg_buf_[0], prod_reg_buf_.size(),
                                      run, v, edm::getReleaseVersion().c_str() , hlt_names,
                                      l1_names));

    // the translator already has the product registry (selections_),
    // so it just needs to serialize it to the init message.
    translator_->serializeRegistry(*init_message);

    return init_message;
}


template <class Consumer>
void StreamerOutputModule<Consumer>::setHltMask(EventPrincipal const& e)
   {

    hltbits_.clear();  // If there was something left over from last event

    const edm::Handle<edm::TriggerResults>& prod = getTriggerResults(e);
    //const Trig& prod = getTrigMask(e);
    std::vector<unsigned char> vHltState; 
    
    if (prod.isValid())
    {
      for(std::vector<unsigned char>::size_type i=0; i != hltsize_ ; ++i) {
        vHltState.push_back(((prod->at(i)).state()));
      }
    }
    else 
    {
     // We fill all Trigger bits to valid state.
     for(std::vector<unsigned char>::size_type i=0; i != hltsize_ ; ++i)
        {
           vHltState.push_back(hlt::Pass);
        }
    }
    
    //Pack into member hltbits_
    packIntoString(vHltState, hltbits_);

    //This is Just a printing code.
    //cout <<"Size of hltbits:"<<hltbits_.size()<<endl;
    //for(unsigned int i=0; i != hltbits_.size() ; ++i) {
    //  printBits(hltbits_[i]);
    //}
    //cout<<"\n";

   }

template <class Consumer>
std::auto_ptr<EventMsgBuilder> StreamerOutputModule<Consumer>::serializeEvent(
                                                 EventPrincipal const& e)
  {
    //Lets Build the Event Message first 

    //Following is strictly DUMMY Data for L! Trig and will be replaced with actual
    // once figured out, there is no logic involved here.
    l1bit_.push_back(true);
    l1bit_.push_back(true);
    l1bit_.push_back(false);
    //End of dummy data

    setHltMask(e);

    std::auto_ptr<EventMsgBuilder> msg( 
                           new EventMsgBuilder(&bufs_[0], bufs_.size(),
                           e.id().run(), e.id().event(), lumi_,
                           l1bit_, (uint8*)&hltbits_[0], hltsize_) );
    msg->setReserved(reserved_); // we need this set to zero

    translator_->serializeEvent(e, *msg, useCompression_, compressionLevel_);

    l1bit_.clear();  //Clear up for the next event to come.
    return msg;
}

} // end of namespace-edm

#endif

