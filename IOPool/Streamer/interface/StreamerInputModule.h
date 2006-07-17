#ifndef StreamerInputModule_h
#define StreamerInputModule_h


#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "IOPool/Streamer/interface/StreamerFileIO.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/InputSource.h"

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm> 
#include <iterator>

namespace edm
{
  template <class Producer>
  class StreamerInputModule : public edm::InputSource
  {
  /**
     Requires the Producer class to provide following functions
           std::auto_ptr<InitMsgView> getHeader();
           std::auto_ptr<EventMsgView> getNextEvent();
  */
  public:  
    explicit StreamerInputModule(edm::ParameterSet const& pset,
                 edm::InputSourceDescription const& desc);
    virtual ~StreamerInputModule();
    virtual std::auto_ptr<edm::EventPrincipal> read();

  private:
    std::auto_ptr<SendJobHeader> readHead(std::auto_ptr<InitMsgView>);
    std::auto_ptr<EventPrincipal> readEvt(std::auto_ptr<EventMsgView>, const ProductRegistry& );

    void mergeWithRegistry(const edm::SendDescs& descs,ProductRegistry&);
    void declareStreamers(const edm::SendDescs& descs);
    void buildClassCache(const edm::SendDescs& descs);

    //ProductRegistry const* prod_reg_;
    TClass* tc_;
    Producer* pr_; 
    
  }; //end-of-class-def

template <class Producer>
StreamerInputModule<Producer>::~StreamerInputModule()
   {
      delete pr_;
      cout<<"~StreamerInputModule is DONE"<<endl;
       
   }

template <class Producer>
StreamerInputModule<Producer>::StreamerInputModule(
                    edm::ParameterSet const& pset,
                    edm::InputSourceDescription const& desc):
    edm::InputSource(pset, desc),
    //prod_reg_(&productRegistry()), 
    pr_(new Producer(pset))
    {
      //Get header/init from Producer
      std::auto_ptr<InitMsgView> header = pr_->getHeader();
      std::auto_ptr<edm::SendJobHeader> p = readHead(header); 
      //header cannot be used anymore beyond this point
      SendDescs & descs = p->descs_;
      mergeWithRegistry(descs, productRegistry());

      // jbk - the next line should not be needed
      declareStreamers(descs);
      buildClassCache(descs);
      edm::loadExtraClasses();
      tc_ = getTClass(typeid(SendEvent));
    }

template <class Producer>
auto_ptr<SendJobHeader> StreamerInputModule<Producer>::readHead(std::auto_ptr<InitMsgView> header)
   {
      if(header->code() != 0) //INIT Msg
      throw cms::Exception("readHeader","StreamerFileReader")
        << "received wrong message type: expected INIT, got "
        << header->code() << "\n";

      TClass* desc = getTClass(typeid(SendJobHeader));

      TBuffer xbuf(TBuffer::kRead, header->descLength(),
                                   (char*)header->descData(),kFALSE);
      RootDebug tracer(10,10);
      auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc));

      if(sd.get()==0) 
      {
          throw cms::Exception("HeaderDecode","DecodeProductList")
            << "Could not read the initial product registry list\n";
      }

      return sd;  
   }

template <class Producer>
std::auto_ptr<edm::EventPrincipal> StreamerInputModule<Producer>::read()
  {
     std::auto_ptr<EventMsgView> eview = pr_->getNextEvent();
     if (eview.get() == 0)  
                          {
                          cout<<"Empty event........"<<endl; 
                          return  std::auto_ptr<edm::EventPrincipal>();
                          }
     return this->readEvt(eview, productRegistry());
  }

template <class Producer>
std::auto_ptr<EventPrincipal> 
StreamerInputModule<Producer>::readEvt(std::auto_ptr<EventMsgView> eview, const ProductRegistry& prod_reg)
  {
      cout << "Decide event: "
              << eview->event() << " "
              << eview->run() << " "
              << eview->size() << " "
              << eview->eventLength() << " "
              << eview->eventData()
              << endl;
    TBuffer xbuf(TBuffer::kRead, eview->eventLength(),(char*) eview->eventData(),kFALSE);
    RootDebug tracer(10,10);
    auto_ptr<SendEvent> sd((SendEvent*)xbuf.ReadObjectAny(tc_));
    if(sd.get()==0)
      {
        throw cms::Exception("EventInput","Read")
          << "got a null event from input stream\n";
      }

    FDEBUG(5) << "Got event: " << sd->id_ << " " << sd->prods_.size() << endl;
    auto_ptr<EventPrincipal> ep(new EventPrincipal(sd->id_,
                                                   sd->time_,
                                                   prod_reg));
    // no process name list handling

    SendProds::iterator spi(sd->prods_.begin()),spe(sd->prods_.end());
    for(;spi!=spe;++spi)
      {
        FDEBUG(10) << "check prodpair" << endl;
        if(spi->prov()==0)
          throw cms::Exception("NoData","EmptyProvenance");
        if(spi->desc()==0)
          throw cms::Exception("NoData","EmptyDesc");
        FDEBUG(5) << "Prov:"
             << " " << spi->desc()->className()
             << " " << spi->desc()->productInstanceName()
             << " " << spi->desc()->productID()
             << " " << spi->prov()->productID_
             << endl;

        if(spi->prod()==0)
          {
            FDEBUG(10) << "Product is null" << endl;
            continue;
            throw cms::Exception("NoData","EmptyProduct");
          }

        auto_ptr<EDProduct>
          aprod(const_cast<EDProduct*>(spi->prod()));
        auto_ptr<BranchEntryDescription>
          aedesc(const_cast<BranchEntryDescription*>(spi->prov()));
        auto_ptr<BranchDescription>
          adesc(const_cast<BranchDescription*>(spi->desc()));

        auto_ptr<Provenance> aprov(new Provenance);
        aprov->event   = *(aedesc.get());
        aprov->product = *(adesc.get());
        if(aprov->isPresent()) {
          FDEBUG(10) << "addgroup next " << aprov->productID() << endl;
          FDEBUG(10) << "addgroup next " << aprov->event.productID_ << endl;
          ep->addGroup(auto_ptr<Group>(new Group(aprod,aprov)));
          FDEBUG(10) << "addgroup done" << endl;
        } else {
          FDEBUG(10) << "addgroup empty next " << aprov->productID() << endl;
          FDEBUG(10) << "addgroup empty next " << aprov->event.productID_ 
                                               << endl;
          ep->addGroup(auto_ptr<Group>(new Group(aprov, false)));
          FDEBUG(10) << "addgroup empty done" << endl;
        }
        spi->clear();
      }

    FDEBUG(10) << "Size = " << ep->numEDProducts() << endl;

    return ep;     
  }

template <class Producer>
void StreamerInputModule<Producer>::mergeWithRegistry(const SendDescs& descs,ProductRegistry& reg)
  {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    // the next line seems to be not good.  what if the productdesc is
    // already there? it looks like I replace it.  maybe that it correct

    FDEBUG(6) << "mergeWithRegistry: Product List: " << endl;
    for(; i != e; ++i) {
        reg.copyProduct(*i);
        FDEBUG(6) << "StreamInput prod = " << i->className() << endl;
    }
  }

template <class Producer>
void StreamerInputModule<Producer>::declareStreamers(const SendDescs& descs)
  {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        string real_name = edm::wrappedClassName(i->className());
        FDEBUG(6) << "declare: " << real_name << endl;
        edm::loadCap(real_name);
    }
  }


template <class Producer>
void StreamerInputModule<Producer>::buildClassCache(const SendDescs& descs)
  { 
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        string real_name = edm::wrappedClassName(i->className());
        FDEBUG(6) << "BuildReadData: " << real_name << endl;
        edm::doBuildRealData(real_name);
    }
  }

} // end of namespace-edm
  
#endif
