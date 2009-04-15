#ifndef IOPool_Streamer_FRDEventOutputModule_h
#define IOPool_Streamer_FRDEventOutputModule_h

#include "FWCore/Framework/interface/OutputModule.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "boost/shared_array.hpp"

template <class Consumer>
class FRDEventOutputModule : public edm::OutputModule
{

  /**
   * Consumers are suppose to provide:
   *   void doOutputEvent(const FRDEventMsgView& msg)
   *   void start()
   *   void stop()
   */

 public:
  explicit FRDEventOutputModule(edm::ParameterSet const& ps);  
  ~FRDEventOutputModule();

 private:
  virtual void write(edm::EventPrincipal const& e);
  virtual void beginRun(edm::RunPrincipal const&);
  virtual void endRun(edm::RunPrincipal const&);
  virtual void writeRun(edm::RunPrincipal const&) {}
  virtual void writeLuminosityBlock(edm::LuminosityBlockPrincipal const&) {}

  std::auto_ptr<Consumer> templateConsumer_;
  std::string label_;
  std::string instance_;
};

template <class Consumer>
FRDEventOutputModule<Consumer>::FRDEventOutputModule(edm::ParameterSet const& ps) :
  edm::OutputModule(ps),
  templateConsumer_(new Consumer(ps)),
  label_(ps.getUntrackedParameter<std::string>("ProductLabel","source")),
  instance_(ps.getUntrackedParameter<std::string>("ProductInstance",""))
{
}

template <class Consumer>
FRDEventOutputModule<Consumer>::~FRDEventOutputModule() {}

template <class Consumer>
void FRDEventOutputModule<Consumer>::write(edm::EventPrincipal const& e) {

  // serialize the FEDRawDataCollection into the format that we expect for
  // FRDEventMsgView objects (may be better ways to do this)
  edm::Event event(const_cast<edm::EventPrincipal&>(e), description());
  edm::Handle<FEDRawDataCollection> fedBuffers;
  event.getByLabel(label_, instance_, fedBuffers);

  // determine the expected size of the FRDEvent
  int expectedSize = (4 + 1024) * sizeof(uint32);
  for (int idx = 0; idx < 1024; ++idx) {
    FEDRawData singleFED = fedBuffers->FEDData(idx);
    expectedSize += singleFED.size();
    //if (singleFED.size() > 0) {
    //  std::cout << "FED #" << idx << " size = " << singleFED.size() << std::endl;
    //}
  }

  // build the FRDEvent into a temporary buffer
  boost::shared_array<unsigned char> workBuffer(new unsigned char[expectedSize + 256]);
  uint32 *bufPtr = (uint32*) workBuffer.get();
  *bufPtr++ = (uint32) 2;  // version number
  *bufPtr++ = (uint32) event.id().run();
  *bufPtr++ = (uint32) event.luminosityBlock();
  *bufPtr++ = (uint32) event.id().event();
  for (int idx = 0; idx < 1024; ++idx) {
    FEDRawData singleFED = fedBuffers->FEDData(idx);
    *bufPtr++ = singleFED.size();
  }
  for (int idx = 0; idx < 1024; ++idx) {
    FEDRawData singleFED = fedBuffers->FEDData(idx);
    if (singleFED.size() > 0) {
      memcpy(bufPtr, singleFED.data(), singleFED.size());
      *bufPtr += singleFED.size();
    }
  }

  // create the FRDEventMsgView and use the template consumer to write it out
  FRDEventMsgView msg(workBuffer.get());
  templateConsumer_->doOutputEvent(msg);
}

template <class Consumer>
void FRDEventOutputModule<Consumer>::beginRun(edm::RunPrincipal const&)
{
  templateConsumer_->start();
}
   
template <class Consumer>
void FRDEventOutputModule<Consumer>::endRun(edm::RunPrincipal const&)
{
  templateConsumer_->stop();
}

#endif
