#ifndef IOPool_Streamer_FRDEventOutputModule_h
#define IOPool_Streamer_FRDEventOutputModule_h

#include "FWCore/Framework/interface/OutputModule.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "boost/shared_array.hpp"

namespace edm {
  class ModuleCallingContext;
}

class FRDEventMsgView;
template <class Consumer>
class FRDEventOutputModule : public edm::OutputModule
{
  typedef unsigned int uint32;
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
  virtual void write(edm::EventPrincipal const& e, edm::ModuleCallingContext const*) override;
  virtual void beginRun(edm::RunPrincipal const&, edm::ModuleCallingContext const*) override;
  virtual void endRun(edm::RunPrincipal const&, edm::ModuleCallingContext const*) override;
  virtual void writeRun(edm::RunPrincipal const&, edm::ModuleCallingContext const*) override {}
  virtual void writeLuminosityBlock(edm::LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*) override {}

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
void FRDEventOutputModule<Consumer>::write(edm::EventPrincipal const& e, edm::ModuleCallingContext const* mcc) {

  static const edm::TypeID fedType(typeid(FEDRawDataCollection));
  static const std::string emptyString("");

  edm::Handle<FEDRawDataCollection> fedBuffers;
  edm::BasicHandle h = e.getByLabel(edm::PRODUCT_TYPE,
                                    fedType,
                                    label_,
                                    instance_,
                                    emptyString,
                                    nullptr,
                                    mcc);
  convert_handle(h, fedBuffers);

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
  *bufPtr++ = (uint32) e.run();
  *bufPtr++ = (uint32) e.luminosityBlock();
  *bufPtr++ = (uint32) e.id().event();
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
void FRDEventOutputModule<Consumer>::beginRun(edm::RunPrincipal const&, edm::ModuleCallingContext const*)
{
  templateConsumer_->start();
}
   
template <class Consumer>
void FRDEventOutputModule<Consumer>::endRun(edm::RunPrincipal const&, edm::ModuleCallingContext const*)
{
  templateConsumer_->stop();
}

#endif
