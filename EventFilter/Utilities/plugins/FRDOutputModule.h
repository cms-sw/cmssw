#ifndef IOPool_Streamer_interface_FRDOutputModule_h
#define IOPool_Streamer_interface_FRDOutputModule_h

// CMSSW headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/one/OutputModule.h"
//#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

class FRDOutputModule : public edm::one::OutputModule<edm::one::WatchLuminosityBlocks> {
public:
  explicit FRDOutputModule(edm::ParameterSet const& ps);
  ~FRDOutputModule() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void write(edm::EventForOutput const& e) override;
  //void beginRun(edm::RunForOutput const&) override {}
  //void endRun(edm::RunForOutput const&) override {}
  void writeRun(const edm::RunForOutput&) override {}
  void writeLuminosityBlock(const edm::LuminosityBlockForOutput&) override {}

  void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
  void endLuminosityBlock(edm::LuminosityBlockForOutput const&) override;

  void finishFileWrite(unsigned int run, int ls);
  uint32_t adler32() const { return (adlerb_ << 16) | adlera_; }

  const edm::EDGetTokenT<FEDRawDataCollection> token_;

  const uint32_t frdVersion_;
  const uint32_t frdFileVersion_;
  std::string filePrefix_;
  std::string fileName_;

  int outfd_ = -1;
  uint32_t adlera_;
  uint32_t adlerb_;

  uint32_t perFileEventCount_;
  uint64_t perFileSize_;

  bool fileWritten_ = false;
};

#endif  // IOPool_Streamer_interface_FRDOutputModule_h
