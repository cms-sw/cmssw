#ifndef EventFilter_Utilities_FRDStreamSource_h
#define EventFilter_Utilities_FRDStreamSource_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include <unistd.h>
#include <string>
#include <vector>
#include <fstream>


class FRDStreamSource : public edm::ProducerSourceFromFiles {

public:
  // construction/destruction
  FRDStreamSource(edm::ParameterSet const& pset,
                  edm::InputSourceDescription const& desc);
  virtual ~FRDStreamSource() {};

private:
  // member functions
  virtual bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& theTime, edm::EventAuxiliary::ExperimentType& eType);
  virtual void produce(edm::Event& e);

  void beginRun(edm::Run&) {}
  void endRun(edm::Run&) {}
  void beginLuminosityBlock(edm::LuminosityBlock&) {}
  void endLuminosityBlock(edm::LuminosityBlock&) {}

  bool openFile(const std::string& fileName);


private:
  // member data
  std::vector<std::string>::const_iterator itFileName_;
  std::ifstream fin_;
  std::auto_ptr<FEDRawDataCollection> rawData_;
  std::vector<char> buffer_;
  const bool verifyAdler32_;
  const bool verifyChecksum_;
  const bool useL1EventID_;
  unsigned int detectedFRDversion_=0;
};

#endif // EventFilter_Utilities_FRDStreamSource_h
