#ifndef IORAWDATA_HCALTBINPUTSERVICE_HCALTBWRITER_H
#define IORAWDATA_HCALTBINPUTSERVICE_HCALTBWRITER_H 1

#include <map>
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "IORawData/HcalTBInputService/src/CDFRunInfo.h"

class TFile;
class TTree;
class CDFChunk;
class CDFEventInfo;

/** \class HcalTBWriter
  *  
  * Writes HCAL-style ROOT files from the RawData block
  *
  * \author J. Mans - Minnesota
  */
class HcalTBWriter : public edm::one::EDAnalyzer<> {
public:
  explicit HcalTBWriter(const edm::ParameterSet& pset);
  ~HcalTBWriter() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& es) override;
  void endJob() override;

private:
  std::string namePattern_;
  // chunk naming...
  std::map<int, std::string> blockToName_;
  void extractEventInfo(const FEDRawDataCollection& raw, const edm::EventID& id);
  void buildTree(const FEDRawDataCollection& raw);
  TFile* file_;
  TTree* tree_;
  CDFEventInfo* eventInfo_;
  CDFRunInfo ri_;
  std::map<int, int> chunkMap_;
  CDFChunk* chunkList_[1024];
  int trigChunk_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
};
#endif
