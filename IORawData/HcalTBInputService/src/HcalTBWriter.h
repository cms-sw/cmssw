#ifndef IORAWDATA_HCALTBINPUTSERVICE_HCALTBWRITER_H
#define IORAWDATA_HCALTBINPUTSERVICE_HCALTBWRITER_H 1

#include <map>
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
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
  * $Date: 2012/09/27 15:40:32 $
  * $Revision: 1.3 $
  * \author J. Mans - Minnesota
  */
class HcalTBWriter : public edm::EDAnalyzer {
public:
  HcalTBWriter(const edm::ParameterSet & pset);
  virtual void analyze(const edm::Event& e, const edm::EventSetup& es);
  virtual void endJob();
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
  std::map<int,int> chunkMap_;
  CDFChunk* chunkList_[1024];
  int trigChunk_;
  edm::InputTag fedRawDataCollectionTag_;
};
#endif
