#include <TFile.h>
#include <TTree.h>
#include "IORawData/HcalTBInputService/src/CDFChunk.h"
#include "IORawData/HcalTBInputService/src/CDFEventInfo.h"
#include "IORawData/HcalTBInputService/plugins/HcalTBWriter.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include <unistd.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalTBWriter::HcalTBWriter(const edm::ParameterSet& pset)
    : namePattern_(pset.getUntrackedParameter<std::string>("FilenamePattern", "/tmp/HTB_%06d.root")) {
  tok_raw_ = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("fedRawDataCollectionTag"));

  std::vector<edm::ParameterSet> names = pset.getUntrackedParameter<std::vector<edm::ParameterSet> >("ChunkNames");
  std::vector<edm::ParameterSet>::iterator j;
  for (j = names.begin(); j != names.end(); j++) {
    std::string name = j->getUntrackedParameter<std::string>("Name");
    int num = j->getUntrackedParameter<int>("Number");
    blockToName_[num] = name;
  }

  file_ = nullptr;
  tree_ = nullptr;
  eventInfo_ = nullptr;
}

void HcalTBWriter::endJob() {
  char buffer[1024];
  if (file_ != nullptr) {
    file_->Write();

    ri_.setInfo("DAQSofwareRelease", "UNKNOWN -- HcalTBWriter");
    gethostname(buffer, 1024);
    ri_.setInfo("WriterHostname", buffer);
    ri_.store(file_);

    file_->Close();
    file_ = nullptr;
    tree_ = nullptr;
    chunkMap_.clear();
    eventInfo_ = nullptr;
  }
}

void HcalTBWriter::analyze(const edm::Event& e, const edm::EventSetup& es) {
  edm::Handle<FEDRawDataCollection> raw;
  e.getByToken(tok_raw_, raw);

  if (file_ == nullptr) {
    char fname[4096];
    snprintf(fname, 4096, namePattern_.c_str(), e.id().run());
    edm::LogInfo("HCAL") << "Opening " << fname << " for writing HCAL-format file.";
    file_ = new TFile(fname, "RECREATE");
    ri_.setInfo("OriginalFile", fname);
    buildTree(*raw);
  }

  // adopt the buffers for writing
  for (std::map<int, int>::const_iterator i = chunkMap_.begin(); i != chunkMap_.end(); i++) {
    CDFChunk* c = chunkList_[i->second];
    const FEDRawData& frd = raw->FEDData(i->first);
    c->adoptBuffer((ULong64_t*)frd.data(), frd.size() / 8);
  }

  // copy the event info bits
  extractEventInfo(*raw, e.id());

  // fill the tree
  tree_->Fill();
  // release all the buffers
  for (std::map<int, int>::const_iterator i = chunkMap_.begin(); i != chunkMap_.end(); i++) {
    CDFChunk* c = chunkList_[i->second];
    c->releaseBuffer();
  }
}

void HcalTBWriter::buildTree(const FEDRawDataCollection& raw) {
  tree_ = new TTree("CMSRAW", "CMS Common Data Format Tree");
  chunkMap_.clear();
  trigChunk_ = -1;
  int j = 0;
  for (int i = 0; i < 2048; i++) {
    const FEDRawData& frd = raw.FEDData(i);
    if (frd.size() < 16)
      continue;  // it's empty... like

    std::string name;
    if (blockToName_.find(i) != blockToName_.end())
      name = blockToName_[i];
    else {
      char sname[64];
      snprintf(sname, 64, "Chunk%03d", i);
      name = sname;
    }

    CDFChunk* c = new CDFChunk(name.c_str());
    chunkList_[j] = c;
    tree_->Branch(name.c_str(), "CDFChunk", &(chunkList_[j]));
    chunkMap_[i] = j;

    if (name == "HCAL_Trigger" || name == "SliceTest_Trigger")
      trigChunk_ = j;

    j++;
  }
  eventInfo_ = new CDFEventInfo();
  tree_->Branch("CDFEventInfo", "CDFEventInfo", &eventInfo_, 16000, 2);
}

typedef struct StandardTrgMsgBlkStruct {
  uint32_t orbitNumber;
  uint32_t eventNumber;
  uint32_t flags_daq_ttype;
  uint32_t algo_bits_3;
  uint32_t algo_bits_2;
  uint32_t algo_bits_1;
  uint32_t algo_bits_0;
  uint32_t tech_bits;
  uint32_t gps_1234;
  uint32_t gps_5678;
} StandardTrgMsgBlk;

typedef struct newExtendedTrgMsgBlkStruct {
  StandardTrgMsgBlk stdBlock;
  uint32_t triggerWord;
  uint32_t triggerTime_usec;
  uint32_t triggerTime_base;
  uint32_t spillNumber;
  uint32_t runNumber;
  char runNumberSequenceId[16];
  uint32_t eventStatus;
} newExtendedTrgMsgBlk;

void HcalTBWriter::extractEventInfo(const FEDRawDataCollection& raw, const edm::EventID& id) {
  int runno = id.run();
  const char* seqid = "";
  int eventNo = id.event();
  int l1aNo = eventNo;
  int orbitNo = 0;
  int bunchNo = 0;

  if (trigChunk_ >= 0) {
    const newExtendedTrgMsgBlk* tinfo = (const newExtendedTrgMsgBlk*)(chunkList_[trigChunk_]->getData() +
                                                                      2);  // assume 2 64-bit words for the CDF header
    orbitNo = tinfo->stdBlock.orbitNumber;
    seqid = tinfo->runNumberSequenceId;
    FEDHeader head((const unsigned char*)chunkList_[trigChunk_]->getData());
    bunchNo = head.bxID();
    l1aNo = head.lvl1ID();
  }

  eventInfo_->Set(runno, seqid, eventNo, l1aNo, orbitNo, bunchNo);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include <cstdint>

DEFINE_FWK_MODULE(HcalTBWriter);
