/** \class CSCDCCUnpacker
 *
 *
 * \author Alex Tumanov
 */

//Framework stuff
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"

//FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

//Digi stuff

#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h"

#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCGEMData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h"

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <cstdio>

class CSCMonitorInterface;

class CSCDCCUnpacker : public edm::stream::EDProducer<> {
public:
  /// Constructor
  CSCDCCUnpacker(const edm::ParameterSet& pset);

  /// Destructor
  ~CSCDCCUnpacker() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// Produce digis out of raw data
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  /// Visualization of raw data in FED-less events (Robert Harr and Alexander Sakharov)
  void visual_raw(int hl, int id, int run, int event, bool fedshort, bool fDump, short unsigned int* buf) const;

private:
  bool debug, printEventNumber, goodEvent, useExaminer, unpackStatusDigis;
  bool useSelectiveUnpacking, useFormatStatus;

  /// option to unpack RPC data
  bool useRPCs_;

  /// option to unpack GEM cluster data
  bool useGEMs_;

  /// option to unpack CSC shower data
  bool useCSCShowers_;

  /// Visualization of raw data
  bool visualFEDInspect, visualFEDShort, formatedEventDump;
  /// Suppress zeros LCTs
  bool SuppressZeroLCT;

  int numOfEvents;
  unsigned int errorMask, examinerMask;
  bool instantiateDQM;

  bool disableMappingCheck, b904Setup;
  int b904vmecrate, b904dmb;

  CSCMonitorInterface* monitor;

  /// Token for consumes interface & access to data
  edm::EDGetTokenT<FEDRawDataCollection> i_token;
  edm::ESGetToken<CSCCrateMap, CSCCrateMapRcd> crateToken;
  edm::ESGetToken<CSCChamberMap, CSCChamberMapRcd> cscmapToken;
};

CSCDCCUnpacker::CSCDCCUnpacker(const edm::ParameterSet& pset) : numOfEvents(0) {
  // Tracked
  i_token = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("InputObjects"));
  crateToken = esConsumes<CSCCrateMap, CSCCrateMapRcd>();
  cscmapToken = esConsumes<CSCChamberMap, CSCChamberMapRcd>();

  useExaminer = pset.getParameter<bool>("UseExaminer");
  examinerMask = pset.getParameter<unsigned int>("ExaminerMask");
  /// Selective unpacking mode will skip only troublesome CSC blocks and not whole DCC/DDU block
  useSelectiveUnpacking = pset.getParameter<bool>("UseSelectiveUnpacking");
  errorMask = pset.getParameter<unsigned int>("ErrorMask");
  unpackStatusDigis = pset.getParameter<bool>("UnpackStatusDigis");
  /// Enable Format Status Digis
  useFormatStatus = pset.getParameter<bool>("UseFormatStatus");

  useRPCs_ = pset.getParameter<bool>("useRPCs");
  useGEMs_ = pset.getParameter<bool>("useGEMs");
  useCSCShowers_ = pset.getParameter<bool>("useCSCShowers");

  // Untracked
  printEventNumber = pset.getUntrackedParameter<bool>("PrintEventNumber", true);
  debug = pset.getUntrackedParameter<bool>("Debug", false);
  instantiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);

  // Disable FED/DDU to chamber mapping inconsistency check
  disableMappingCheck = pset.getUntrackedParameter<bool>("DisableMappingCheck", false);
  // Make aware the unpacker that B904 test setup is used (disable mapping inconsistency check)
  b904Setup = pset.getUntrackedParameter<bool>("B904Setup", false);
  b904vmecrate = pset.getUntrackedParameter<int>("B904vmecrate", 1);
  b904dmb = pset.getUntrackedParameter<int>("B904dmb", 3);

  /// Visualization of raw data
  visualFEDInspect = pset.getUntrackedParameter<bool>("VisualFEDInspect", false);
  visualFEDShort = pset.getUntrackedParameter<bool>("VisualFEDShort", false);
  formatedEventDump = pset.getUntrackedParameter<bool>("FormatedEventDump", false);

  /// Suppress zeros LCTs
  SuppressZeroLCT = pset.getUntrackedParameter<bool>("SuppressZeroLCT", true);

  if (instantiateDQM) {
    monitor = edm::Service<CSCMonitorInterface>().operator->();
  }

  produces<CSCWireDigiCollection>("MuonCSCWireDigi");
  produces<CSCStripDigiCollection>("MuonCSCStripDigi");
  produces<CSCComparatorDigiCollection>("MuonCSCComparatorDigi");
  produces<CSCALCTDigiCollection>("MuonCSCALCTDigi");
  produces<CSCCLCTDigiCollection>("MuonCSCCLCTDigi");
  produces<CSCCorrelatedLCTDigiCollection>("MuonCSCCorrelatedLCTDigi");

  if (unpackStatusDigis) {
    produces<CSCCFEBStatusDigiCollection>("MuonCSCCFEBStatusDigi");
    produces<CSCTMBStatusDigiCollection>("MuonCSCTMBStatusDigi");
    produces<CSCDMBStatusDigiCollection>("MuonCSCDMBStatusDigi");
    produces<CSCALCTStatusDigiCollection>("MuonCSCALCTStatusDigi");
    produces<CSCDDUStatusDigiCollection>("MuonCSCDDUStatusDigi");
    produces<CSCDCCStatusDigiCollection>("MuonCSCDCCStatusDigi");
  }

  if (useFormatStatus) {
    produces<CSCDCCFormatStatusDigiCollection>("MuonCSCDCCFormatStatusDigi");
  }

  if (useRPCs_) {
    produces<CSCRPCDigiCollection>("MuonCSCRPCDigi");
  }

  if (useGEMs_) {
    produces<GEMPadDigiClusterCollection>("MuonGEMPadDigiCluster");
  }

  if (useCSCShowers_) {
    produces<CSCShowerDigiCollection>("MuonCSCShowerDigi");
    produces<CSCShowerDigiCollection>("MuonCSCShowerDigiAnode");
    produces<CSCShowerDigiCollection>("MuonCSCShowerDigiCathode");
    produces<CSCShowerDigiCollection>("MuonCSCShowerDigiAnodeALCT");
  }

  //CSCAnodeData::setDebug(debug);
  CSCALCTHeader::setDebug(debug);
  CSCComparatorData::setDebug(debug);
  CSCEventData::setDebug(debug);
  CSCTMBData::setDebug(debug);
  CSCDCCEventData::setDebug(debug);
  CSCDDUEventData::setDebug(debug);
  CSCTMBHeader::setDebug(debug);
  CSCRPCData::setDebug(debug);
  CSCDDUEventData::setErrorMask(errorMask);
}

CSCDCCUnpacker::~CSCDCCUnpacker() {
  //fill destructor here
}

void CSCDCCUnpacker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects", edm::InputTag("rawDataCollector"))
      ->setComment("# Define input to the unpacker");
  desc.add<bool>("UseExaminer", true)
      ->setComment("# Use CSC examiner to check for corrupt or semi-corrupt data & avoid unpacker crashes");
  desc.add<unsigned int>("ExaminerMask", 535557110)->setComment("# This mask is needed by the examiner");
  desc.add<bool>("UseSelectiveUnpacking", true)
      ->setComment("# Use Examiner to unpack good chambers and skip only bad ones");
  desc.add<unsigned int>("ErrorMask", 0)->setComment("# This mask simply reduces error reporting");
  desc.add<bool>("UnpackStatusDigis", false)->setComment("# Unpack general status digis?");
  desc.add<bool>("UseFormatStatus", true)->setComment("# Unpack FormatStatus digi?");
  desc.add<bool>("useRPCs", false)->setComment("Unpack RPC data");
  desc.add<bool>("useGEMs", true)->setComment("Unpack GEM trigger data");
  desc.add<bool>("useCSCShowers", true)->setComment("Unpack CSCShower trigger data");
  desc.addUntracked<bool>("Debug", false)->setComment("# Turn on lots of output");
  desc.addUntracked<bool>("PrintEventNumber", false);
  desc.addUntracked<bool>("runDQM", false);
  desc.addUntracked<bool>("VisualFEDInspect", false)->setComment("# Visualization of raw data in corrupted events");
  desc.addUntracked<bool>("VisualFEDShort", false)->setComment("# Visualization of raw data in corrupted events");
  desc.addUntracked<bool>("FormatedEventDump", false);
  desc.addUntracked<bool>("SuppressZeroLCT", true);
  desc.addUntracked<bool>("DisableMappingCheck", false)
      ->setComment("# Disable FED/DDU to chamber mapping inconsistency check");
  desc.addUntracked<bool>("B904Setup", false)->setComment("# Make the unpacker aware of B904 test setup configuration");
  desc.addUntracked<int>("B904vmecrate", 1)->setComment("# Set vmecrate number for chamber used in B904 test setup");
  desc.addUntracked<int>("B904dmb", 3)->setComment("# Set dmb slot for chamber used in B904 test setup");
  descriptions.add("muonCSCDCCUnpacker", desc);
  descriptions.setComment(" This is the generic cfi file for CSC unpacking");
}

void CSCDCCUnpacker::produce(edm::Event& e, const edm::EventSetup& c) {
  ///access database for mapping
  // Do we really have to do this every event???
  // ... Yes, because framework is more efficient than you are at caching :)
  // (But if you want to actually DO something specific WHEN the mapping changes, check out ESWatcher)
  edm::ESHandle<CSCCrateMap> hcrate = c.getHandle(crateToken);
  const CSCCrateMap* pcrate = hcrate.product();

  // Need access to CSCChamberMap for chamber<->FED/DDU mapping consistency checks
  edm::ESHandle<CSCChamberMap> cscmap = c.getHandle(cscmapToken);
  const CSCChamberMap* cscmapping = cscmap.product();

  if (printEventNumber)
    ++numOfEvents;

  /// Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByToken(i_token, rawdata);

  /// create the collections of CSC digis
  auto wireProduct = std::make_unique<CSCWireDigiCollection>();
  auto stripProduct = std::make_unique<CSCStripDigiCollection>();
  auto alctProduct = std::make_unique<CSCALCTDigiCollection>();
  auto clctProduct = std::make_unique<CSCCLCTDigiCollection>();
  auto comparatorProduct = std::make_unique<CSCComparatorDigiCollection>();
  auto rpcProduct = std::make_unique<CSCRPCDigiCollection>();
  auto corrlctProduct = std::make_unique<CSCCorrelatedLCTDigiCollection>();
  auto cfebStatusProduct = std::make_unique<CSCCFEBStatusDigiCollection>();
  auto dmbStatusProduct = std::make_unique<CSCDMBStatusDigiCollection>();
  auto tmbStatusProduct = std::make_unique<CSCTMBStatusDigiCollection>();
  auto dduStatusProduct = std::make_unique<CSCDDUStatusDigiCollection>();
  auto dccStatusProduct = std::make_unique<CSCDCCStatusDigiCollection>();
  auto alctStatusProduct = std::make_unique<CSCALCTStatusDigiCollection>();

  auto formatStatusProduct = std::make_unique<CSCDCCFormatStatusDigiCollection>();

  auto gemProduct = std::make_unique<GEMPadDigiClusterCollection>();

  auto lctShowerProduct =
      std::make_unique<CSCShowerDigiCollection>();  // HMT shower objects from OTMB/MPC LCT trigger data frames
  auto anodeShowerProductOTMB =
      std::make_unique<CSCShowerDigiCollection>();  // anode HMT shower objects from (O)TMB header data (matched at OTMB)
  auto cathodeShowerProductOTMB =
      std::make_unique<CSCShowerDigiCollection>();  // cathode HMT shower objects from (O)TMB header data
  auto anodeShowerProductALCT = std::make_unique<
      CSCShowerDigiCollection>();  // anode HMT shower objects from ALCT data (vector of HMT shower objects per ALCT BX)

  // If set selective unpacking mode
  // hardcoded examiner mask below to check for DCC and DDU level errors will be used first
  // then examinerMask for CSC level errors will be used during unpacking of each CSC block
  unsigned long dccBinCheckMask = 0x06080016;

  // Post-LS1 FED/DDU ID mapping fix
  const unsigned postLS1_map[] = {841, 842, 843, 844, 845, 846, 847, 848, 849, 831, 832, 833,
                                  834, 835, 836, 837, 838, 839, 861, 862, 863, 864, 865, 866,
                                  867, 868, 869, 851, 852, 853, 854, 855, 856, 857, 858, 859};

  // For new CSC readout layout, which wont include DCCs need to loop over DDU FED IDs. DCC IDs are included for backward compatibility with old data
  std::vector<unsigned int> cscFEDids;

  for (unsigned int id = FEDNumbering::MINCSCFEDID; id <= FEDNumbering::MAXCSCFEDID; ++id)  // loop over DCCs
  {
    cscFEDids.push_back(id);
  }

  for (unsigned int id = FEDNumbering::MINCSCDDUFEDID; id <= FEDNumbering::MAXCSCDDUFEDID; ++id)  // loop over DDUs
  {
    cscFEDids.push_back(id);
  }

  for (unsigned int i = 0; i < cscFEDids.size(); i++)  // loop over all CSC FEDs (DCCs and DDUs)
  {
    unsigned int id = cscFEDids[i];
    bool isDDU_FED = ((id >= FEDNumbering::MINCSCDDUFEDID) && (id <= FEDNumbering::MAXCSCDDUFEDID)) ? true : false;

    /// uncomment this for regional unpacking
    /// if (id!=SOME_ID) continue;

    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    unsigned long length = fedData.size();

    if (length >= 32)  ///if fed has data then unpack it
    {
      CSCDCCExaminer* examiner = nullptr;
      goodEvent = true;
      if (useExaminer)  ///examine event for integrity
      {
        // CSCDCCExaminer examiner;
        examiner = new CSCDCCExaminer();
        if (examinerMask & 0x40000)
          examiner->crcCFEB(true);
        if (examinerMask & 0x8000)
          examiner->crcTMB(true);
        if (examinerMask & 0x0400)
          examiner->crcALCT(true);
        examiner->setMask(examinerMask);

        /// If we have DCC or only DDU FED by checking FED ID set examiner to uswe DCC or DDU mode
        if (isDDU_FED) {
          if (examiner != nullptr)
            examiner->modeDDU(true);
        }

        const short unsigned int* data = (short unsigned int*)fedData.data();

        LogTrace("badData") << "Length: " << length / 2;
        // Event data hex dump
        /*
              short unsigned * buf = (short unsigned int *)fedData.data();
                std::cout <<std::endl<<length/2<<" words of data:"<<std::endl;
                for (short unsigned int i=0;i<length/2;i++) {
                printf("%04x %04x %04x %04x\n",buf[i+3],buf[i+2],buf[i+1],buf[i]);
                i+=3;
                }
                      */

        int res = examiner->check(data, long(fedData.size() / 2));
        if (res < 0) {
          goodEvent = false;
        } else {
          if (useSelectiveUnpacking)
            goodEvent = !(examiner->errors() & dccBinCheckMask);
          else
            goodEvent = !(examiner->errors() & examinerMask);
        }

        /*
        std::cout << "FED" << std::dec << id << " size:" << fedData.size() << " good:" << goodEvent << " errs 0x"
              << std::hex << examiner->errors() << std::dec << std::endl;
         */

        // Fill Format status digis per FED
        // Remove examiner->errors() != 0 check if we need to put status digis for every event
        if (useFormatStatus && (examiner->errors() != 0))
          // formatStatusProduct->insertDigi(CSCDetId(1,1,1,1,1), CSCDCCFormatStatusDigi(id,examiner,dccBinCheckMask));
          formatStatusProduct->insertDigi(CSCDetId(1, 1, 1, 1, 1),
                                          CSCDCCFormatStatusDigi(id,
                                                                 dccBinCheckMask,
                                                                 examiner->getMask(),
                                                                 examiner->errors(),
                                                                 examiner->errorsDetailedDDU(),
                                                                 examiner->errorsDetailed(),
                                                                 examiner->payloadDetailed(),
                                                                 examiner->statusDetailed()));
      }

      /// Visualization of raw data
      if (visualFEDInspect || formatedEventDump) {
        if (!goodEvent || formatedEventDump) {
          short unsigned* buf = (short unsigned int*)fedData.data();
          visual_raw(length / 2, id, (int)e.id().run(), (int)e.id().event(), visualFEDShort, formatedEventDump, buf);
        }
      }

      if (goodEvent) {
        ///get a pointer to data and pass it to constructor for unpacking

        CSCDCCExaminer* ptrExaminer = examiner;
        if (!useSelectiveUnpacking)
          ptrExaminer = nullptr;

        std::vector<CSCDDUEventData> fed_Data;
        std::vector<CSCDDUEventData>* ptr_fedData = &fed_Data;

        /// set default detid to that for E=+z, S=1, R=1, C=1, L=1
        CSCDetId layer(1, 1, 1, 1, 1);

        if (isDDU_FED)  // Use new DDU FED readout mode
        {
          CSCDDUEventData single_dduData((short unsigned int*)fedData.data(), ptrExaminer);
          fed_Data.push_back(single_dduData);

          // if(instantiateDQM) monitor->process(examiner, &single_dduData);

        } else  // Use old DCC FED readout mode
        {
          CSCDCCEventData dccData((short unsigned int*)fedData.data(), ptrExaminer);

          //std::cout << " DCC Size [UNPK] " << dccData.sizeInWords() << std::endl;

          if (instantiateDQM)
            monitor->process(examiner, &dccData);

          ///get a reference to dduData
          // const std::vector<CSCDDUEventData> & dduData = dccData.dduData();
          // ptr_fedData = &(dccData.dduData());
          fed_Data = dccData.dduData();

          if (unpackStatusDigis) {
            /// DCC Trailer 2 added to dcc status product (to access TTS from DCC)
            short unsigned* bufForDcc = (short unsigned int*)fedData.data();

            //std::cout << "FED Length: " << std::dec << length/2 <<
            //" Trailer 2: " << std::hex << bufForDcc[length/2-4] << std::endl;

            dccStatusProduct->insertDigi(layer,
                                         CSCDCCStatusDigi(dccData.dccHeader().data(),
                                                          dccData.dccTrailer().data(),
                                                          examiner->errors(),
                                                          bufForDcc[length / 2 - 4]));
          }
        }

        const std::vector<CSCDDUEventData>& dduData = *ptr_fedData;

        for (unsigned int iDDU = 0; iDDU < dduData.size(); ++iDDU)  // loop over DDUs
        {
          /// skip the DDU if its data has serious errors
          /// define a mask for serious errors
          if (dduData[iDDU].trailer().errorstat() & errorMask) {
            LogTrace("CSCDCCUnpacker|CSCRawToDigi")
                << "FED ID" << id << " DDU# " << iDDU << " has serious error - no digis unpacked! " << std::hex
                << dduData[iDDU].trailer().errorstat();
            continue;  // to next iteration of DDU loop
          }

          if (unpackStatusDigis)
            dduStatusProduct->insertDigi(
                layer,
                CSCDDUStatusDigi(dduData[iDDU].header().data(),
                                 dduData[iDDU].trailer().data(),
                                 /// DDU Trailer 0 added to ddu status product (to access TTS from DDU)
                                 dduData[iDDU].trailer0()));

          ///get a reference to chamber data
          const std::vector<CSCEventData>& cscData = dduData[iDDU].cscData();

          // if (cscData.size() != 0) std::cout << "FED" << id << " DDU Source ID: " << dduData[iDDU].header().source_id() << " firmware version: " << dduData[iDDU].header().format_version() << std::endl;

          for (unsigned int iCSC = 0; iCSC < cscData.size(); ++iCSC)  // loop over CSCs
          {
            ///first process chamber-wide digis such as LCT

            int vmecrate = b904Setup ? b904vmecrate : cscData[iCSC].dmbHeader()->crateID();
            int dmb = b904Setup ? b904dmb : cscData[iCSC].dmbHeader()->dmbID();

            int icfeb = 0;   /// default value for all digis not related to cfebs
            int ilayer = 0;  /// layer=0 flags entire chamber

            if (debug)
              LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "crate = " << vmecrate << "; dmb = " << dmb;

            if ((vmecrate >= 1) && (vmecrate <= 60) && (dmb >= 1) && (dmb <= 10) && (dmb != 6)) {
              layer = pcrate->detId(vmecrate, dmb, icfeb, ilayer);
            } else {
              LogTrace("CSCDCCUnpacker|CSCRawToDigi") << " detID input out of range!!! ";
              LogTrace("CSCDCCUnpacker|CSCRawToDigi") << " skipping chamber vme= " << vmecrate << " dmb= " << dmb;
              continue;  // to next iteration of iCSC loop
            }

            /// For Post-LS1 readout only. Check Chamber->FED/DDU mapping consistency.
            /// Skip chambers (special case of data corruption), which report wrong ID and pose as different chamber
            if (isDDU_FED) {
              unsigned int dduid = cscmapping->ddu(layer);
              if ((dduid >= 1) && (dduid <= 36)) {
                dduid = postLS1_map[dduid - 1];  // Fix for Post-LS1 FED/DDU IDs mappings
                // std::cout << "CSC " << layer << " -> " << id << ":" << dduid << ":" << vmecrate << ":" << dmb << std::endl;
              }

              /// Do not skip chamber data if mapping check is disabled or b904 setup data file is used
              if ((!disableMappingCheck) && (!b904Setup) && (id != dduid)) {
                LogTrace("CSCDDUUnpacker|CSCRawToDigi") << " CSC->FED/DDU mapping inconsistency!!! ";
                LogTrace("CSCDCCUnpacker|CSCRawToDigi")
                    << "readout FED/DDU ID=" << id << " expected ID=" << dduid << ", skipping chamber " << layer
                    << " vme= " << vmecrate << " dmb= " << dmb;
                continue;
              }
            }

            /// check alct data integrity
            int nalct = cscData[iCSC].dmbHeader()->nalct();
            bool goodALCT = false;
            //if (nalct&&(cscData[iCSC].dataPresent>>6&0x1)==1) {
            if (nalct && cscData[iCSC].alctHeader()) {
              if (cscData[iCSC].alctHeader()->check()) {
                goodALCT = true;
              } else {
                LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "not storing ALCT digis; alct is bad or not present";
              }
            } else {
              if (debug)
                LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "nALCT==0 !!!";
            }

            /// fill alct digi
            if (goodALCT) {
              std::vector<CSCALCTDigi> alctDigis = cscData[iCSC].alctHeader()->ALCTDigis();
              if (SuppressZeroLCT) {
                std::vector<CSCALCTDigi> alctDigis_0;
                for (int unsigned i = 0; i < alctDigis.size(); ++i) {
                  if (alctDigis[i].isValid()) {
                    if (debug)
                      LogTrace("CSCDCCUnpacker|CSCRawToDigi") << alctDigis[i] << std::endl;
                    alctDigis_0.push_back(alctDigis[i]);
                  }
                }
                alctProduct->move(std::make_pair(alctDigis_0.begin(), alctDigis_0.end()), layer);
              } else
                alctProduct->move(std::make_pair(alctDigis.begin(), alctDigis.end()), layer);

              /// fill Run3 anode HMT Shower digis
              /// anode shower digis vector per ALCT BX from ALCT data
              if (useCSCShowers_) {
                std::vector<CSCShowerDigi> anodeShowerDigisALCT = cscData[iCSC].alctHeader()->alctShowerDigis();
                anodeShowerProductALCT->move(std::make_pair(anodeShowerDigisALCT.begin(), anodeShowerDigisALCT.end()),
                                             layer);
              }
            }

            ///check tmb data integrity
            int nclct = cscData[iCSC].dmbHeader()->nclct();
            bool goodTMB = false;
            //	    if (nclct&&(cscData[iCSC].dataPresent>>5&0x1)==1) {
            if (nclct && cscData[iCSC].tmbData()) {
              if (cscData[iCSC].tmbHeader()->check()) {
                if (cscData[iCSC].comparatorData()->check())
                  goodTMB = true;
              } else {
                LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "one of TMB checks failed! not storing TMB digis ";
              }
            } else {
              if (debug)
                LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "nCLCT==0 !!!";
            }

            /// fill correlatedlct and clct digis
            if (goodTMB) {
              std::vector<CSCCorrelatedLCTDigi> correlatedlctDigis =
                  cscData[iCSC].tmbHeader()->CorrelatedLCTDigis(layer.rawId());
              if (SuppressZeroLCT) {
                std::vector<CSCCorrelatedLCTDigi> correlatedlctDigis_0;
                for (int unsigned i = 0; i < correlatedlctDigis.size(); ++i) {
                  if (correlatedlctDigis[i].isValid()) {
                    if (debug)
                      LogTrace("CSCDCCUnpacker|CSCRawToDigi") << correlatedlctDigis[i] << std::endl;
                    correlatedlctDigis_0.push_back(correlatedlctDigis[i]);
                  }
                }
                corrlctProduct->move(std::make_pair(correlatedlctDigis_0.begin(), correlatedlctDigis_0.end()), layer);
              } else
                corrlctProduct->move(std::make_pair(correlatedlctDigis.begin(), correlatedlctDigis.end()), layer);

              std::vector<CSCCLCTDigi> clctDigis = cscData[iCSC].tmbHeader()->CLCTDigis(layer.rawId());
              if (SuppressZeroLCT) {
                std::vector<CSCCLCTDigi> clctDigis_0;
                for (int unsigned i = 0; i < clctDigis.size(); ++i) {
                  if (clctDigis[i].isValid()) {
                    if (debug)
                      LogTrace("CSCDCCUnpacker|CSCRawToDigi") << clctDigis[i] << std::endl;
                    clctDigis_0.push_back(clctDigis[i]);
                  }
                }
                clctProduct->move(std::make_pair(clctDigis_0.begin(), clctDigis_0.end()), layer);
              } else
                clctProduct->move(std::make_pair(clctDigis.begin(), clctDigis.end()), layer);

              /// fill Run3 HMT Shower digis
              if (useCSCShowers_) {
                /// (O)TMB Shower digi sent to MPC LCT trigger data
                CSCShowerDigi lctShowerDigi = cscData[iCSC].tmbHeader()->showerDigi(layer.rawId());
                if (lctShowerDigi.isValid()) {
                  std::vector<CSCShowerDigi> lctShowerDigis;
                  lctShowerDigis.push_back(lctShowerDigi);
                  lctShowerProduct->move(std::make_pair(lctShowerDigis.begin(), lctShowerDigis.end()), layer);
                }

                /// anode shower digis from OTMB header data
                CSCShowerDigi anodeShowerDigiOTMB = cscData[iCSC].tmbHeader()->anodeShowerDigi(layer.rawId());
                if (anodeShowerDigiOTMB.isValid()) {
                  std::vector<CSCShowerDigi> anodeShowerDigis;
                  anodeShowerDigis.push_back(anodeShowerDigiOTMB);
                  anodeShowerProductOTMB->move(std::make_pair(anodeShowerDigis.begin(), anodeShowerDigis.end()), layer);
                }

                /// cathode shower digis from OTMB header data
                CSCShowerDigi cathodeShowerDigiOTMB = cscData[iCSC].tmbHeader()->cathodeShowerDigi(layer.rawId());
                if (cathodeShowerDigiOTMB.isValid()) {
                  std::vector<CSCShowerDigi> cathodeShowerDigis;
                  cathodeShowerDigis.push_back(cathodeShowerDigiOTMB);
                  cathodeShowerProductOTMB->move(std::make_pair(cathodeShowerDigis.begin(), cathodeShowerDigis.end()),
                                                 layer);
                }
              }

              /// fill CSC-RPC or CSC-GEMs digis
              if (cscData[iCSC].tmbData()->checkSize()) {
                if (useRPCs_ && cscData[iCSC].tmbData()->hasRPC()) {
                  std::vector<CSCRPCDigi> rpcDigis = cscData[iCSC].tmbData()->rpcData()->digis();
                  rpcProduct->move(std::make_pair(rpcDigis.begin(), rpcDigis.end()), layer);
                }

                /// fill CSC-GEM GEMPadCluster digis
                if (useGEMs_ && cscData[iCSC].tmbData()->hasGEM()) {
                  for (int unsigned igem = 0; igem < (int unsigned)(cscData[iCSC].tmbData()->gemData()->numGEMs());
                       ++igem) {
                    int gem_chamber = layer.chamber();
                    int gem_region = (layer.endcap() == 1) ? 1 : -1;
                    // Loop over GEM layer eta/rolls
                    for (unsigned ieta = 0; ieta < 8; ieta++) {
                      // GE11 eta/roll collection addressing according to GEMDetID definition is 1-8 (eta 8 being closest to beampipe)
                      GEMDetId gemid(gem_region, layer.ring(), layer.station(), igem + 1, gem_chamber, ieta + 1);
                      // GE11 trigger data format reports eta/rolls in 0-7 range (eta 0 being closest to beampipe)
                      // mapping agreement is that real data eta needs to be reversed from 0-7 to 8-1 for GEMDetId collection convention
                      std::vector<GEMPadDigiCluster> gemDigis = cscData[iCSC].tmbData()->gemData()->etaDigis(
                          igem, 7 - ieta, cscData[iCSC].tmbHeader()->ALCTMatchTime());
                      if (!gemDigis.empty()) {
                        gemProduct->move(std::make_pair(gemDigis.begin(), gemDigis.end()), gemid);
                      }
                    }
                  }
                }
              } else
                LogTrace("CSCDCCUnpacker|CSCRawToDigi") << " TMBData check size failed!";
            }

            /// fill cfeb status digi
            if (unpackStatusDigis) {
              for (icfeb = 0; icfeb < CSCConstants::MAX_CFEBS_RUN2; ++icfeb)  ///loop over status digis
              {
                if (cscData[iCSC].cfebData(icfeb) != nullptr)
                  cfebStatusProduct->insertDigi(layer, cscData[iCSC].cfebData(icfeb)->statusDigi());
              }
              /// fill dmb status digi
              dmbStatusProduct->insertDigi(
                  layer, CSCDMBStatusDigi(cscData[iCSC].dmbHeader()->data(), cscData[iCSC].dmbTrailer()->data()));
              if (goodTMB)
                tmbStatusProduct->insertDigi(
                    layer,
                    CSCTMBStatusDigi(cscData[iCSC].tmbHeader()->data(), cscData[iCSC].tmbData()->tmbTrailer()->data()));
              if (goodALCT)
                alctStatusProduct->insertDigi(
                    layer, CSCALCTStatusDigi(cscData[iCSC].alctHeader()->data(), cscData[iCSC].alctTrailer()->data()));
            }

            /// fill wire, strip and comparator digis...
            for (int ilayer = CSCDetId::minLayerId(); ilayer <= CSCDetId::maxLayerId(); ++ilayer) {
              /// set layer, dmb and vme are valid because already checked in line 240
              // (You have to be kidding. Line 240 in whose universe?)

              // Allocate all ME1/1 wire digis to ring 1
              layer = pcrate->detId(vmecrate, dmb, 0, ilayer);
              {
                std::vector<CSCWireDigi> wireDigis = cscData[iCSC].wireDigis(ilayer);
                wireProduct->move(std::make_pair(wireDigis.begin(), wireDigis.end()), layer);
              }

              for (icfeb = 0; icfeb < CSCConstants::MAX_CFEBS_RUN2; ++icfeb) {
                layer = pcrate->detId(vmecrate, dmb, icfeb, ilayer);
                if (cscData[iCSC].cfebData(icfeb) && cscData[iCSC].cfebData(icfeb)->check()) {
                  std::vector<CSCStripDigi> stripDigis;
                  cscData[iCSC].cfebData(icfeb)->digis(layer.rawId(), stripDigis);
                  stripProduct->move(std::make_pair(stripDigis.begin(), stripDigis.end()), layer);
                }
              }

              if (goodTMB && (cscData[iCSC].tmbHeader() != nullptr)) {
                int nCFEBs = cscData[iCSC].tmbHeader()->NCFEBs();
                for (icfeb = 0; icfeb < nCFEBs; ++icfeb) {
                  layer = pcrate->detId(vmecrate, dmb, icfeb, ilayer);
                  std::vector<CSCComparatorDigi> comparatorDigis =
                      cscData[iCSC].comparatorData()->comparatorDigis(layer.rawId(), icfeb);
                  // Set cfeb=0, so that ME1/a and ME1/b comparators go to
                  // ring 1.
                  layer = pcrate->detId(vmecrate, dmb, 0, ilayer);
                  comparatorProduct->move(std::make_pair(comparatorDigis.begin(), comparatorDigis.end()), layer);
                }
              }  // end of loop over cfebs
            }    // end of loop over layers
          }      // end of loop over chambers
        }        // endof loop over DDUs
      }          // end of good event
      else {
        LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "ERROR! Examiner rejected FED #" << id;
        if (examiner) {
          for (int i = 0; i < examiner->nERRORS; ++i) {
            if (((examinerMask & examiner->errors()) >> i) & 0x1)
              LogTrace("CSCDCCUnpacker|CSCRawToDigi") << examiner->errName(i);
          }
          if (debug) {
            LogTrace("CSCDCCUnpacker|CSCRawToDigi")
                << " Examiner errors:0x" << std::hex << examiner->errors() << " & 0x" << examinerMask << " = "
                << (examiner->errors() & examinerMask);
          }
        }

        // dccStatusProduct->insertDigi(CSCDetId(1,1,1,1,1), CSCDCCStatusDigi(examiner->errors()));
        // if(instantiateDQM)  monitor->process(examiner, NULL);
      }
      if (examiner != nullptr)
        delete examiner;
    }  // end of if fed has data
  }    // end of loop over DCCs
  // put into the event
  e.put(std::move(wireProduct), "MuonCSCWireDigi");
  e.put(std::move(stripProduct), "MuonCSCStripDigi");
  e.put(std::move(alctProduct), "MuonCSCALCTDigi");
  e.put(std::move(clctProduct), "MuonCSCCLCTDigi");
  e.put(std::move(comparatorProduct), "MuonCSCComparatorDigi");
  e.put(std::move(corrlctProduct), "MuonCSCCorrelatedLCTDigi");

  if (useFormatStatus)
    e.put(std::move(formatStatusProduct), "MuonCSCDCCFormatStatusDigi");

  if (unpackStatusDigis) {
    e.put(std::move(cfebStatusProduct), "MuonCSCCFEBStatusDigi");
    e.put(std::move(dmbStatusProduct), "MuonCSCDMBStatusDigi");
    e.put(std::move(tmbStatusProduct), "MuonCSCTMBStatusDigi");
    e.put(std::move(dduStatusProduct), "MuonCSCDDUStatusDigi");
    e.put(std::move(dccStatusProduct), "MuonCSCDCCStatusDigi");
    e.put(std::move(alctStatusProduct), "MuonCSCALCTStatusDigi");
  }

  if (useRPCs_) {
    e.put(std::move(rpcProduct), "MuonCSCRPCDigi");
  }
  if (useGEMs_) {
    e.put(std::move(gemProduct), "MuonGEMPadDigiCluster");
  }
  if (useCSCShowers_) {
    e.put(std::move(lctShowerProduct), "MuonCSCShowerDigi");
    e.put(std::move(anodeShowerProductOTMB), "MuonCSCShowerDigiAnode");
    e.put(std::move(cathodeShowerProductOTMB), "MuonCSCShowerDigiCathode");
    e.put(std::move(anodeShowerProductALCT), "MuonCSCShowerDigiAnodeALCT");
  }
  if (printEventNumber)
    LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "[CSCDCCUnpacker]: " << numOfEvents << " events processed ";
}

/// Visualization of raw data

void CSCDCCUnpacker::visual_raw(
    int hl, int id, int run, int event, bool fedshort, bool fDump, short unsigned int* buf) const {
  std::cout << std::endl << std::endl << std::endl;
  std::cout << "Run: " << run << " Event: " << event << std::endl;
  std::cout << std::endl << std::endl;
  if (formatedEventDump)
    std::cout << "FED-" << id << "  "
              << "(scroll down to see summary)" << std::endl;
  else
    std::cout << "Problem seems in FED-" << id << "  "
              << "(scroll down to see summary)" << std::endl;
  std::cout << "********************************************************************************" << std::endl;
  std::cout << hl << " words of data:" << std::endl;

  //================================================
  // FED codes in DCC
  std::vector<int> dcc_id;
  int dcc_h1_id = 0;
  // Current codes
  for (int i = 750; i < 758; i++)
    dcc_id.push_back(i);
  // Codes for upgrade
  for (int i = 830; i < 838; i++)
    dcc_id.push_back(i);

  char dcc_common[] = "DCC-";

  //================================================
  // DDU codes per FED
  std::vector<int> ddu_id;
  int ddu_h1_12_13 = 0;
  for (int i = 1; i < 37; i++)
    ddu_id.push_back(i);
  // For DDU Headers and tarailers
  char ddu_common[] = "DDU-";
  char ddu_header1[] = "Header 1";
  char ddu_header2[] = "Header 2";
  char ddu_header3[] = "Header 3";
  char ddu_trail1[] = "Trailer 1", ddu_trail2[] = "Trailer 2", ddu_trail3[] = "Trailer 3";
  // For Header 2
  char ddu_trailer1_bit[] = {'8', '0', '0', '0', 'f', 'f', 'f', 'f', '8', '0', '0', '0', '8', '0', '0', '0'};
  char ddu_trailer3_bit[] = {'a'};
  // Corrupted Trailers
  char ddu_tr1_err_common[] = "Incomplet";
  //====================================================

  //DMB
  char dmb_common[] = "DMB", dmb_header1[] = "Header 1", dmb_header2[] = "Header 2";
  char dmb_common_crate[] = "crate:", dmb_common_slot[] = "slot:";
  char dmb_common_l1a[] = "L1A:";
  char dmb_header1_bit[] = {'9', '9', '9', '9'};
  char dmb_header2_bit[] = {'a', 'a', 'a', 'a'};
  char dmb_tr1[] = "Trailer 1", dmb_tr2[] = "Trailer 2";
  char dmb_tr1_bit[] = {'f', 'f', 'f', 'f'}, dmb_tr2_bit[] = {'e', 'e', 'e', 'e'};

  //=====================================================

  // ALCT
  char alct_common[] = "ALCT", alct_header1[] = "Header 1", alct_header2[] = "Header 2";
  char alct_common_bxn[] = "BXN:";
  char alct_common_wcnt2[] = "| Actual word count:";
  char alct_common_wcnt1[] = "Expected word count:";
  char alct_header1_bit[] = {'d', 'd', 'd', 'd', 'b', '0', 'a'};
  char alct_header2_bit[] = {'0', '0', '0', '0'};
  char alct_tr1[] = "Trailer 1";

  //======================================================

  //TMB
  char tmb_common[] = "TMB", tmb_header1[] = "Header", tmb_tr1[] = "Trailer";
  char tmb_header1_bit[] = {'d', 'd', 'd', 'd', 'b', '0', 'c'};
  char tmb_tr1_bit[] = {'d', 'd', 'd', 'd', 'e', '0', 'f'};

  //======================================================

  //CFEB
  char cfeb_common[] = "CFEB", cfeb_tr1[] = "Trailer", cfeb_b[] = "B-word";
  char cfeb_common_sample[] = "sample:";

  //======================================================

  //Auxiliary variables

  // Bufers
  int word_lines = hl / 4;
  char tempbuf[80];
  char tempbuf1[130];
  char tempbuf_short[17];
  char sign1[] = "  --->| ";

  // Counters
  int word_numbering = 0;
  int ddu_inst_i = 0, ddu_inst_n = 0, ddu_inst_l1a = 0;
  int ddu_inst_bxn = 0;
  int dmb_inst_crate = 0, dmb_inst_slot = 0, dmb_inst_l1a = 0;
  int cfeb_sample = 0;
  int alct_inst_l1a = 0;
  int alct_inst_bxn = 0;
  int alct_inst_wcnt1 = 0;
  int alct_inst_wcnt2 = 0;
  int alct_start = 0;
  int alct_stop = 0;
  int tmb_inst_l1a = 0;
  int tmb_inst_wcnt1 = 0;
  int tmb_inst_wcnt2 = 0;
  int tmb_start = 0;
  int tmb_stop = 0;
  int dcc_h1_check = 0;

  //Flags
  int ddu_h2_found = 0;  //DDU Header 2 found
  int w = 0;

  //Logic variables
  const int sz1 = 5;
  bool dcc_check = false;
  bool ddu_h2_check[sz1] = {false};
  bool ddu_h1_check = false;
  bool dmb_h1_check[sz1] = {false};
  bool dmb_h2_check[sz1] = {false};
  bool ddu_h2_h1 = false;
  bool ddu_tr1_check[sz1] = {false};
  bool alct_h1_check[sz1] = {false};
  bool alct_h2_check[sz1] = {false};
  bool alct_tr1_check[sz1] = {false};
  bool dmb_tr1_check[sz1] = {false};
  bool dmb_tr2_check[sz1] = {false};
  bool tmb_h1_check[sz1] = {false};
  bool tmb_tr1_check[sz1] = {false};
  bool cfeb_tr1_check[sz1] = {false};
  bool cfeb_b_check[sz1] = {false};
  bool ddu_tr1_bad_check[sz1] = {false};
  bool extraction = fedshort;

  //Summary vectors
  //DDU
  std::vector<int> ddu_h1_coll;
  std::vector<int> ddu_h1_n_coll;
  std::vector<int> ddu_h2_coll;
  std::vector<int> ddu_h3_coll;
  std::vector<int> ddu_t1_coll;
  std::vector<int> ddu_t2_coll;
  std::vector<int> ddu_t3_coll;
  std::vector<int> ddu_l1a_coll;
  std::vector<int> ddu_bxn_coll;
  //DMB
  std::vector<int> dmb_h1_coll;
  std::vector<int> dmb_h2_coll;
  std::vector<int> dmb_t1_coll;
  std::vector<int> dmb_t2_coll;
  std::vector<int> dmb_crate_coll;
  std::vector<int> dmb_slot_coll;
  std::vector<int> dmb_l1a_coll;
  //ALCT
  std::vector<int> alct_h1_coll;
  std::vector<int> alct_h2_coll;
  std::vector<int> alct_t1_coll;
  std::vector<int> alct_l1a_coll;
  std::vector<int> alct_bxn_coll;
  std::vector<int> alct_wcnt1_coll;
  std::vector<int> alct_wcnt2_coll;
  std::vector<int> alct_wcnt2_id_coll;
  //TMB
  std::vector<int> tmb_h1_coll;
  std::vector<int> tmb_t1_coll;
  std::vector<int> tmb_l1a_coll;
  std::vector<int> tmb_wcnt1_coll;
  std::vector<int> tmb_wcnt2_coll;
  //CFEB
  std::vector<int> cfeb_t1_coll;

  //========================================================

  // DCC Header and Ttrailer information
  char dcc_header1[] = "DCC Header 1";
  char dcc_header2[] = "DCC Header 2";
  char dcc_trail1[] = "DCC Trailer 1", dcc_trail1_bit[] = {'e'};
  char dcc_trail2[] = "DCC Trailer 2", dcc_trail2_bit[] = {'a'};
  //=========================================================

  for (int i = 0; i < hl; i++) {
    ++word_numbering;
    for (int j = -1; j < 4; j++) {
      sprintf(tempbuf_short,
              "%04x%04x%04x%04x",
              buf[i + 4 * (j - 1) + 3],
              buf[i + 4 * (j - 1) + 2],
              buf[i + 4 * (j - 1) + 1],
              buf[i + 4 * (j - 1)]);

      // WARNING in 5_0_X for time being
      ddu_h2_found++;
      ddu_h2_found--;

      ddu_h2_check[j] = ((buf[i + 4 * (j - 1) + 1] == 0x8000) && (buf[i + 4 * (j - 1) + 2] == 0x0001) &&
                         (buf[i + 4 * (j - 1) + 3] == 0x8000));

      ddu_tr1_check[j] = ((tempbuf_short[0] == ddu_trailer1_bit[0]) && (tempbuf_short[1] == ddu_trailer1_bit[1]) &&
                          (tempbuf_short[2] == ddu_trailer1_bit[2]) && (tempbuf_short[3] == ddu_trailer1_bit[3]) &&
                          (tempbuf_short[4] == ddu_trailer1_bit[4]) && (tempbuf_short[5] == ddu_trailer1_bit[5]) &&
                          (tempbuf_short[6] == ddu_trailer1_bit[6]) && (tempbuf_short[7] == ddu_trailer1_bit[7]) &&
                          (tempbuf_short[8] == ddu_trailer1_bit[8]) && (tempbuf_short[9] == ddu_trailer1_bit[9]) &&
                          (tempbuf_short[10] == ddu_trailer1_bit[10]) && (tempbuf_short[11] == ddu_trailer1_bit[11]) &&
                          (tempbuf_short[12] == ddu_trailer1_bit[12]) && (tempbuf_short[13] == ddu_trailer1_bit[13]) &&
                          (tempbuf_short[14] == ddu_trailer1_bit[14]) && (tempbuf_short[15] == ddu_trailer1_bit[15]));

      dmb_h1_check[j] = ((tempbuf_short[0] == dmb_header1_bit[0]) && (tempbuf_short[4] == dmb_header1_bit[1]) &&
                         (tempbuf_short[8] == dmb_header1_bit[2]) && (tempbuf_short[12] == dmb_header1_bit[3]));

      dmb_h2_check[j] = ((tempbuf_short[0] == dmb_header2_bit[0]) && (tempbuf_short[4] == dmb_header2_bit[1]) &&
                         (tempbuf_short[8] == dmb_header2_bit[2]) && (tempbuf_short[12] == dmb_header2_bit[3]));
      alct_h1_check[j] = ((tempbuf_short[0] == alct_header1_bit[0]) && (tempbuf_short[4] == alct_header1_bit[1]) &&
                          (tempbuf_short[8] == alct_header1_bit[2]) && (tempbuf_short[12] == alct_header1_bit[3]) &&
                          (tempbuf_short[13] == alct_header1_bit[4]) && (tempbuf_short[14] == alct_header1_bit[5]) &&
                          (tempbuf_short[15] == alct_header1_bit[6]));
      alct_h2_check[j] = (((tempbuf_short[0] == alct_header2_bit[0]) && (tempbuf_short[1] == alct_header2_bit[1]) &&
                           (tempbuf_short[2] == alct_header2_bit[2]) && (tempbuf_short[3] == alct_header2_bit[3])) ||
                          ((tempbuf_short[4] == alct_header2_bit[0]) && (tempbuf_short[5] == alct_header2_bit[1]) &&
                           (tempbuf_short[6] == alct_header2_bit[2]) && (tempbuf_short[7] == alct_header2_bit[3])) ||
                          ((tempbuf_short[8] == alct_header2_bit[0]) && (tempbuf_short[9] == alct_header2_bit[1]) &&
                           (tempbuf_short[10] == alct_header2_bit[2]) && (tempbuf_short[11] == alct_header2_bit[3])) ||
                          ((tempbuf_short[12] == alct_header2_bit[0]) && (tempbuf_short[13] == alct_header2_bit[1]) &&
                           (tempbuf_short[14] == alct_header2_bit[2]) && (tempbuf_short[15] == alct_header2_bit[3]))
                          //(tempbuf_short[4]==alct_header2_bit[4])&&(tempbuf_short[5]==alct_header2_bit[5])
      );
      // ALCT Trailers
      alct_tr1_check[j] =
          (((buf[i + 4 * (j - 1)] & 0xFFFF) == 0xDE0D) && ((buf[i + 4 * (j - 1) + 1] & 0xF800) == 0xD000) &&
           ((buf[i + 4 * (j - 1) + 2] & 0xF800) == 0xD000) && ((buf[i + 4 * (j - 1) + 3] & 0xF000) == 0xD000));
      // DMB Trailers
      dmb_tr1_check[j] = ((tempbuf_short[0] == dmb_tr1_bit[0]) && (tempbuf_short[4] == dmb_tr1_bit[1]) &&
                          (tempbuf_short[8] == dmb_tr1_bit[2]) && (tempbuf_short[12] == dmb_tr1_bit[3]));
      dmb_tr2_check[j] = ((tempbuf_short[0] == dmb_tr2_bit[0]) && (tempbuf_short[4] == dmb_tr2_bit[1]) &&
                          (tempbuf_short[8] == dmb_tr2_bit[2]) && (tempbuf_short[12] == dmb_tr2_bit[3]));
      // TMB
      tmb_h1_check[j] = ((tempbuf_short[0] == tmb_header1_bit[0]) && (tempbuf_short[4] == tmb_header1_bit[1]) &&
                         (tempbuf_short[8] == tmb_header1_bit[2]) && (tempbuf_short[12] == tmb_header1_bit[3]) &&
                         (tempbuf_short[13] == tmb_header1_bit[4]) && (tempbuf_short[14] == tmb_header1_bit[5]) &&
                         (tempbuf_short[15] == tmb_header1_bit[6]));
      tmb_tr1_check[j] = ((tempbuf_short[0] == tmb_tr1_bit[0]) && (tempbuf_short[4] == tmb_tr1_bit[1]) &&
                          (tempbuf_short[8] == tmb_tr1_bit[2]) && (tempbuf_short[12] == tmb_tr1_bit[3]) &&
                          (tempbuf_short[13] == tmb_tr1_bit[4]) && (tempbuf_short[14] == tmb_tr1_bit[5]) &&
                          (tempbuf_short[15] == tmb_tr1_bit[6]));
      // CFEB
      cfeb_tr1_check[j] =
          (((buf[i + 4 * (j - 1) + 1] & 0xF000) == 0x7000) && ((buf[i + 4 * (j - 1) + 2] & 0xF000) == 0x7000) &&
           ((buf[i + 4 * (j - 1) + 1] != 0x7FFF) || (buf[i + 4 * (j - 1) + 2] != 0x7FFF)) &&
           ((buf[i + 4 * (j - 1) + 3] == 0x7FFF) || ((buf[i + 4 * (j - 1) + 3] & buf[i + 4 * (j - 1)]) == 0x0 &&
                                                     (buf[i + 4 * (j - 1) + 3] + buf[i + 4 * (j - 1)] == 0x7FFF))));
      cfeb_b_check[j] =
          (((buf[i + 4 * (j - 1) + 3] & 0xF000) == 0xB000) && ((buf[i + 4 * (j - 1) + 2] & 0xF000) == 0xB000) &&
           ((buf[i + 4 * (j - 1) + 1] & 0xF000) == 0xB000) && ((buf[i + 4 * (j - 1)] = 3 & 0xF000) == 0xB000));
      // DDU Trailers with errors
      ddu_tr1_bad_check[j] =
          ((tempbuf_short[0] != ddu_trailer1_bit[0]) &&
           //(tempbuf_short[1]!=ddu_trailer1_bit[1])&&(tempbuf_short[2]!=ddu_trailer1_bit[2])&&
           //(tempbuf_short[3]==ddu_trailer1_bit[3])&&
           (tempbuf_short[4] != ddu_trailer1_bit[4]) &&
           //(tempbuf_short[5]==ddu_trailer1_bit[5])&&
           //(tempbuf_short[6]==ddu_trailer1_bit[6])&&(tempbuf_short[7]==ddu_trailer1_bit[7])&&
           (tempbuf_short[8] == ddu_trailer1_bit[8]) && (tempbuf_short[9] == ddu_trailer1_bit[9]) &&
           (tempbuf_short[10] == ddu_trailer1_bit[10]) && (tempbuf_short[11] == ddu_trailer1_bit[11]) &&
           (tempbuf_short[12] == ddu_trailer1_bit[12]) && (tempbuf_short[13] == ddu_trailer1_bit[13]) &&
           (tempbuf_short[14] == ddu_trailer1_bit[14]) && (tempbuf_short[15] == ddu_trailer1_bit[15]));
    }

    // DDU Header 2 next to Header 1
    ddu_h2_h1 = ddu_h2_check[2];

    sprintf(tempbuf_short, "%04x%04x%04x%04x", buf[i + 3], buf[i + 2], buf[i + 1], buf[i]);

    // Looking for DDU Header 1
    ddu_h1_12_13 = (buf[i] >> 8);
    for (int kk = 0; kk < 36; kk++) {
      if (((buf[i + 3] & 0xF000) == 0x5000) && (ddu_h1_12_13 == ddu_id[kk]) && ddu_h2_h1) {
        ddu_h1_coll.push_back(word_numbering);
        ddu_h1_n_coll.push_back(ddu_id[kk]);
        ddu_inst_l1a = ((buf[i + 2] & 0xFFFF) + ((buf[i + 3] & 0x00FF) << 16));
        ddu_l1a_coll.push_back(ddu_inst_l1a);
        ddu_inst_bxn = (buf[i + 1] & 0xFFF0) >> 4;
        ddu_bxn_coll.push_back(ddu_inst_bxn);
        sprintf(tempbuf1,
                "%6i    %04x %04x %04x %04x%s%s%i %s%s %s %i %s %i",
                word_numbering,
                buf[i + 3],
                buf[i + 2],
                buf[i + 1],
                buf[i],
                sign1,
                ddu_common,
                ddu_id[kk],
                ddu_header1,
                sign1,
                dmb_common_l1a,
                ddu_inst_l1a,
                alct_common_bxn,
                ddu_inst_bxn);
        std::cout << tempbuf1 << std::endl;
        w = 0;
        ddu_h1_check = true;
        cfeb_sample = 0;
      }
    }

    // Looking for DCC Header 1
    dcc_h1_id = (((buf[i + 1] << 12) & 0xF000) >> 4) + (buf[i] >> 8);
    for (int dcci = 0; dcci < 16; dcci++) {
      if ((dcc_id[dcci] == dcc_h1_id) && (((buf[i + 3] & 0xF000) == 0x5000) && (!ddu_h1_check))) {
        sprintf(tempbuf1,
                "%6i    %04x %04x %04x %04x%s%s%i %s",
                word_numbering,
                buf[i + 3],
                buf[i + 2],
                buf[i + 1],
                buf[i],
                sign1,
                dcc_common,
                dcc_h1_id,
                dcc_header1);
        dcc_h1_check = word_numbering;
        w = 0;
        dcc_check = true;
        std::cout << tempbuf1 << std::endl;
      }
    }

    // Looking for DCC Header 2 and trailers
    if (((word_numbering - 1) == dcc_h1_check) && ((buf[i + 3] & 0xFF00) == 0xD900)) {
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              dcc_header2);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    } else if ((word_numbering == word_lines - 1) && (tempbuf_short[0] == dcc_trail1_bit[0])) {
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              dcc_trail1);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    } else if ((word_numbering == word_lines) && (tempbuf_short[0] == dcc_trail2_bit[0])) {
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              dcc_trail2);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }

    // DDU Header 2
    else if (ddu_h2_check[1]) {
      ddu_inst_i = ddu_h1_n_coll.size();  //ddu_inst_n=ddu_h1_n_coll[0];
      if (ddu_inst_i > 0) {
        ddu_inst_n = ddu_h1_n_coll[ddu_inst_i - 1];
      }
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s%i %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              ddu_common,
              ddu_inst_n,
              ddu_header2);
      ddu_h2_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
      ddu_h2_found = 1;
    }

    // DDU Header 3 (either between DDU Header 2 DMB Header or DDU Header 2 DDU Trailer1)
    else if ((ddu_h2_check[0] && dmb_h1_check[2]) || (ddu_h2_check[0] && ddu_tr1_check[2])) {
      ddu_inst_i = ddu_h1_n_coll.size();
      if (ddu_inst_i > 0) {
        ddu_inst_n = ddu_h1_n_coll[ddu_inst_i - 1];
      }
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s%i %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              ddu_common,
              ddu_inst_n,
              ddu_header3);
      ddu_h3_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
      ddu_h2_found = 0;
    }

    // DMB Header 1,2

    else if (dmb_h1_check[1]) {
      dmb_inst_crate = 0;
      dmb_inst_slot = 0;
      dmb_inst_l1a = ((buf[i] & 0x0FFF) + ((buf[i + 1] & 0xFFF) << 12));
      dmb_l1a_coll.push_back(dmb_inst_l1a);
      if (dmb_h2_check[2]) {
        dmb_inst_crate = ((buf[i + 4 + 1] >> 4) & 0xFF);
        dmb_inst_slot = (buf[i + 4 + 1] & 0xF);
        dmb_crate_coll.push_back(dmb_inst_crate);
        dmb_slot_coll.push_back(dmb_inst_slot);
      }
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s%s%s %i %s %i %s %i",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              dmb_common,
              dmb_header1,
              sign1,
              dmb_common_crate,
              dmb_inst_crate,
              dmb_common_slot,
              dmb_inst_slot,
              dmb_common_l1a,
              dmb_inst_l1a);
      dmb_h1_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
      ddu_h2_found = 1;
    }

    else if (dmb_h2_check[1]) {
      dmb_inst_crate = ((buf[i + 1] >> 4) & 0xFF);
      dmb_inst_slot = (buf[i + 1] & 0xF);
      dmb_h2_coll.push_back(word_numbering);
      if (dmb_h1_check[0])
        dmb_inst_l1a = ((buf[i - 4] & 0x0FFF) + ((buf[i - 4 + 1] & 0xFFF) << 12));
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s%s%s %i %s %i %s %i",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              dmb_common,
              dmb_header2,
              sign1,
              dmb_common_crate,
              dmb_inst_crate,
              dmb_common_slot,
              dmb_inst_slot,
              dmb_common_l1a,
              dmb_inst_l1a);
      std::cout << tempbuf1 << std::endl;
      w = 0;
      ddu_h2_found = 1;
    }

    //DDU Trailer 1

    else if (ddu_tr1_check[1]) {
      ddu_inst_i = ddu_h1_n_coll.size();
      if (ddu_inst_i > 0) {
        ddu_inst_n = ddu_h1_n_coll[ddu_inst_i - 1];
      }
      //ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s%i %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              ddu_common,
              ddu_inst_n,
              ddu_trail1);
      ddu_t1_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }

    ///ALCT Header 1,2
    else if (alct_h1_check[1]) {
      alct_start = word_numbering;
      alct_inst_l1a = (buf[i + 2] & 0x0FFF);
      alct_l1a_coll.push_back(alct_inst_l1a);
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s%s %s %i",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              alct_common,
              alct_header1,
              sign1,
              dmb_common_l1a,
              alct_inst_l1a);
      alct_h1_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }

    else if ((alct_h1_check[0]) && (alct_h2_check[2])) {
      alct_inst_bxn = (buf[i] & 0x0FFF);
      alct_bxn_coll.push_back(alct_inst_bxn);
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s%s%s %i",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              alct_common,
              alct_header2,
              sign1,
              alct_common_bxn,
              alct_inst_bxn);
      alct_h2_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }

    //ALCT Trailer 1
    else if (alct_tr1_check[1]) {
      alct_stop = word_numbering;
      if ((alct_start != 0) && (alct_stop != 0) && (alct_stop > alct_start)) {
        alct_inst_wcnt2 = 4 * (alct_stop - alct_start + 1);
        alct_wcnt2_coll.push_back(alct_inst_wcnt2);
        alct_wcnt2_id_coll.push_back(alct_start);
      }
      alct_inst_wcnt1 = (buf[i + 3] & 0x7FF);
      alct_wcnt1_coll.push_back(alct_inst_wcnt1);
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s%s%s %i %s %i",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              alct_common,
              alct_tr1,
              sign1,
              alct_common_wcnt1,
              alct_inst_wcnt1,
              alct_common_wcnt2,
              alct_inst_wcnt2);
      alct_t1_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
      alct_inst_wcnt2 = 0;
    }

    //DDU Trailer 3

    //      else if ((ddu_tr1_check[-1])&&(tempbuf_short[0]==ddu_trailer3_bit[0])) { // !!! TO FIX: negative index
    else if ((ddu_h2_h1) && (tempbuf_short[0] == ddu_trailer3_bit[0])) {
      //&&(tempbuf_short[0]==ddu_trailer3_bit[0])){
      ddu_inst_i = ddu_h1_n_coll.size();
      if (ddu_inst_i > 0) {
        ddu_inst_n = ddu_h1_n_coll[ddu_inst_i - 1];
      }
      //ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s%i %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              ddu_common,
              ddu_inst_n,
              ddu_trail3);
      ddu_t3_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }
    //DDU Trailer 2
    else if ((ddu_tr1_check[0]) && (tempbuf_short[0] != ddu_trailer3_bit[0])) {
      //&&(tempbuf_short[0]==ddu_trailer3_bit[0])){
      ddu_inst_i = ddu_h1_n_coll.size();
      if (ddu_inst_i > 0) {
        ddu_inst_n = ddu_h1_n_coll[ddu_inst_i - 1];
      }
      //ddu_inst_n=ddu_h1_n_coll[ddu_inst_i-1];
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s%i %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              ddu_common,
              ddu_inst_n,
              ddu_trail2);
      ddu_t2_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }

    //DMB Trailer 1,2
    else if (dmb_tr1_check[1]) {
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              dmb_common,
              dmb_tr1);
      dmb_t1_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
      cfeb_sample = 0;
    }

    else if (dmb_tr2_check[1]) {
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              dmb_common,
              dmb_tr2);
      dmb_t2_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }
    // TMB
    else if (tmb_h1_check[1]) {
      tmb_start = word_numbering;
      tmb_inst_l1a = (buf[i + 2] & 0x000F);
      tmb_l1a_coll.push_back(tmb_inst_l1a);
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s%s%s %i",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              tmb_common,
              tmb_header1,
              sign1,
              dmb_common_l1a,
              tmb_inst_l1a);
      tmb_h1_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    } else if (tmb_tr1_check[1]) {
      tmb_stop = word_numbering;
      if ((tmb_start != 0) && (tmb_stop != 0) && (tmb_stop > tmb_start)) {
        tmb_inst_wcnt2 = 4 * (tmb_stop - tmb_start + 1);
        tmb_wcnt2_coll.push_back(tmb_inst_wcnt2);
      }
      tmb_inst_wcnt1 = (buf[i + 3] & 0x7FF);
      tmb_wcnt1_coll.push_back(tmb_inst_wcnt1);
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s%s%s %i %s %i",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              tmb_common,
              tmb_tr1,
              sign1,
              alct_common_wcnt1,
              tmb_inst_wcnt1,
              alct_common_wcnt2,
              tmb_inst_wcnt2);
      tmb_t1_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
      tmb_inst_wcnt2 = 0;
    }
    // CFEB
    else if (cfeb_tr1_check[1]) {
      ++cfeb_sample;
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s%s %s %i",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              cfeb_common,
              cfeb_tr1,
              sign1,
              cfeb_common_sample,
              cfeb_sample);
      cfeb_t1_coll.push_back(word_numbering);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    } else if (cfeb_b_check[1]) {
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              cfeb_common,
              cfeb_b);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }

    //ERRORS ddu_tr1_bad_check

    else if (ddu_tr1_bad_check[1]) {
      ddu_inst_i = ddu_h1_n_coll.size();
      ddu_inst_n = ddu_h1_n_coll[ddu_inst_i - 1];
      sprintf(tempbuf1,
              "%6i    %04x %04x %04x %04x%s%s%i %s %s",
              word_numbering,
              buf[i + 3],
              buf[i + 2],
              buf[i + 1],
              buf[i],
              sign1,
              ddu_common,
              ddu_inst_n,
              ddu_trail1,
              ddu_tr1_err_common);
      std::cout << tempbuf1 << std::endl;
      w = 0;
    }

    else if (extraction && (!ddu_h1_check) && (!dcc_check)) {
      if (w < 3) {
        sprintf(tempbuf, "%6i    %04x %04x %04x %04x", word_numbering, buf[i + 3], buf[i + 2], buf[i + 1], buf[i]);
        std::cout << tempbuf << std::endl;
        w++;
      }
      if (w == 3) {
        std::cout << "..................................................." << std::endl;
        w++;
      }
    }

    else if ((!ddu_h1_check) && (!dcc_check)) {
      sprintf(tempbuf, "%6i    %04x %04x %04x %04x", word_numbering, buf[i + 3], buf[i + 2], buf[i + 1], buf[i]);
      std::cout << tempbuf << std::endl;
    }

    i += 3;
    ddu_h1_check = false;
    dcc_check = false;
  }
  //char sign[30]; //WARNING 5_0_X
  std::cout << "********************************************************************************" << std::endl
            << std::endl;
  if (fedshort)
    std::cout << "For complete output turn off VisualFEDShort in muonCSCDigis configuration file." << std::endl;
  std::cout << "********************************************************************************" << std::endl
            << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "            Summary                " << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << ddu_h1_coll.size() << "  " << ddu_common << "  " << ddu_header1 << "  "
            << "found" << std::endl;
  /*
  std::cout << ddu_h1_coll.size() << " " << ddu_h1_n_coll.size() << " " << ddu_l1a_coll.size() <<
  " " << ddu_bxn_coll.size() << std::endl;
  */
  for (unsigned int k = 0; k < ddu_h1_coll.size(); ++k) {
    /*
      sprintf(sign,"%s%6i%5s %s%i %s %i %s %i","Line: ",
      ddu_h1_coll[k],sign1,ddu_common,ddu_h1_n_coll[k],dmb_common_l1a,ddu_l1a_coll[k],
      alct_common_bxn,ddu_bxn_coll[k]);
      */
    std::cout << "Line: "
              << "    " << ddu_h1_coll[k] << " " << sign1 << " " << ddu_common << " " << ddu_h1_n_coll[k] << " "
              << dmb_common_l1a << " " << ddu_l1a_coll[k] << " " << alct_common_bxn << " " << ddu_bxn_coll[k]
              << std::endl;
  }

  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << ddu_h2_coll.size() << "  " << ddu_common << "  " << ddu_header2 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < ddu_h2_coll.size(); ++k)
    std::cout << "Line:  " << ddu_h2_coll[k] << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << ddu_h3_coll.size() << "  " << ddu_common << "  " << ddu_header3 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < ddu_h3_coll.size(); ++k)
    std::cout << "Line:  " << ddu_h3_coll[k] << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << ddu_t1_coll.size() << "  " << ddu_common << "  " << ddu_trail1 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < ddu_t1_coll.size(); ++k)
    std::cout << "Line:  " << ddu_t1_coll[k] << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << ddu_t2_coll.size() << "  " << ddu_common << "  " << ddu_trail2 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < ddu_t2_coll.size(); ++k)
    std::cout << "Line:  " << ddu_t2_coll[k] << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << ddu_t3_coll.size() << "  " << ddu_common << "  " << ddu_trail3 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < ddu_t3_coll.size(); ++k)
    std::cout << "Line:  " << ddu_t3_coll[k] << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << dmb_h1_coll.size() << "  " << dmb_common << "  " << dmb_header1 << "  "
            << "found" << std::endl;

  for (unsigned int k = 0; k < dmb_h1_coll.size(); ++k) {
    /*
      sprintf(sign,"%s%6i%5s %s %s %i %s %i %s %i","Line: ",
      dmb_h1_coll[k],sign1,dmb_common,dmb_common_crate,dmb_crate_coll[k],dmb_common_slot,
      dmb_slot_coll[k],dmb_common_l1a,dmb_l1a_coll[k]);
      */
    std::cout << "Line: "
              << "    " << dmb_h1_coll[k] << " " << sign1 << dmb_common << " " << dmb_common_crate << " "
              << dmb_crate_coll[k] << " " << dmb_common_slot << " " << dmb_slot_coll[k] << " " << dmb_common_l1a << " "
              << dmb_l1a_coll[k] << std::endl;
  }
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << dmb_h2_coll.size() << "  " << dmb_common << "  " << dmb_header2 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < dmb_h2_coll.size(); ++k)
    std::cout << "Line:  " << dmb_h2_coll[k] << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << dmb_t1_coll.size() << "  " << dmb_common << "  " << dmb_tr1 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < dmb_t1_coll.size(); ++k)
    std::cout << "Line:  " << dmb_t1_coll[k] << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << dmb_t2_coll.size() << "  " << dmb_common << "  " << dmb_tr2 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < dmb_t2_coll.size(); ++k)
    std::cout << "Line:  " << dmb_t2_coll[k] << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << alct_h1_coll.size() << "  " << alct_common << "  " << alct_header1 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < alct_h1_coll.size(); ++k) {
    /*
      sprintf(sign,"%s%6i%5s %s %s %i","Line: ",
      alct_h1_coll[k],sign1,alct_common,
      dmb_common_l1a,alct_l1a_coll[k]);
      std::cout << sign << std::endl;
      */
    std::cout << "Line: "
              << "    " << alct_h1_coll[k] << " " << sign1 << " " << alct_common << " " << dmb_common_l1a << " "
              << alct_l1a_coll[k] << std::endl;
  }

  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << alct_h2_coll.size() << "  " << alct_common << "  " << alct_header2 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < alct_h2_coll.size(); ++k) {
    /*
      sprintf(sign,"%s%6i%5s %s %s %i","Line: ",
      alct_h1_coll[k],sign1,alct_common,
      alct_common_bxn,alct_bxn_coll[k]);
      std::cout << sign << std::endl;
      */
    std::cout << "Line: "
              << "    " << alct_h1_coll[k] << " " << sign1 << " " << alct_common << " " << alct_common_bxn << " "
              << alct_bxn_coll[k] << std::endl;
  }

  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << alct_t1_coll.size() << "  " << alct_common << "  " << alct_tr1 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < alct_t1_coll.size(); ++k) {
    /*
         sprintf(sign,"%s%6i%5s %s %s %i %s %i","Line: ",
         alct_t1_coll[k],sign1,alct_common,
         alct_common_wcnt1,alct_wcnt1_coll[k],alct_common_wcnt2,alct_wcnt2_coll[k]);
         std::cout << sign << std::endl;
       */
    std::cout << "Line: "
              << "    " << alct_t1_coll[k] << " " << sign1 << " " << alct_common << " " << alct_common_wcnt1 << " "
              << alct_wcnt1_coll[k] << " " << alct_common_wcnt2 << " ";
    if (!alct_wcnt2_coll.empty()) {
      std::cout << alct_wcnt2_coll[k] << std::endl;
    } else {
      std::cout << "Undefined (ALCT Header is not found) " << std::endl;
    }
  }

  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << tmb_h1_coll.size() << "  " << tmb_common << "  " << tmb_header1 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < tmb_h1_coll.size(); ++k) {
    /*
      sprintf(sign,"%s%6i%5s %s %s %i","Line: ",
      tmb_h1_coll[k],sign1,tmb_common,
      dmb_common_l1a,tmb_l1a_coll[k]);
      std::cout << sign << std::endl;
      */
    std::cout << "Line: "
              << "    " << tmb_h1_coll[k] << " " << sign1 << " " << tmb_common << " " << dmb_common_l1a << " "
              << tmb_l1a_coll[k] << std::endl;
  }

  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << tmb_t1_coll.size() << "  " << tmb_common << "  " << tmb_tr1 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < tmb_t1_coll.size(); ++k) {
    /*
      sprintf(sign,"%s%6i%5s %s %s %i %s %i","Line: ",
      tmb_t1_coll[k],sign1,tmb_common,
      alct_common_wcnt1,tmb_wcnt1_coll[k],alct_common_wcnt2,tmb_wcnt2_coll[k]);
      std::cout << sign << std::endl;
      */
    std::cout << "Line: "
              << "    " << tmb_t1_coll[k] << " " << sign1 << " " << tmb_common << " " << alct_common_wcnt1 << " "
              << tmb_wcnt1_coll[k] << " " << alct_common_wcnt2 << " " << tmb_wcnt2_coll[k] << std::endl;
  }

  std::cout << std::endl << std::endl;
  std::cout << "||||||||||||||||||||" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << cfeb_t1_coll.size() << "  " << cfeb_common << "  " << cfeb_tr1 << "  "
            << "found" << std::endl;
  for (unsigned int k = 0; k < cfeb_t1_coll.size(); ++k)
    std::cout << "Line:  " << cfeb_t1_coll[k] << std::endl;
  std::cout << "********************************************************************************" << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCDCCUnpacker);
