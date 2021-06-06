//-------------------------------------------------
//
//   Class: DTTM7FEDReader
//
//   L1 DT TwinMux Raw-to-Digi
//
//
//
//   Author :
//   C. F. Bedoya  - CIEMAT
//   G. Codispoti -- INFN Bologna
//   J. Pazzini   -- INFN Padova
//   C. Heidemann -- RWTH Aachen
//
//
//   Changes & Notes:
//   This version is able to unpack the normal and the zero suppressed data
//   For storing the additional info from the ud flag in the raw data and not changing the output data format of the unpacker, it will be stored as the sign to the station number for Phi In and Out segments. ud == 0 is stored as a positive station number, while ud==1 is stored as a negative station number. Use abs(station) to obtain the real station.
//   This change is transparent to old data, too.
//   The switch between old and new unpacker code is done by the firmware revision. If it >= 93 the new unpacker code is used, otherwise the old and unmodified code is used.
//
//--------------------------------------------------

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/Utilities/interface/DTCRC.h"

#include <iostream>
#include <fstream>

class L1TTwinMuxRawToDigi : public edm::global::EDProducer<> {
public:
  /// Constructor
  L1TTwinMuxRawToDigi(const edm::ParameterSet& pset);

  /// Destructor
  ~L1TTwinMuxRawToDigi() override;

  /// Produce digis out of raw data
  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

  /// Generate and fill FED raw data for a full event
  bool fillRawData(edm::Event& e,
                   L1MuDTChambPhContainer::Phi_Container& phi_data,
                   L1MuDTChambThContainer::The_Container& the_data,
                   L1MuDTChambPhContainer::Phi_Container& phi_out_data) const;

  void processFed(int twinmuxfed,
                  int wheel,
                  std::array<short, 12> const& twinMuxAmcSec,
                  edm::Handle<FEDRawDataCollection> data,
                  L1MuDTChambPhContainer::Phi_Container& phi_data,
                  L1MuDTChambThContainer::The_Container& the_data,
                  L1MuDTChambPhContainer::Phi_Container& phi_out_data) const;

private:
  bool debug_;
  size_t nfeds_;
  edm::InputTag DTTM7InputTag_;
  std::vector<int> feds_;
  std::vector<int> wheels_;
  std::vector<long long int> amcsecmap_;
  std::vector<std::array<short, 12> > amcsec_;

  // utilities
  inline unsigned char* readline(unsigned char* lineFED, int& lines, long& dataWord) const {
    dataWord = *((long*)lineFED);
    lineFED += 8;
    ++lines;
    return lineFED;
  }

  edm::InputTag getDTTM7InputTag() const { return DTTM7InputTag_; }

  edm::EDGetTokenT<FEDRawDataCollection> Raw_token;

  int normBx(int bx_, int bxCnt_) const;
  int radAngConversion(int radAng_) const;
  int benAngConversion(int benAng_) const;
};

L1TTwinMuxRawToDigi::L1TTwinMuxRawToDigi(const edm::ParameterSet& pset)
    :

      debug_(pset.getUntrackedParameter<bool>("debug", false)),
      nfeds_(0),
      DTTM7InputTag_(pset.getParameter<edm::InputTag>("DTTM7_FED_Source")),
      feds_(pset.getUntrackedParameter<std::vector<int> >("feds", std::vector<int>())),
      wheels_(pset.getUntrackedParameter<std::vector<int> >("wheels", std::vector<int>())),
      amcsecmap_(pset.getUntrackedParameter<std::vector<long long int> >("amcsecmap", std::vector<long long int>()))

{
  produces<L1MuDTChambPhContainer>("PhIn").setBranchAlias("PhIn");
  produces<L1MuDTChambThContainer>("ThIn").setBranchAlias("ThIn");
  produces<L1MuDTChambPhContainer>("PhOut").setBranchAlias("PhOut");

  Raw_token = consumes<FEDRawDataCollection>(DTTM7InputTag_);

  nfeds_ = feds_.size();

  if (nfeds_ != wheels_.size())
    throw cms::Exception("TwinMux_unpacker") << "Configuration file error. Size of \'wheels\' and \'feds\' differs.\n";

  if (amcsecmap_.size() != wheels_.size())
    throw cms::Exception("TwinMux_unpacker")
        << "Configuration file error. Size of \'wheels\' and \'amcsecmap\' differs.\n";

  for (size_t wh_i = 0; wh_i < amcsecmap_.size(); ++wh_i) {
    std::array<short, 12> whmap;
    for (size_t amc_i = 1; amc_i < 13; ++amc_i) {
      short shift = (12 - amc_i) * 4;
      whmap[amc_i - 1] = (amcsecmap_[wh_i] >> shift) & 0xF;
    }
    amcsec_.push_back(whmap);
  }
}

L1TTwinMuxRawToDigi::~L1TTwinMuxRawToDigi() {}

void L1TTwinMuxRawToDigi::produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const {
  std::unique_ptr<L1MuDTChambPhContainer> TM7phi_product(new L1MuDTChambPhContainer);
  std::unique_ptr<L1MuDTChambThContainer> TM7the_product(new L1MuDTChambThContainer);
  std::unique_ptr<L1MuDTChambPhContainer> TM7phi_out_product(new L1MuDTChambPhContainer);

  L1MuDTChambPhContainer::Phi_Container phi_data;
  L1MuDTChambThContainer::The_Container the_data;
  L1MuDTChambPhContainer::Phi_Container phi_out_data;

  if (!fillRawData(e, phi_data, the_data, phi_out_data))
    return;

  TM7phi_product->setContainer(phi_data);
  TM7the_product->setContainer(the_data);
  TM7phi_out_product->setContainer(phi_out_data);

  e.put(std::move(TM7phi_product), "PhIn");
  e.put(std::move(TM7the_product), "ThIn");
  e.put(std::move(TM7phi_out_product), "PhOut");
}

bool L1TTwinMuxRawToDigi::fillRawData(edm::Event& e,
                                      L1MuDTChambPhContainer::Phi_Container& phi_data,
                                      L1MuDTChambThContainer::The_Container& the_data,
                                      L1MuDTChambPhContainer::Phi_Container& phi_out_data) const {
  edm::Handle<FEDRawDataCollection> data;
  e.getByToken(Raw_token, data);

  for (size_t w_i = 0; w_i < nfeds_; ++w_i) {
    processFed(feds_[w_i], wheels_[w_i], amcsec_[w_i], data, phi_data, the_data, phi_out_data);
  }

  return true;
}

int L1TTwinMuxRawToDigi::normBx(int bx_, int bxCnt_) const {
  int bxNorm_ = bx_ - bxCnt_;
  if (abs(bxNorm_) < 3000)
    return bxNorm_;

  if (bxNorm_ > 0)
    return bxNorm_ - 3564;
  if (bxNorm_ < 0)
    return bxNorm_ + 3564;

  return -99;
}

int L1TTwinMuxRawToDigi::radAngConversion(int radAng_) const {
  if (radAng_ > 2047)
    return radAng_ - 4096;

  return radAng_;
}

int L1TTwinMuxRawToDigi::benAngConversion(int benAng_) const {
  if (benAng_ > 511)
    return benAng_ - 1024;

  return benAng_;
}

void L1TTwinMuxRawToDigi::processFed(int twinMuxFed,
                                     int twinMuxWheel,
                                     std::array<short, 12> const& twinMuxAmcSec,
                                     edm::Handle<FEDRawDataCollection> data,
                                     L1MuDTChambPhContainer::Phi_Container& phiSegments,
                                     L1MuDTChambThContainer::The_Container& theSegments,
                                     L1MuDTChambPhContainer::Phi_Container& phioutSegments) const {
  const unsigned int fw_rev_with_zerosupression =
      93;  // put the correct Firmware Revision of the first version with zerosuppression
  int previous_selector = -100;
  /// Container
  std::vector<long> DTTM7WordContainer;

  /// Debug
  std::ofstream logfile;
  if (debug_) {
    std::ostringstream fname;
    fname << "eventDump_" << twinMuxFed << ".txt";
    logfile.open(fname.str());
  }

  /// Header
  FEDRawData TM7data = data->FEDData(twinMuxFed);
  if (TM7data.size() == 0)
    return;

  /// Variables
  unsigned char* lineFED = TM7data.data();
  int nline = 0;  // counting already include header
  long dataWord = 0;
  int newCRC = 0xFFFF;

  ///--> Header - line 1 [must start with 0x5]
  lineFED = readline(lineFED, nline, dataWord);
  dt_crc::calcCRC(dataWord, newCRC);

  int TM7fedId = (dataWord >> 8) & 0xFFF;  // positions 8 -> 19
  /*** NOT UNPACKED  
  int bunchCnt = ( dataWord >> 20 ) & 0xFFF;  // positions 20 -> 31
  int eventCnt = ( dataWord >> 32 ) & 0xFFFFFF;  // positions 32 -> 55
  ***/
  int BOEevTy = (dataWord >> 60) & 0xF;  // positions 60 -> 63

  int linecounter = 0;
  if (debug_)
    logfile << '[' << ++linecounter << "]\t" << std::hex << dataWord << std::dec << "\t|\t"
            << "BOEevTy " << BOEevTy << '\t' << "TM7fedId " << TM7fedId << '\n';

  if ((BOEevTy != 0x5) || (TM7fedId != twinMuxFed)) {
    edm::LogWarning("TwinMux_unpacker") << "Not a TM7 of FED " << twinMuxFed << " header " << std::hex << dataWord;
    return;
  }

  ///--> Header - line 2
  lineFED = readline(lineFED, nline, dataWord);
  dt_crc::calcCRC(dataWord, newCRC);

  std::map<int, int> AMCsizes;
  /*** NOT UNPACKED  
  int orbit = ( dataWord >> 4 ) & 0xFFFFFFFF;  // positions 4 -> 35
  ***/
  int nAMC = (dataWord >> 52) & 0xF;  // positions 52 -> 55

  if (debug_)
    logfile << '[' << ++linecounter << "]\t" << std::hex << dataWord << std::dec << "\t|\t"
            << "nAMC " << nAMC << '\n';

  ///--> AMC - line 3 to 3+nAMC
  for (int j = 0; j < nAMC; ++j) {
    lineFED = readline(lineFED, nline, dataWord);
    dt_crc::calcCRC(dataWord, newCRC);

    int AMCno = (dataWord >> 16) & 0xF;  // positions 16 -> 19
    /*** NOT UNPACKED  
    int TM7boardID = dataWord & 0xFFFF;  // positions 0 -> 15
    int bulkno = (dataWord >> 20 ) & 0xFF;  // positions 20 -> 27
    ***/
    if ((AMCno < 1) || (AMCno > 12)) {
      edm::LogWarning("TwinMux_unpacker") << "AMCnumber " << std::dec << AMCno << " out of range (1-12)";
      return;
    }

    AMCsizes[AMCno] = (dataWord >> 32) & 0xFFFFFF;  // positions 32 -> 55

    if (debug_)
      logfile << '[' << ++linecounter << "]\t" << std::hex << dataWord << std::dec << "\t|\t"
              << "AMCsizes[" << AMCno << "] " << AMCsizes[AMCno] << std::dec << '\n';
  }

  ///--> Store payloads
  std::map<int, int>::iterator AMCiterator = AMCsizes.begin();
  std::map<int, int>::iterator AMCitend = AMCsizes.end();
  for (; AMCiterator != AMCitend; ++AMCiterator) {
    for (int k = 0; k < AMCiterator->second; ++k) {
      lineFED = readline(lineFED, nline, dataWord);
      dt_crc::calcCRC(dataWord, newCRC);

      DTTM7WordContainer.push_back(dataWord);
    }
  }

  ///--> Trailer - line 1
  lineFED = readline(lineFED, nline, dataWord);
  dt_crc::calcCRC(dataWord, newCRC);

  ///--> Trailer - line 2 [must start with 0xA]

  lineFED = readline(lineFED, nline, dataWord);
  dt_crc::calcCRC(dataWord & 0xFFFFFFFF0000FFFF, newCRC);

  ///--> AMC trailer - line 2
  int chkEOE = (dataWord >> 60) & 0xF;        // positions 60 -> 63
  int CRC = (dataWord >> 16) & 0xFFFF;        // positions 17 ->32
  int evtLgth = (dataWord >> 32) & 0xFFFFFF;  // positions 33 ->56

  if (chkEOE != 0xA) {
    edm::LogWarning("TwinMux_unpacker") << "AMC block closing line " << std::hex << dataWord << std::dec
                                        << " does not start with 0xA";
    return;
  }

  if (debug_)
    logfile << "\tevtLgth " << std::hex << evtLgth << "\tCRC " << CRC << std::dec << '\n';

  if (nline != evtLgth) {
    edm::LogWarning("TwinMux_unpacker") << "Number of words read " << std::dec << nline << " and event length "
                                        << std::dec << evtLgth << " differ ";
    return;
  }

  if (newCRC != CRC) {
    edm::LogWarning("TwinMux_unpacker") << "Calculated CRC " << std::hex << newCRC << " differs from CRC in trailer "
                                        << std::hex << CRC;
    return;
  }

  // --> Analyze event
  std::vector<long>::iterator DTTM7iterator = DTTM7WordContainer.begin();
  std::vector<long>::iterator DTTM7itend = DTTM7WordContainer.end();

  int lcounter = 0;
  for (; DTTM7iterator != DTTM7itend; ++DTTM7iterator) {
    dataWord = (*DTTM7iterator);
    int dataLenght = (dataWord & 0xFFFFF);     // positions 0 -> 19
    int bxCounter = (dataWord >> 20) & 0xFFF;  // positions 20 -> 31
    int event = (dataWord >> 32) & 0xFFFFFF;   // positions 32 -> 55
    int AMC_ID = (dataWord >> 56) & 0xF;       // positions 56 -> 59
    int control = (dataWord >> 60) & 0xF;      // positions 59 -> 63
    int wheel = twinMuxWheel;

    if ((AMC_ID < 1) or (AMC_ID > 12)) {
      edm::LogWarning("TwinMux_unpacker") << "%%%%%% AMC_ID OUT OF RANGE \n"
                                          << " TM7fedId " << TM7fedId << " AMC_ID " << AMC_ID;
      break;
    }

    int sector = twinMuxAmcSec[AMC_ID - 1];

    if ((sector < 1) or (sector > 12)) {
      if (sector != 15)
        edm::LogWarning("TwinMux_unpacker")
            << "%%%%%% VALID AMC_ID POINTS TO SECTOR OUT OF RANGE \n"
            << " TM7fedId " << TM7fedId << " AMC_ID " << AMC_ID << " wheel " << wheel << " sector " << sector;
      break;
    }

    if (debug_)
      logfile << '[' << ++lcounter << "]\t" << std::hex << dataWord << std::dec << "\t|\t"
              << "AMC_ID " << AMC_ID << '\t' << "control " << control << '\t' << "event " << event << '\t'
              << "bxCounter " << bxCounter << '\t' << "dataLenght " << dataLenght << '\n';

    ++DTTM7iterator;  // User word empty  /// ==>> increment 2
    if (DTTM7iterator == DTTM7itend) {
      LogDebug("TwinMux_unpacker") << "TRAILING WORD AS A PAYLOAD END in FED " << std::hex << TM7fedId << std::hex
                                   << dataWord << std::dec << " [it pos " << int(DTTM7iterator - DTTM7itend) << " ]";
      break;
    }

    dataWord = (*DTTM7iterator);
    unsigned int firmware_rev = (dataWord >> 7) & 0x1FF;
    int boardID = (dataWord & 0xFFFF);      // positions  0 -> 15
    int orbit = (dataWord >> 16) & 0xFFFF;  // positions 15 -> 32

    if (DTTM7iterator == DTTM7itend) {
      edm::LogWarning("TwinMux_unpacker")
          << "%%%%%% AMC_ID " << AMC_ID << " control " << control << " event " << event << " bxCounter " << bxCounter
          << " size " << dataLenght << " orbit " << orbit << " board " << boardID << " AMCsizes " << AMCsizes[AMC_ID]
          << " it pos " << int(DTTM7iterator - DTTM7itend);
      break;
    }

    if (debug_)
      logfile << '[' << ++lcounter << "]\t" << std::hex << dataWord << std::dec << "\t|\t"
              << " orbit " << orbit << " board " << boardID << '\n';

    int AMCsize = AMCsizes[AMC_ID] - 1;  /// do not consider the trailer
    int bxID = 99;
    int bc0 = -99;
    int bxNr = -99;

    /// 2 words already read, last removed because trailer with CRC
    for (int tm7eventsize = 2; tm7eventsize < AMCsize; ++tm7eventsize) {
      ++DTTM7iterator;  /// ==>> increment 3
      if (DTTM7iterator == DTTM7itend) {
        edm::LogWarning("TwinMux_unpacker") << "UNEXPECTED END OF PAYLOAD INSIDE CHAMBER DESCRIPTION"
                                            << " [it pos " << int(DTTM7iterator - DTTM7itend) << " ]";
        break;
      }

      long dataWordSub = (*DTTM7iterator);
      int selector = (dataWordSub >> 60) & 0xF;  // positions 60 -> 63

      if (firmware_rev >= fw_rev_with_zerosupression) {  // zerosuppressed data unpacking
        if (selector == 0xC) {                           /// BX info + theta info word

          bc0 = (dataWordSub >> 59) & 0x1;  // position 59
          /*** NOT UNPACKED  
			int L1A = ( dataWordSub >> 58) & 0x1;    // position 58
			***/
          int bxOffset = (dataWordSub >> 48) & 0x3F;  // positions 48 -> 53
          /// Two's complement conversion
          /// Important: here value 0x20 has been treated as -32 when in fact it is an error code
          /// denoting incorrect readout window configuration, but the case should really not happen
          bxNr = bxOffset < 32 ? bxOffset : bxOffset - 64;

          //eta info

          int posBTI[7], qualBTI[7];

          int mb3_eta = (dataWordSub & 0xFF);           // positions  0 -> 7
          int mb3_eta_HQ = (dataWordSub >> 8) & 0xFF;   // positions  8 -> 15
          int mb2_eta = (dataWordSub >> 16) & 0xFF;     // positions 16 -> 23
          int mb2_eta_HQ = (dataWordSub >> 24) & 0xFF;  // positions 24 -> 31
          int mb1_eta = (dataWordSub >> 32) & 0xFF;     // positions 32 -> 39
          int mb1_eta_HQ = (dataWordSub >> 40) & 0xFF;  // positions 40 -> 47

          //MB1
          for (int i = 0; i <= 5; i++) {
            posBTI[i] = (mb1_eta >> i) & 0x01;
            qualBTI[i] = (mb1_eta_HQ >> i) & 0x01;
          }
          posBTI[6] = (((mb1_eta >> 6) & 0x01) || ((mb1_eta >> 7) & 0x01));
          qualBTI[6] = (((mb1_eta_HQ >> 6) & 0x01) || ((mb1_eta_HQ >> 7) & 0x01));

          theSegments.push_back(L1MuDTChambThDigi(bxNr, wheel, sector - 1, 1, posBTI, qualBTI));

          // posBTI and qualBTI are reused!

          //MB2
          for (int i = 0; i <= 5; i++) {
            posBTI[i] = (mb2_eta >> i) & 0x01;
            qualBTI[i] = (mb2_eta_HQ >> i) & 0x01;
          }
          posBTI[6] = ((mb2_eta >> 6) & 0x01) || ((mb2_eta >> 7) & 0x01);
          qualBTI[6] = ((mb2_eta_HQ >> 6) & 0x01) || ((mb2_eta_HQ >> 7) & 0x01);

          theSegments.push_back(L1MuDTChambThDigi(bxNr, wheel, sector - 1, 2, posBTI, qualBTI));

          //MB3
          for (int i = 0; i <= 5; i++) {
            posBTI[i] = (mb3_eta >> i) & 0x01;
            qualBTI[i] = (mb3_eta_HQ >> i) & 0x01;
          }
          posBTI[6] = ((mb3_eta >> 6) & 0x01) || ((mb3_eta >> 7) & 0x01);
          qualBTI[6] = ((mb3_eta_HQ >> 6) & 0x01) || ((mb3_eta_HQ >> 7) & 0x01);

          theSegments.push_back(L1MuDTChambThDigi(bxNr, wheel, sector - 1, 3, posBTI, qualBTI));

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t bx info word\t"
                    << "bxOffset " << bxOffset << '\t' << "bc0  " << bc0 << '\t' << "mb1_eta " << mb1_eta << '\t'
                    << "mb2_eta " << mb2_eta << '\t' << "mb3_eta " << mb3_eta << '\t' << "mb1_eta_HQ " << mb1_eta_HQ
                    << '\t' << "mb2_eta_HQ " << mb2_eta_HQ << '\t' << "mb3_eta_HQ " << mb3_eta_HQ << '\n';

        }  /// BX info + theta info word

        else if (selector >= 1 && selector <= 4) {  ///Phi info word. MBx determined by selector 0x01 - 0x04

          int out_phi = (dataWordSub >> 0) & 0xFFF;    // positions 0 -> 11
          int out_phib = (dataWordSub >> 12) & 0x3FF;  // positions 12 -> 21
          int out_qual = (dataWordSub >> 22) & 0x7;    // positions 22 -> 24
          int out_q3 = (dataWordSub >> 25) & 0x1;      // positions 25
          int out_q4 = (dataWordSub >> 26) & 0x1;      // positions 26
          int out_updown = (dataWordSub >> 27) & 0x1;  // positions 27
          int out_ts2tag = (dataWordSub >> 28) & 0x1;  // positions 28
          /*** NOT UNPACKED  
			int out_parity = ( dataWordSub >> 29) & 0x1;    // positions 29
			***/
          int in_phi = (dataWordSub >> 30) & 0xFFF;   // positions  30 -> 41
          int in_phib = (dataWordSub >> 42) & 0x3FF;  // positions 42 -> 51
          int in_qual = (dataWordSub >> 52) & 0x7;    // positions 52 -> 54
          int in_updown = (dataWordSub >> 57) & 0x1;  // positions 57
          int in_ts2tag = (dataWordSub >> 58) & 0x1;  // positions 58
          /*** NOT UNPACKED  
			int in_parity = ( dataWordSub >> 59) & 0x1;    // positions 59
			***/

          int in_phi_conv = radAngConversion(in_phi);
          int in_phib_conv = benAngConversion(in_phib);
          int out_phi_conv = radAngConversion(out_phi);
          int out_phib_conv = benAngConversion(out_phib);

          if (previous_selector != selector) {  // first track
            if (in_qual != 7) {
              phiSegments.push_back(L1MuDTChambPhDigi(bxNr,
                                                      wheel,
                                                      sector - 1,
                                                      selector,
                                                      in_phi_conv,
                                                      in_phib_conv,
                                                      in_qual,
                                                      in_ts2tag + in_updown * 2,
                                                      bxCounter));
            }
            if (out_qual != 7) {
              phioutSegments.push_back(L1MuDTChambPhDigi(bxNr,
                                                         wheel,
                                                         sector - 1,
                                                         selector,
                                                         out_phi_conv,
                                                         out_phib_conv,
                                                         out_qual,
                                                         out_ts2tag + out_updown * 2,
                                                         bxCounter,
                                                         out_q3 + 2 * out_q4));
            }
          } else {  // currently no seperation between first/second in data, keep it for possible later fix
            // second track
            if (in_qual != 7) {
              phiSegments.push_back(L1MuDTChambPhDigi(bxNr,
                                                      wheel,
                                                      sector - 1,
                                                      selector,
                                                      in_phi_conv,
                                                      in_phib_conv,
                                                      in_qual,
                                                      in_ts2tag + in_updown * 2,
                                                      bxCounter));
            }

            if (out_qual != 7) {
              phioutSegments.push_back(L1MuDTChambPhDigi(bxNr,
                                                         wheel,
                                                         sector - 1,
                                                         selector,
                                                         out_phi_conv,
                                                         out_phib_conv,
                                                         out_qual,
                                                         out_ts2tag + out_updown * 2,
                                                         bxCounter,
                                                         out_q3 + 2 * out_q4));
            }
          }
          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t|\t"
                    << "in_ts2tag " << in_ts2tag << '\t' << "in_updown " << in_updown << '\t' << "in_qual " << in_qual
                    << '\t' << "in_phib " << in_phib_conv << '\t' << "in_phi " << in_phi_conv << '\t' << "out_ts2tag "
                    << out_ts2tag << '\t' << "out_updown " << out_updown << '\t' << "out_qual " << out_qual << '\t'
                    << "out_q3_out " << out_q3 << '\t' << "out_q4_out " << out_q4 << '\t' << "out_phib "
                    << out_phib_conv << '\t' << "out_phi " << out_phi_conv << '\t' << "2nd track "
                    << ((previous_selector == selector) ? 1 : 0) << '\n';
        }  ///Phi info word.

        else if (selector == 0x9 || selector == 0xE) {  //RPC word

          LogDebug("TwinMux_unpacker") << "RPC WORD [" << std::dec << tm7eventsize << "] : " << std::hex << dataWordSub
                                       << std::dec << " it pos " << int(DTTM7iterator - DTTM7itend);

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t RPC WORD\n";

        }  //RPC word

        else if (selector == 0x6) {  //HO word

          LogDebug("TwinMux_unpacker") << "HO WORD [" << std::dec << tm7eventsize << "] : " << std::hex << dataWordSub
                                       << std::dec << " it pos " << int(DTTM7iterator - DTTM7itend);

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t HO WORD\n";

        }  //HO word

        else if (selector == 0xF) {  //ERROR word

          LogDebug("TwinMux_unpacker") << "ERROR WORD [" << std::dec << tm7eventsize << "] : " << std::hex
                                       << dataWordSub << std::dec << " it pos " << int(DTTM7iterator - DTTM7itend);

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t ERROR WORD\n";
        }  //ERROR word

        else {  //unkown word

          LogDebug("TwinMux_unpacker") << "UNKNOWN WORD received " << std::hex << dataWordSub << " in FED " << std::hex
                                       << TM7fedId;

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t UNKNOWN WORD\n";
        }
        previous_selector = selector;  // store selector to identify 2nd track
      } else {
        // normal data unpacking below
        if (selector == 0x4) {  //TSC word

          bxID = (dataWordSub >> 48) & 0xFFF;  // positions 48 -> 60
          bc0 = (dataWordSub >> 22) & 0x1;     // positions 22 -> 23
          bxNr = normBx(bxID, bxCounter);      /// bx normalized to the bxcounter

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t TSC WORD\t"
                    << "bxID " << bxID << '\t' << "bc0  " << bc0 << '\n';

        }  //TSC WORD

        else if (selector == 0x1) {  //MB1/2 word

          int mb2_phi = (dataWordSub & 0xFFF);         // positions  0 -> 11
          int mb2_phib = (dataWordSub >> 12) & 0x3FF;  // positions 12 -> 21
          int mb2_qual = (dataWordSub >> 22) & 0x7;    // positions 22 -> 24
          int mb2_ts2tag = (dataWordSub >> 28) & 0x1;  // positions 28
          /*** NOT UNPACKED  
			int mb2_parity = ( dataWordSub >> 29) & 0x1;    // positions 29
			***/

          int mb1_phi = (dataWordSub >> 30) & 0xFFF;   // positions 30 -> 41
          int mb1_phib = (dataWordSub >> 42) & 0x3FF;  // positions 42 -> 51
          int mb1_qual = (dataWordSub >> 52) & 0x7;    // positions 52 -> 54
          int mb1_ts2tag = (dataWordSub >> 58) & 0x1;  // positions 58
          /*** NOT UNPACKED  
			int mb1_parity = ( dataWordSub >> 59 ) &0x1;    // positions 59
			***/

          int mb1_phi_conv = radAngConversion(mb1_phi);
          int mb1_phib_conv = benAngConversion(mb1_phib);

          int mb2_phi_conv = radAngConversion(mb2_phi);
          int mb2_phib_conv = benAngConversion(mb2_phib);

          phiSegments.push_back(L1MuDTChambPhDigi(
              bxNr, wheel, sector - 1, 1, mb1_phi_conv, mb1_phib_conv, mb1_qual, mb1_ts2tag, bxCounter, -1));
          phiSegments.push_back(L1MuDTChambPhDigi(
              bxNr, wheel, sector - 1, 2, mb2_phi_conv, mb2_phib_conv, mb2_qual, mb2_ts2tag, bxCounter, -1));

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t|\t"
                    << "mb1_ts2tag " << mb1_ts2tag << '\t' << "mb1_qual " << mb1_qual << '\t' << "mb1_phib "
                    << mb1_phib_conv << '\t' << "mb1_phi " << mb1_phi_conv << '\t' << "mb2_ts2tag " << mb2_ts2tag
                    << '\t' << "mb2_qual " << mb2_qual << '\t' << "mb2_phib " << mb2_phib_conv << '\t' << "mb2_phi "
                    << mb2_phi_conv << '\n';
        }  //MB1/2 word

        else if (selector == 0x2) {
          int mb4_phi = (dataWordSub & 0xFFF);         // positions  0 -> 11
          int mb4_phib = (dataWordSub >> 12) & 0x3FF;  // positions 12 -> 21
          int mb4_qual = (dataWordSub >> 22) & 0x7;    // positions 22 -> 24
          int mb4_ts2tag = (dataWordSub >> 28) & 0x1;  // positions 28
          /*** NOT UNPACKED  
			int mb4_parity = ( dataWordSub >> 29) & 0x1;    // positions 29
			***/

          int mb3_phi = (dataWordSub >> 30) & 0xFFF;   // positions 30 -> 41
          int mb3_phib = (dataWordSub >> 42) & 0x3FF;  // positions 42 -> 51
          int mb3_qual = (dataWordSub >> 52) & 0x7;    // positions 52 -> 54
          int mb3_ts2tag = (dataWordSub >> 58) & 0x1;  // positions 58
          /*** NOT UNPACKED  
			int mb3_parity = ( dataWordSub >> 59 ) &0x1;    // positions 59
			***/

          int mb3_phi_conv = radAngConversion(mb3_phi);
          int mb3_phib_conv = benAngConversion(mb3_phib);

          int mb4_phi_conv = radAngConversion(mb4_phi);
          int mb4_phib_conv = benAngConversion(mb4_phib);

          phiSegments.push_back(L1MuDTChambPhDigi(
              bxNr, wheel, sector - 1, 3, mb3_phi_conv, mb3_phib_conv, mb3_qual, mb3_ts2tag, bxCounter, -1));
          phiSegments.push_back(L1MuDTChambPhDigi(
              bxNr, wheel, sector - 1, 4, mb4_phi_conv, mb4_phib_conv, mb4_qual, mb4_ts2tag, bxCounter, -1));

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t|\t"
                    << "mb3_ts2tag " << mb3_ts2tag << '\t' << "mb3_qual " << mb3_qual << '\t' << "mb3_phib "
                    << mb3_phib_conv << '\t' << "mb3_phi " << mb3_phi_conv << '\t' << "mb4_ts2tag " << mb4_ts2tag
                    << '\t' << "mb4_qual " << mb4_qual << '\t' << "mb4_phib " << mb4_phib_conv << '\t' << "mb4_phi "
                    << mb4_phi_conv << '\n';

        }  //MB3/4 word

        else if (selector == 0x3) {  //etha word

          int posBTI[7], qualBTI[7];

          int mb3_eta = (dataWordSub & 0xFF);        // positions  0 -> 7
          int mb2_eta = (dataWordSub >> 16) & 0xFF;  // positions 16 -> 23
          int mb1_eta = (dataWordSub >> 40) & 0xFF;  // positions 40 -> 47

          int mb3_eta_HQ = (dataWordSub >> 8) & 0xFF;   // positions  8 -> 15
          int mb2_eta_HQ = (dataWordSub >> 32) & 0xFF;  // positions 32 -> 39
          int mb1_eta_HQ = (dataWordSub >> 48) & 0xFF;  // positions 48 -> 55

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t|\t"
                    << "mb1_eta " << mb1_eta << '\t' << "mb2_eta " << mb2_eta << '\t' << "mb3_eta " << mb3_eta << '\t'
                    << "mb1_eta_HQ " << mb1_eta_HQ << '\t' << "mb2_eta_HQ " << mb2_eta_HQ << '\t' << "mb3_eta_HQ "
                    << mb3_eta_HQ << '\n';

          //MB1
          posBTI[0] = (mb1_eta & 0x01);
          posBTI[1] = ((mb1_eta & 0x02) >> 1);
          posBTI[2] = ((mb1_eta & 0x04) >> 2);
          posBTI[3] = ((mb1_eta & 0x08) >> 3);
          posBTI[4] = ((mb1_eta & 0x10) >> 4);
          posBTI[5] = ((mb1_eta & 0x20) >> 5);
          posBTI[6] = (((mb1_eta & 0x40) >> 6) || ((mb1_eta & 0x80) >> 7));

          qualBTI[0] = (mb1_eta_HQ & 0x01);
          qualBTI[1] = ((mb1_eta_HQ & 0x02) >> 1);
          qualBTI[2] = ((mb1_eta_HQ & 0x04) >> 2);
          qualBTI[3] = ((mb1_eta_HQ & 0x08) >> 3);
          qualBTI[4] = ((mb1_eta_HQ & 0x10) >> 4);
          qualBTI[5] = ((mb1_eta_HQ & 0x20) >> 5);
          qualBTI[6] = (((mb1_eta_HQ & 0x40) >> 6) || ((mb1_eta_HQ & 0x80) >> 7));

          theSegments.push_back(L1MuDTChambThDigi(bxNr, wheel, sector - 1, 1, posBTI, qualBTI));

          //MB2
          posBTI[0] = (mb2_eta & 0x01);
          posBTI[1] = ((mb2_eta & 0x02) >> 1);
          posBTI[2] = ((mb2_eta & 0x04) >> 2);
          posBTI[3] = ((mb2_eta & 0x08) >> 3);
          posBTI[4] = ((mb2_eta & 0x10) >> 4);
          posBTI[5] = ((mb2_eta & 0x20) >> 5);
          posBTI[6] = (((mb2_eta & 0x40) >> 6) || ((mb2_eta & 0x80) >> 7));

          qualBTI[0] = (mb2_eta_HQ & 0x01);
          qualBTI[1] = ((mb2_eta_HQ & 0x02) >> 1);
          qualBTI[2] = ((mb2_eta_HQ & 0x04) >> 2);
          qualBTI[3] = ((mb2_eta_HQ & 0x08) >> 3);
          qualBTI[4] = ((mb2_eta_HQ & 0x10) >> 4);
          qualBTI[5] = ((mb2_eta_HQ & 0x20) >> 5);
          qualBTI[6] = (((mb2_eta_HQ & 0x40) >> 6) || ((mb2_eta_HQ & 0x80) >> 7));

          theSegments.push_back(L1MuDTChambThDigi(bxNr, wheel, sector - 1, 2, posBTI, qualBTI));

          //MB3
          posBTI[0] = (mb3_eta & 0x01);
          posBTI[1] = ((mb3_eta & 0x02) >> 1);
          posBTI[2] = ((mb3_eta & 0x04) >> 2);
          posBTI[3] = ((mb3_eta & 0x08) >> 3);
          posBTI[4] = ((mb3_eta & 0x10) >> 4);
          posBTI[5] = ((mb3_eta & 0x20) >> 5);
          posBTI[6] = (((mb3_eta & 0x40) >> 6) || ((mb3_eta & 0x80) >> 7));

          qualBTI[0] = (mb3_eta_HQ & 0x01);
          qualBTI[1] = ((mb3_eta_HQ & 0x02) >> 1);
          qualBTI[2] = ((mb3_eta_HQ & 0x04) >> 2);
          qualBTI[3] = ((mb3_eta_HQ & 0x08) >> 3);
          qualBTI[4] = ((mb3_eta_HQ & 0x10) >> 4);
          qualBTI[5] = ((mb3_eta_HQ & 0x20) >> 5);
          qualBTI[6] = (((mb3_eta_HQ & 0x40) >> 6) || ((mb3_eta_HQ & 0x80) >> 7));

          theSegments.push_back(L1MuDTChambThDigi(bxNr, wheel, sector - 1, 3, posBTI, qualBTI));

        }  //etha word

        else if (selector == 0xB) {  //MB1/2 output word

          int mb2_phi = (dataWordSub & 0xFFF);         // positions  0 -> 11
          int mb2_phib = (dataWordSub >> 12) & 0x3FF;  // positions 12 -> 21
          int mb2_qual = (dataWordSub >> 22) & 0x7;    // positions 22 -> 24
          int mb2_q3 = (dataWordSub >> 25) & 0x1;      // positions 25
          int mb2_q4 = (dataWordSub >> 26) & 0x1;      // positions 26
          int mb2_ts2tag = (dataWordSub >> 28) & 0x1;  // positions 28
          /*** NOT UNPACKED  
			int mb2_parity = ( dataWordSub >> 29) & 0x1;    // positions 29
			***/

          int mb1_phi = (dataWordSub >> 30) & 0xFFF;   // positions 30 -> 41
          int mb1_phib = (dataWordSub >> 42) & 0x3FF;  // positions 42 -> 51
          int mb1_qual = (dataWordSub >> 52) & 0x7;    // positions 52 -> 54
          int mb1_q3 = (dataWordSub >> 55) & 0x1;      // positions 55
          int mb1_q4 = (dataWordSub >> 56) & 0x1;      // positions 56
          int mb1_ts2tag = (dataWordSub >> 58) & 0x1;  // positions 58
          /*** NOT UNPACKED  
			int mb1_parity = ( dataWordSub >> 59 ) &0x1;    // positions 59
			***/

          int mb1_phi_conv = radAngConversion(mb1_phi);
          int mb1_phib_conv = benAngConversion(mb1_phib);

          int mb2_phi_conv = radAngConversion(mb2_phi);
          int mb2_phib_conv = benAngConversion(mb2_phib);

          phioutSegments.push_back(L1MuDTChambPhDigi(bxNr,
                                                     wheel,
                                                     sector - 1,
                                                     1,
                                                     mb1_phi_conv,
                                                     mb1_phib_conv,
                                                     mb1_qual,
                                                     mb1_ts2tag,
                                                     bxCounter,
                                                     mb1_q3 + 2 * mb1_q4));
          phioutSegments.push_back(L1MuDTChambPhDigi(bxNr,
                                                     wheel,
                                                     sector - 1,
                                                     2,
                                                     mb2_phi_conv,
                                                     mb2_phib_conv,
                                                     mb2_qual,
                                                     mb2_ts2tag,
                                                     bxCounter,
                                                     mb2_q3 + 2 * mb2_q4));

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t|\t"
                    << "mb1_ts2tag_out " << mb1_ts2tag << '\t' << "mb1_qual_out " << mb1_qual << '\t' << "mb1_q3_out "
                    << mb1_q3 << '\t' << "mb1_q4_out " << mb1_q4 << '\t' << "mb1_phib_out " << mb1_phib_conv << '\t'
                    << "mb1_phi_out " << mb1_phi_conv << '\t' << "mb2_ts2tag_out " << mb2_ts2tag << '\t'
                    << "mb2_qual_out " << mb2_qual << '\t' << "mb2_q3_out " << mb2_q3 << '\t' << "mb2_q4_out " << mb2_q4
                    << '\t' << "mb2_phib_out " << mb2_phib_conv << '\t' << "mb2_phi_out " << mb2_phi_conv << '\n';

        }  //MB1/2 output word

        else if (selector == 0xC) {  //MB3/4 output word

          int mb4_phi = (dataWordSub & 0xFFF);         // positions  0 -> 11
          int mb4_phib = (dataWordSub >> 12) & 0x3FF;  // positions 12 -> 21
          int mb4_qual = (dataWordSub >> 22) & 0x7;    // positions 22 -> 24
          int mb4_q3 = (dataWordSub >> 25) & 0x1;      // positions 25
          int mb4_q4 = (dataWordSub >> 26) & 0x1;      // positions 26
          int mb4_ts2tag = (dataWordSub >> 28) & 0x1;  // positions 28
          /*** NOT UNPACKED  
			int mb4_parity = ( dataWordSub >> 29) & 0x1;    // positions 29
			***/

          int mb3_phi = (dataWordSub >> 30) & 0xFFF;   // positions 30 -> 41
          int mb3_phib = (dataWordSub >> 42) & 0x3FF;  // positions 42 -> 51
          int mb3_qual = (dataWordSub >> 52) & 0x7;    // positions 52 -> 54
          int mb3_q3 = (dataWordSub >> 55) & 0x1;      // positions 55
          int mb3_q4 = (dataWordSub >> 56) & 0x1;      // positions 56
          int mb3_ts2tag = (dataWordSub >> 58) & 0x1;  // positions 58
          /*** NOT UNPACKED  
			int mb3_parity = ( dataWordSub >> 59 ) &0x1;    // positions 59
			***/

          int mb3_phi_conv = radAngConversion(mb3_phi);
          int mb3_phib_conv = benAngConversion(mb3_phib);

          int mb4_phi_conv = radAngConversion(mb4_phi);
          int mb4_phib_conv = benAngConversion(mb4_phib);

          phioutSegments.push_back(L1MuDTChambPhDigi(bxNr,
                                                     wheel,
                                                     sector - 1,
                                                     3,
                                                     mb3_phi_conv,
                                                     mb3_phib_conv,
                                                     mb3_qual,
                                                     mb3_ts2tag,
                                                     bxCounter,
                                                     mb3_q3 + 2 * mb3_q4));
          phioutSegments.push_back(L1MuDTChambPhDigi(bxNr,
                                                     wheel,
                                                     sector - 1,
                                                     4,
                                                     mb4_phi_conv,
                                                     mb4_phib_conv,
                                                     mb4_qual,
                                                     mb4_ts2tag,
                                                     bxCounter,
                                                     mb4_q3 + 2 * mb4_q4));

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t|\t"
                    << "mb3_ts2tag_out " << mb3_ts2tag << '\t' << "mb3_qual_out " << mb3_qual << '\t' << "mb3_q3_out "
                    << mb3_q3 << '\t' << "mb3_q4_out " << mb3_q4 << '\t' << "mb3_phib_out " << mb3_phib_conv << '\t'
                    << "mb3_phi_out " << mb3_phi_conv << '\t' << "mb4_ts2tag_out " << mb4_ts2tag << '\t'
                    << "mb4_qual_out " << mb4_qual << '\t' << "mb4_q3_out " << mb4_q3 << '\t' << "mb4_q4_out " << mb4_q4
                    << '\t' << "mb4_phib_out " << mb4_phib_conv << '\t' << "mb4_phi_out " << mb4_phi_conv << '\n';

        }  //MB3/4 output word

        else if (selector == 0xD) {  //etha output word

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t ETHA OUTPUT WORD\n";

        }  //etha output word

        else if (selector == 0x9 || selector == 0xE) {  //RPC word

          LogDebug("TwinMux_unpacker") << "RPC WORD [" << std::dec << tm7eventsize << "] : " << std::hex << dataWordSub
                                       << std::dec << " it pos " << int(DTTM7iterator - DTTM7itend);

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t RPC WORD\n";

        }  //RPC word

        else if (selector == 0x6) {  //HO word

          LogDebug("TwinMux_unpacker") << "HO WORD [" << std::dec << tm7eventsize << "] : " << std::hex << dataWordSub
                                       << std::dec << " it pos " << int(DTTM7iterator - DTTM7itend);

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t HO WORD\n";

        }  //HO word

        else if (selector == 0xF) {  //ERROR word

          LogDebug("TwinMux_unpacker") << "ERROR WORD [" << std::dec << tm7eventsize << "] : " << std::hex
                                       << dataWordSub << std::dec << " it pos " << int(DTTM7iterator - DTTM7itend);

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t ERROR WORD\n";
        }  //ERROR word

        else {  //unkown word

          LogDebug("TwinMux_unpacker") << "UNKNOWN WORD received " << std::hex << dataWordSub << " in FED " << std::hex
                                       << TM7fedId;

          if (debug_)
            logfile << '[' << ++lcounter << "]\t" << std::hex << dataWordSub << std::dec << "\t UNKNOWN WORD\n";
        }

        if (DTTM7iterator == DTTM7itend)
          break;
      }
    }  //end of loop over AMCsize

    /// Trailer of payload with CRC
    ++DTTM7iterator;

    if (DTTM7iterator == DTTM7itend)
      break;

  }  // end for-loop container content

  return;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TTwinMuxRawToDigi);
