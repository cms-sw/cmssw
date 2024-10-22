#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "EventFilter/ESDigiToRaw/src/ESDataFormatterV4.h"

using namespace std;
using namespace edm;

const int ESDataFormatterV4::bDHEAD = 2;
const int ESDataFormatterV4::bDH = 6;
const int ESDataFormatterV4::bDEL = 24;
const int ESDataFormatterV4::bDERR = 8;
const int ESDataFormatterV4::bDRUN = 24;
const int ESDataFormatterV4::bDRUNTYPE = 32;
const int ESDataFormatterV4::bDTRGTYPE = 16;
const int ESDataFormatterV4::bDCOMFLAG = 8;
const int ESDataFormatterV4::bDORBIT = 32;
const int ESDataFormatterV4::bDVMAJOR = 8;
const int ESDataFormatterV4::bDVMINOR = 8;
const int ESDataFormatterV4::bDCH = 4;
const int ESDataFormatterV4::bDOPTO = 8;

const int ESDataFormatterV4::sDHEAD = 28;
const int ESDataFormatterV4::sDH = 24;
const int ESDataFormatterV4::sDEL = 0;
const int ESDataFormatterV4::sDERR = bDEL + sDEL;
const int ESDataFormatterV4::sDRUN = 0;
const int ESDataFormatterV4::sDRUNTYPE = 0;
const int ESDataFormatterV4::sDTRGTYPE = 0;
const int ESDataFormatterV4::sDCOMFLAG = bDTRGTYPE + sDTRGTYPE;
const int ESDataFormatterV4::sDORBIT = 0;
const int ESDataFormatterV4::sDVMINOR = 8;
const int ESDataFormatterV4::sDVMAJOR = bDVMINOR + sDVMINOR;
const int ESDataFormatterV4::sDCH = 0;
const int ESDataFormatterV4::sDOPTO = 16;

const int ESDataFormatterV4::bKEC = 8;  // KCHIP packet event counter
const int ESDataFormatterV4::bKFLAG2 = 8;
const int ESDataFormatterV4::bKBC = 12;  // KCHIP packet bunch counter
const int ESDataFormatterV4::bKFLAG1 = 4;
const int ESDataFormatterV4::bKET = 1;
const int ESDataFormatterV4::bKCRC = 1;
const int ESDataFormatterV4::bKCE = 1;
const int ESDataFormatterV4::bKID = 16;
const int ESDataFormatterV4::bFIBER = 6;  // Fiber number
const int ESDataFormatterV4::bKHEAD1 = 2;
const int ESDataFormatterV4::bKHEAD2 = 2;
const int ESDataFormatterV4::bKHEAD = 4;

const int ESDataFormatterV4::sKEC = 16;
const int ESDataFormatterV4::sKFLAG2 = 16;
const int ESDataFormatterV4::sKBC = 0;
const int ESDataFormatterV4::sKFLAG1 = 24;
const int ESDataFormatterV4::sKET = 0;
const int ESDataFormatterV4::sKCRC = bKET + sKET;
const int ESDataFormatterV4::sKCE = bKCRC + sKCRC;
const int ESDataFormatterV4::sKID = 0;
const int ESDataFormatterV4::sFIBER = bKID + sKID + 1;
const int ESDataFormatterV4::sKHEAD1 = bFIBER + sFIBER + 2;
const int ESDataFormatterV4::sKHEAD2 = bKHEAD1 + sKHEAD1;
const int ESDataFormatterV4::sKHEAD = 28;

const int ESDataFormatterV4::bADC0 = 16;
const int ESDataFormatterV4::bADC1 = 16;
const int ESDataFormatterV4::bADC2 = 16;
const int ESDataFormatterV4::bPACE = 2;
const int ESDataFormatterV4::bSTRIP = 5;
const int ESDataFormatterV4::bE0 = 1;
const int ESDataFormatterV4::bE1 = 1;
const int ESDataFormatterV4::bHEAD = 4;

const int ESDataFormatterV4::sADC0 = 0;
const int ESDataFormatterV4::sADC1 = bADC0 + sADC0;
const int ESDataFormatterV4::sADC2 = 0;
const int ESDataFormatterV4::sSTRIP = bADC2 + sADC2;
const int ESDataFormatterV4::sPACE = bSTRIP + sSTRIP;
const int ESDataFormatterV4::sE0 = bSTRIP + sSTRIP + 1;
const int ESDataFormatterV4::sE1 = bE0 + sE0;
const int ESDataFormatterV4::sHEAD = 28;

const int ESDataFormatterV4::bOEMUTTCEC = 32;
const int ESDataFormatterV4::bOEMUTTCBC = 16;
const int ESDataFormatterV4::bOEMUKEC = 8;
const int ESDataFormatterV4::bOHEAD = 4;

const int ESDataFormatterV4::sOEMUTTCEC = 0;
const int ESDataFormatterV4::sOEMUTTCBC = 0;
const int ESDataFormatterV4::sOEMUKEC = 16;
const int ESDataFormatterV4::sOHEAD = 28;

ESDataFormatterV4::ESDataFormatterV4(const ParameterSet& ps) : ESDataFormatter(ps) {
  lookup_ = ps.getUntrackedParameter<FileInPath>("LookupTable");

  // initialize look-up table
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 40; ++k)
        for (int m = 0; m < 40; m++) {
          fedId_[i][j][k][m] = -1;
          kchipId_[i][j][k][m] = -1;
          paceId_[i][j][k][m] = -1;
          bundleId_[i][j][k][m] = -1;
          fiberId_[i][j][k][m] = -1;
          optoId_[i][j][k][m] = -1;
        }

  for (int i = 0; i < 56; ++i) {
    for (int j = 0; j < 3; ++j)
      fedIdOptoRx_[i][j] = false;
  }

  for (int i = 0; i < 56; ++i) {
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 12; k++)
        fedIdOptoRxFiber_[i][j][k] = false;
  }

  // read in look-up table
  int nLines, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  ifstream file;
  file.open(lookup_.fullPath().c_str());
  if (file.is_open()) {
    file >> nLines;

    for (int i = 0; i < nLines; ++i) {
      int fedId = -1;
      file >> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;

      fedId = fedId_[(3 - iz) / 2 - 1][ip - 1][ix - 1][iy - 1] = fed;
      kchipId_[(3 - iz) / 2 - 1][ip - 1][ix - 1][iy - 1] = kchip;
      paceId_[(3 - iz) / 2 - 1][ip - 1][ix - 1][iy - 1] = pace - 1;
      bundleId_[(3 - iz) / 2 - 1][ip - 1][ix - 1][iy - 1] = bundle;
      fiberId_[(3 - iz) / 2 - 1][ip - 1][ix - 1][iy - 1] = fiber;
      optoId_[(3 - iz) / 2 - 1][ip - 1][ix - 1][iy - 1] = optorx;

      if (fedId < FEDNumbering::MINPreShowerFEDID || fedId > FEDNumbering::MAXPreShowerFEDID) {
        if (debug_)
          cout << "ESDataFormatterV4::ESDataFormatterV4 : fedId value : " << fedId
               << " out of ES range, at lookup table line : " << i << endl;
      } else if (optorx < 1 || optorx > 3) {
        if (debug_)
          cout << "ESDataFormatterV4::ESDataFormatterV4 : optorx value : " << optorx
               << " out of ES range, at lookup table line : " << i << endl;
      } else {  // all good ..
        int fedidx = fed - FEDNumbering::MINPreShowerFEDID;
        fedIdOptoRx_[fedidx][optorx - 1] = true;
        if (fiber > 0 && fiber < 13) {
          fedIdOptoRxFiber_[fedidx][optorx - 1][fiber - 1] = true;
        } else {
          if (debug_)
            cout << "ESDataFormatterV4::ESDataFormatterV4 : fiber value : " << fiber
                 << " out of ES range, at lookup table line : " << i << endl;
        }
      }
    }

  } else {
    if (debug_)
      cout << "ESDataFormatterV4::ESDataFormatterV4 : Look up table file can not be found in "
           << lookup_.fullPath().c_str() << endl;
  }

  file.close();
}

ESDataFormatterV4::~ESDataFormatterV4() {}

struct ltfiber {
  bool operator()(const pair<int, int> s1, const pair<int, int> s2) const { return (s1.second < s2.second); }
};

bool ltstrip(const ESDataFormatterV4::Word64& s1, const ESDataFormatterV4::Word64& s2) {
  ESDataFormatterV4::Word64 PACESTRIP_MASK = 0x00ff000000000000ull;
  ESDataFormatterV4::Word64 PACESTRIP_OFFSET = 48ull;

  ESDataFormatterV4::Word64 val1 = (s1 & PACESTRIP_MASK) >> PACESTRIP_OFFSET;
  ESDataFormatterV4::Word64 val2 = (s2 & PACESTRIP_MASK) >> PACESTRIP_OFFSET;

  return (val1 < val2);
}

void ESDataFormatterV4::DigiToRaw(int fedId, Digis& digis, FEDRawData& fedRawData, Meta_Data const& meta_data) const {
  int ts[3] = {0, 0, 0};
  Word32 word1, word2;
  Word64 word;
  int numberOfStrips = 0;

  int optorx_ch_counts[3][12];

  int kchip, pace, optorx, fiber;
  map<int, vector<Word64> > map_data;
  vector<Word64> words;

  vector<Word32> testVector;

  set<pair<int, int>, ltfiber> set_of_kchip_fiber_in_optorx[3];

  map_data.clear();

  // clean optorx channel status fields:
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 12; ++j)
      optorx_ch_counts[i][j] = 0;

  const DetDigis& detDigis = digis[fedId];

  if (debug_) {
    cout << "ESDataFormatterV4::DigiToRaw : FEDID : " << fedId << " size of detDigis : " << detDigis.size() << endl;
  }

  for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); ++it) {
    const ESDataFrame& dataframe = (*it);
    const ESDetId& detId = dataframe.id();

    for (int is = 0; is < dataframe.size(); ++is)
      ts[is] = dataframe.sample(is).adc();

    kchip = kchipId_[(3 - detId.zside()) / 2 - 1][detId.plane() - 1][detId.six() - 1][detId.siy() - 1];
    pace = paceId_[(3 - detId.zside()) / 2 - 1][detId.plane() - 1][detId.six() - 1][detId.siy() - 1];

    if (debug_)
      cout << "Si : " << detId.zside() << " " << detId.plane() << " " << detId.six() << " " << detId.siy() << " "
           << detId.strip() << " (" << kchip << "," << pace << ") " << ts[0] << " " << ts[1] << " " << ts[2] << endl;

    // convert strip number from detector id to electronics id
    int siz = detId.zside();
    int sip = detId.plane();
    int six = detId.six();
    int siy = detId.siy();
    int sistrip = detId.strip();
    if (siz == 1 && sip == 1 && siy <= 20)
      sistrip = 33 - sistrip;
    if (siz == 1 && sip == 2 && six > 20)
      sistrip = 33 - sistrip;
    if (siz == -1 && sip == 1 && siy > 20)
      sistrip = 33 - sistrip;
    if (siz == -1 && sip == 2 && six <= 20)
      sistrip = 33 - sistrip;

    word1 = (ts[1] << sADC1) | (ts[0] << sADC0);
    word2 = (0xc << sHEAD) | (pace << sPACE) | ((sistrip - 1) << sSTRIP) | (ts[2] << sADC2);
    word = (Word64(word2) << 32) | Word64(word1);

    map_data[kchip].push_back(word);

    optorx = optoId_[(3 - detId.zside()) / 2 - 1][detId.plane() - 1][detId.six() - 1][detId.siy() - 1];
    fiber = fiberId_[(3 - detId.zside()) / 2 - 1][detId.plane() - 1][detId.six() - 1][detId.siy() - 1];

    optorx_ch_counts[optorx - 1][fiber - 1]++;  // increment number of strip hits on fiber status field ;

    set<pair<int, int>, ltfiber>& theSet = set_of_kchip_fiber_in_optorx[optorx - 1];
    theSet.insert(pair<int, int>(kchip, fiber));

    // mark global strip number in this FED
    ++numberOfStrips;
  }

  for (int iopto = 0; iopto < 3; ++iopto) {
    if (fedIdOptoRx_[fedId - FEDNumbering::MINPreShowerFEDID][iopto]) {
      word2 = (0x6 << sOHEAD) | (meta_data.kchip_ec << sOEMUKEC) | (meta_data.kchip_bc << sOEMUTTCBC);
      word1 = (meta_data.kchip_ec << sOEMUTTCEC);
      word = (Word64(word2) << 32) | Word64(word1);
      if (debug_)
        cout << "OPTORX: " << print(word) << endl;
      words.push_back(word);

      set<pair<int, int>, ltfiber>& theSet = set_of_kchip_fiber_in_optorx[iopto];

      if (debug_) {
        cout << "ESDataFormatterV4::DigiToRaw : FEDID : " << fedId << " size of  set_of_kchip_fiber_in_optorx[" << iopto
             << "] : " << theSet.size() << endl;
      }

      set<pair<int, int>, ltfiber>::const_iterator kit = theSet.begin();

      while (kit != theSet.end()) {
        const pair<int, int>& kchip_fiber = (*kit);

        if (debug_)
          cout << "KCHIP : " << kchip_fiber.first << " FIBER: " << kchip_fiber.second << endl;

        if (fedIdOptoRxFiber_[fedId - FEDNumbering::MINPreShowerFEDID][iopto][kchip_fiber.second - 1]) {
          // Set all PACEs enabled for MC
          word1 = (0 << sKFLAG1) | (0xf << sKFLAG2) | (((kchip_fiber.first << 2) | 0x02) << sKID);
          word2 = (0x9 << sKHEAD) | (meta_data.kchip_ec << sKEC) | (meta_data.kchip_bc << sKBC);

          word = (Word64(word2) << 32) | Word64(word1);
          if (debug_)
            cout << "KCHIP : " << print(word) << endl;

          words.push_back(word);

          vector<Word64>& data = map_data[kchip_fiber.first];

          // sort against stripid field, as hardware gives this order to strip data :
          sort(data.begin(), data.end(), ltstrip);

          for (unsigned int id = 0; id < data.size(); ++id) {
            if (debug_)
              cout << "Data  : " << print(data[id]) << endl;
            words.push_back(data[id]);
          }
        }
        ++kit;
      }
    }
  }

  int dataSize = (words.size() + 8) * sizeof(Word64);

  vector<Word64> DCCwords;

  word2 = (3 << sDHEAD) | (1 << sDH) | (meta_data.run_number << sDRUN);
  word1 = (numberOfStrips << sDEL) | (0xff << sDERR);
  word = (Word64(word2) << 32) | Word64(word1);
  DCCwords.push_back(word);

  word2 = (3 << sDHEAD) | (2 << sDH);
  word1 = 0;
  word = (Word64(word2) << 32) | Word64(word1);
  DCCwords.push_back(word);

  word2 = (3 << sDHEAD) | (3 << sDH) | (4 << sDVMAJOR) | (3 << sDVMINOR);
  word1 = (meta_data.orbit_number << sDORBIT);
  word = (Word64(word2) << 32) | Word64(word1);
  DCCwords.push_back(word);

  for (int iopto = 0; iopto < 3; ++iopto) {
    // N optorx module header word:
    word1 = 0;
    if (fedIdOptoRx_[fedId - FEDNumbering::MINPreShowerFEDID][iopto]) {
      word2 = (3 << sDHEAD) | ((iopto + 4) << sDH) | (0x80 << sDOPTO);
      int ich = 0;
      for (ich = 0; ich < 4; ++ich) {
        int chStatus = (optorx_ch_counts[iopto][ich + 8] > 0) ? 0xe : 0xd;
        chStatus = (fedIdOptoRxFiber_[fedId - FEDNumbering::MINPreShowerFEDID][iopto][ich + 8]) ? chStatus : 0x00;
        word2 |= (chStatus << (ich * 4));  //
      }

      for (ich = 0; ich < 8; ++ich) {
        int chStatus = (optorx_ch_counts[iopto][ich] > 0) ? 0xe : 0xd;
        chStatus = (fedIdOptoRxFiber_[fedId - FEDNumbering::MINPreShowerFEDID][iopto][ich]) ? chStatus : 0x00;
        word1 |= (chStatus << (ich * 4));
      }
    } else
      word2 = (3 << sDHEAD) | ((iopto + 4) << sDH) | (0x00 << sDOPTO);

    word = (Word64(word2) << 32) | Word64(word1);
    DCCwords.push_back(word);
  }

  // Output (data size in Bytes)
  // FEDRawData * rawData = new FEDRawData(dataSize);
  fedRawData.resize(dataSize);

  Word64* w = reinterpret_cast<Word64*>(fedRawData.data());

  // header
  FEDHeader::set(reinterpret_cast<unsigned char*>(w), trgtype_, meta_data.lv1, meta_data.bx, fedId);
  w++;

  // ES-DCC
  for (unsigned int i = 0; i < DCCwords.size(); ++i) {
    if (debug_)
      cout << "DCC  : " << print(DCCwords[i]) << endl;
    *w = DCCwords[i];
    w++;
  }

  // event data
  for (unsigned int i = 0; i < words.size(); ++i) {
    *w = words[i];
    w++;
  }

  // trailer
  FEDTrailer::set(reinterpret_cast<unsigned char*>(w),
                  dataSize / sizeof(Word64),
                  evf::compute_crc(fedRawData.data(), dataSize),
                  0,
                  0);
}
