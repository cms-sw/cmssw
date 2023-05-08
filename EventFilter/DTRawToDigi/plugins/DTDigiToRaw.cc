#include <EventFilter/DTRawToDigi/plugins/DTDigiToRaw.h>
#include <DataFormats/DTDigi/interface/DTDDUWords.h>

#include <cmath>
#include <iostream>

using namespace edm;
using namespace std;

DTDigiToRaw::DTDigiToRaw(const edm::ParameterSet& ps) : pset(ps) {
  debug = pset.getUntrackedParameter<bool>("debugMode", false);
  if (debug)
    cout << "[DTDigiToRaw]: constructor" << endl;
}

DTDigiToRaw::~DTDigiToRaw() {
  if (debug)
    cout << "[DTDigiToRaw]: destructor" << endl;
}

FEDRawData* DTDigiToRaw::createFedBuffers(const DTDigiCollection& digis, edm::ESHandle<DTReadOutMapping>& map) {
  int NROS = 12;
  int NROB = 25;

  vector<uint32_t> words;

  uint32_t fakeROSHeaderWord = DTROSWordType::headerControlWord << WORDCONTROLSHIFT | DTROSWordType::rosTypeWord
                                                                                          << WORDTYPESHIFT;

  uint32_t fakeROSTrailerWord = DTROSWordType::trailerControlWord << WORDCONTROLSHIFT | DTROSWordType::rosTypeWord
                                                                                            << WORDTYPESHIFT;

  int NWords = 2;
  words.push_back(0);
  words.push_back(0);

  DTDigiCollection::DigiRangeIterator detUnitIt;
  detUnitIt = digis.begin();

  bool b_ros[12] = {false, false, false, false, false, false, false, false, false, false, false, false};
  vector<uint32_t> w_ROBROS[12][25];

  for (detUnitIt = digis.begin(); detUnitIt != digis.end(); ++detUnitIt) {
    const DTLayerId layerId = (*detUnitIt).first;
    const DTDigiCollection::Range& digiRange = (*detUnitIt).second;
    // Loop over all digis in the given range

    for (DTDigiCollection::const_iterator digi = digiRange.first; digi != digiRange.second; digi++) {
      int dduId = -1, rosId = -1, robId = -1, tdcId = -1, channelId = -1;

      int layer = layerId.layer();
      DTSuperLayerId superlayerID = layerId.superlayerId();
      int superlayer = superlayerID.superlayer();
      DTChamberId chamberID = superlayerID.chamberId();
      int station = chamberID.station();
      int wheel = chamberID.wheel();
      int sector = chamberID.sector();

      int searchstatus = map->geometryToReadOut(wheel,
                                                station,
                                                sector,
                                                superlayer,
                                                layer,
                                                (*digi).wire(),  //"input"
                                                dduId,
                                                rosId,
                                                robId,
                                                tdcId,
                                                channelId);  //"output"

      if (searchstatus == 1 && debug)
        cout << "[DTDigiToRaw]: warning, geometryToReadOut status = 1" << endl;

      //create FED corresponding to current ddu
      if (dduID_ != dduId)
        continue;

      DTTDCMeasurementWord dttdc_mw;
      uint32_t word;
      int ntdc = (*digi).countsTDC();
      dttdc_mw.set(word, 0, 0, 0, tdcId, channelId, ntdc * 4);  //FIXME

      //provisional fix
      DTTDCMeasurementWord tdcMeasurementWord(word);
      int tdcIDCheck = tdcMeasurementWord.tdcID();
      int tdcChannelCheck = tdcMeasurementWord.tdcChannel();
      int tdcCountsCheck = tdcMeasurementWord.tdcTime();
      if (tdcIDCheck == tdcId && channelId == tdcChannelCheck && ntdc == tdcCountsCheck) {
        if (rosId <= NROS && rosId > 0)
          b_ros[rosId - 1] = true;
        else if (debug) {
          cout << "[DTDigiToRaw]: invalid value for rosId" << endl;
        }

        w_ROBROS[rosId - 1][robId].push_back(word);
      }
    }
  }

  uint32_t therosList = 0;
  for (int i_ros = 0; i_ros < NROS; i_ros++) {
    if (b_ros[i_ros])
      therosList += uint32_t(pow(2.0, i_ros));
  }

  if (debug)
    cout << "[DTDigiToRaw]: therosList = " << therosList << endl;

  for (int i_ros = 0; i_ros < NROS; i_ros++) {
    if (b_ros[i_ros]) {
      words.push_back(fakeROSHeaderWord);
      NWords++;
    }

    for (int i_rob = 0; i_rob < NROB; i_rob++) {
      vector<uint32_t>::const_iterator i_robros;
      if (w_ROBROS[i_ros][i_rob].begin() != w_ROBROS[i_ros][i_rob].end()) {
        uint32_t word = 0;
        DTROBHeaderWord rob_header;
        rob_header.set(word, i_rob, 0, 0);
        //static void set(uint32_t &word, int rob_id, int event_id, int bunch_id)
        words.push_back(word);
        NWords++;

        int n_robros = 0;
        for (i_robros = w_ROBROS[i_ros][i_rob].begin(); i_robros != w_ROBROS[i_ros][i_rob].end(); i_robros++) {
          NWords++;
          words.push_back((*i_robros));
          n_robros++;
        }

        NWords++;
        DTROBTrailerWord rob_trailer;
        rob_trailer.set(word, i_rob, 0, n_robros + 2);
        //static void set(uint32_t &word, int rob_id, int event_id, int word_count)
        words.push_back(word);
      }
    }

    if (b_ros[i_ros]) {
      words.push_back(fakeROSTrailerWord);
      NWords++;
    }
  }

  if (NWords % 2 == 1) {
    words.push_back(0);
    NWords++;
  }

  words.push_back(0);
  words.push_back(0);
  //include rosList in raw data information
  uint32_t secondstatusword = therosList << 16;
  words.push_back(secondstatusword);

  words.push_back(0);
  words.push_back(0);
  words.push_back(0);

  // Write Raw Data
  int dataSize = words.size() * sizeof(Word32);
  FEDRawData* rawData = new FEDRawData(dataSize);
  Word64* word64 = reinterpret_cast<Word64*>(rawData->data());
  for (unsigned int i = 0; i < words.size(); i += 2) {
    *word64 = (Word64(words[i]) << 32) | words[i + 1];
    word64++;
  }

  return rawData;
}

void DTDigiToRaw::SetdduID(int x) { dduID_ = x; }
