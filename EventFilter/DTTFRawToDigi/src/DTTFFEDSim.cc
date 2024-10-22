//-------------------------------------------------
//
//   Class: DTTFFEDSim
//
//   L1 DT Track Finder Digi-to-Raw
//
//
//
//   Author :
//   J. Troconiz  UAM Madrid
//
//--------------------------------------------------

#include "EventFilter/DTTFRawToDigi/interface/DTTFFEDSim.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/Utilities/interface/DTCRC.h"
#include <DataFormats/FEDRawData/interface/FEDRawData.h>

#include <iostream>

using namespace std;

DTTFFEDSim::DTTFFEDSim(const edm::ParameterSet &pset) : eventNum(0) {
  produces<FEDRawDataCollection>();

  DTDigiInputTag = pset.getParameter<edm::InputTag>("DTDigi_Source");
  DTPHTFInputTag = pset.getParameter<edm::InputTag>("DTTracks_Source");

  ChPh_tok = consumes<L1MuDTChambPhContainer>(DTDigiInputTag);
  ChTh_tok = consumes<L1MuDTChambThContainer>(DTDigiInputTag);
  Trk_tok = consumes<L1MuDTTrackContainer>(DTPHTFInputTag);
}

DTTFFEDSim::~DTTFFEDSim() {}

void DTTFFEDSim::produce(edm::Event &e, const edm::EventSetup &c) {
  FEDRawDataCollection data;

  if (!fillRawData(e, data))
    return;

  unique_ptr<FEDRawDataCollection> fed_product(new FEDRawDataCollection(data));

  e.put(std::move(fed_product));
}

bool DTTFFEDSim::fillRawData(edm::Event &e, FEDRawDataCollection &data) {
  eventNum = e.id().event();

  int lines = 2;

  edm::Handle<L1MuDTChambPhContainer> phtrig;
  e.getByToken(ChPh_tok, phtrig);
  lines += phtrig->bxSize(-1, 1);

  edm::Handle<L1MuDTChambThContainer> thtrig;
  e.getByToken(ChTh_tok, thtrig);
  lines += thtrig->bxSize(-1, 1);

  edm::Handle<L1MuDTTrackContainer> trtrig;
  e.getByToken(Trk_tok, trtrig);
  lines += trtrig->bxSize(-1, 1) * 3;

  FEDRawData &dttfdata = data.FEDData(0x30C);
  dttfdata.resize(lines * 8);  // size in bytes
  unsigned char *LineFED = dttfdata.data();

  int *dataWord1 = new int;
  int *dataWord2 = new int;

  //--> Header

  *dataWord1 = 0x50000000 + (eventNum & 0xFFFFFF);
  *dataWord2 = 0x00030C00;

  int newCRC = 0xFFFF;
  dt_crc::calcCRC(*dataWord1, *dataWord2, newCRC);

  *((int *)LineFED) = *dataWord2;
  LineFED += 4;
  *((int *)LineFED) = *dataWord1;

  //--> DTTF data

  int TS1Id[4], TS2Id[4];  // word identifier for TS #1,#2 for stations
  TS1Id[0] = 0x0E;
  TS2Id[0] = 0x1E;
  TS1Id[1] = 0x2E;
  TS2Id[1] = 0x3E;
  TS1Id[3] = 0x4E;
  TS2Id[3] = 0x5E;
  TS1Id[2] = 0x8FFF8;
  TS2Id[2] = 0x9FFF8;

  // Input
  L1MuDTChambPhContainer::Phi_iterator tsphi;

  for (tsphi = phtrig->getContainer()->begin(); tsphi != phtrig->getContainer()->end(); tsphi++) {
    if (tsphi->code() != 7) {
      int wheelID = tsphi->whNum() + 1;
      if (wheelID <= 0)
        wheelID -= 2;
      int stationID = tsphi->stNum() - 1;
      int is2nd = tsphi->Ts2Tag();

      int channelNr = channel(wheelID, tsphi->scNum(), tsphi->bxNum() - is2nd);
      if (channelNr == 255)
        continue;
      int TSId = (is2nd == 0) ? TS1Id[stationID] : TS2Id[stationID];

      *dataWord1 = ((channelNr & 0xFF) << 24) + 0x00FFFFFF;

      if (stationID != 2) {
        *dataWord2 = ((TSId & 0x0FF) << 24) + (~(tsphi->code() + 1) & 0x007) + ((~tsphi->phiB() & 0x3FF) << 3) +
                     ((~tsphi->phi() & 0xFFF) << 13);
      } else {
        *dataWord2 = ((TSId & 0xFFFFF) << 12) + (~(tsphi->code() + 1) & 0x00007) + ((~tsphi->phi() & 0x00FFF) << 3);
      }

      dt_crc::calcCRC(*dataWord1, *dataWord2, newCRC);

      LineFED += 4;
      *((int *)LineFED) = *dataWord2;
      LineFED += 4;
      *((int *)LineFED) = *dataWord1;
    }
  }
  // Input

  // Input
  L1MuDTChambThContainer::The_iterator tsthe;

  for (tsthe = thtrig->getContainer()->begin(); tsthe != thtrig->getContainer()->end(); tsthe++) {
    int wheelTh = tsthe->whNum();
    int sectorID = tsthe->scNum();

    int channelNr = channel(0, sectorID, tsthe->bxNum());
    if (channelNr == 255)
      continue;
    int TSId = wheelTh + 2;

    *dataWord1 = ((channelNr & 0xFF) << 24) + 0x00FFFFFF;

    *dataWord2 = ((TSId & 0x07) << 28) + 0x0FFFFFFF;

    int stationID = tsthe->stNum() - 1;
    for (int bti = 0; bti < 7; bti++)
      if (wheelTh == -2 || wheelTh == -1 ||
          (wheelTh == 0 &&
           (sectorID == 0 || sectorID == 3 || sectorID == 4 || sectorID == 7 || sectorID == 8 || sectorID == 11)))
        *dataWord2 -= (tsthe->position(bti) & 0x1) << (stationID * 7 + bti);
      else
        *dataWord2 -= (tsthe->position(6 - bti) & 0x1) << (stationID * 7 + bti);

    dt_crc::calcCRC(*dataWord1, *dataWord2, newCRC);

    LineFED += 4;
    *((int *)LineFED) = *dataWord2;
    LineFED += 4;
    *((int *)LineFED) = *dataWord1;
  }
  // Input

  // Output
  L1MuDTTrackContainer::Trackiterator tstrk;

  for (tstrk = trtrig->getContainer()->begin(); tstrk != trtrig->getContainer()->end(); tstrk++) {
    int channelNr = channel(tstrk->whNum(), tstrk->scNum(), tstrk->bx());
    if (channelNr == 255)
      continue;
    int TSId = (tstrk->TrkTag() == 0) ? 0xAFFF : 0xBFFF;

    *dataWord1 = ((channelNr & 0xFF) << 24) + 0x00FFFFFF;

    *dataWord2 = ((TSId & 0xFFFF) << 16) + (tstrk->stNum(4) & 0x0000F) + ((tstrk->stNum(3) & 0x0000F) << 4) +
                 ((tstrk->stNum(2) & 0x0000F) << 8) + ((tstrk->stNum(1) & 0x00003) << 12);

    dt_crc::calcCRC(*dataWord1, *dataWord2, newCRC);

    LineFED += 4;
    *((int *)LineFED) = *dataWord2;
    LineFED += 4;
    *((int *)LineFED) = *dataWord1;

    TSId = (tstrk->TrkTag() == 0) ? 0xCFFE : 0xDFFE;

    *dataWord1 = ((channelNr & 0xFF) << 24) + 0x00FFFFFF;

    *dataWord2 = ((TSId & 0xFFFE) << 16) + (~tstrk->quality_packed() & 0x0007) + ((tstrk->phi_packed() & 0x00FF) << 3) +
                 ((~tstrk->charge_packed() & 0x0001) << 11) + ((~tstrk->pt_packed() & 0x001F) << 12);

    dt_crc::calcCRC(*dataWord1, *dataWord2, newCRC);

    LineFED += 4;
    *((int *)LineFED) = *dataWord2;
    LineFED += 4;
    *((int *)LineFED) = *dataWord1;

    channelNr = channel(0, tstrk->scNum(), tstrk->bx());
    if (channelNr == 255)
      continue;
    TSId = (tstrk->whNum() + 3) << 16;
    TSId += (tstrk->whNum() < 0) ? 0x8FFFC : 0x7FFFC;

    *dataWord1 = ((channelNr & 0xFF) << 24) + 0x00FFFFFF;

    *dataWord2 = (TSId & 0xFFFFC) << 12;

    if (tstrk->TrkTag() == 0) {
      *dataWord2 += 0x3F80 + (tstrk->eta_packed() & 0x003F) + ((~tstrk->finehalo_packed() & 0x0001) << 6);
    } else {
      *dataWord2 += 0x007F + ((tstrk->eta_packed() & 0x003F) << 7) + ((~tstrk->finehalo_packed() & 0x0001) << 13);
    }

    dt_crc::calcCRC(*dataWord1, *dataWord2, newCRC);

    LineFED += 4;
    *((int *)LineFED) = *dataWord2;
    LineFED += 4;
    *((int *)LineFED) = *dataWord1;
  }
  // Output

  //--> Trailer

  *dataWord1 = 0xA0000000 + (lines & 0xFFFFFF);
  *dataWord2 = 0;

  dt_crc::calcCRC(*dataWord1, *dataWord2 & 0xFFFF, newCRC);

  *dataWord2 += (newCRC & 0xFFFF) << 16;

  LineFED += 4;
  *((int *)LineFED) = *dataWord2;
  LineFED += 4;
  *((int *)LineFED) = *dataWord1;

  delete dataWord1;
  delete dataWord2;
  return true;
}

int DTTFFEDSim::channel(int wheel, int sector, int bx) {
  // wheel  :  -3 -2 -1 +1 +2 +3 <=> PHTF's : N2, N1, N0, P0, P1, P2
  //                           0 <=> ETTF
  // sector :  0 -> 11
  // bx     : -1 -> +1

  int myChannel = 255;

  if (abs(bx) > 1) {
    return myChannel;
  }
  if (sector < 0 || sector > 11) {
    return myChannel;
  }
  if (abs(wheel) > 3) {
    return myChannel;
  }

  myChannel = sector * 21 + wheel * 3 - bx + 10;

  if (myChannel > 125)
    myChannel += 2;

  return myChannel;
}

int DTTFFEDSim::bxNr(int channel) {
  int myChannel = channel;

  if (myChannel > 127)
    myChannel -= 2;

  if (myChannel < 0 || myChannel > 251) {
    return -999;
  }

  int myBx = 1 - (myChannel % 3);

  return myBx;
}

int DTTFFEDSim::sector(int channel) {
  int myChannel = channel;

  if (myChannel > 127)
    myChannel -= 2;

  if (myChannel < 0 || myChannel > 251) {
    return -999;
  }

  return myChannel / 21;
}

int DTTFFEDSim::wheel(int channel) {
  int myChannel = channel;

  if (myChannel > 127)
    myChannel -= 2;

  if (myChannel < 0 || myChannel > 251) {
    return -999;
  }

  int myWheel = ((myChannel % 21) / 3) - 3;

  return myWheel;
}
