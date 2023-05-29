#include <DataFormats/MuonDetId/interface/CSCIndexer.h>
#include <iostream>

std::vector<CSCIndexer::IndexType> CSCIndexer::fillChamberLabel() {
  // Fill the member vector which permits decoding of the linear chamber index
  // Logically const since initializes cache only,
  // Beware that the ME42 indices 235-270 within this vector do NOT correspond to
  // their 'real' linear indices (which are 469-504 for +z)
  std::vector<IndexType> tChamberLabel;

  tChamberLabel.resize(271);  // one more than #chambers per endcap. Includes ME42.
  IndexType count = 0;
  tChamberLabel[count] = 0;

  for (IndexType is = 1; is != 5; ++is) {
    IndexType irmax = ringsInStation(is);
    for (IndexType ir = 1; ir != irmax + 1; ++ir) {
      IndexType icmax = chambersInRingOfStation(is, ir);
      for (IndexType ic = 1; ic != icmax + 1; ++ic) {
        tChamberLabel[++count] = is * 1000 + ir * 100 + ic;
      }
    }
  }
  return tChamberLabel;
}

const std::vector<CSCIndexer::IndexType>& CSCIndexer::chamberLabel() {
  static const auto s_chamberLabel = fillChamberLabel();
  return s_chamberLabel;
}

CSCDetId CSCIndexer::detIdFromChamberIndex_OLD(IndexType ici) const {
  // Will not work as is for ME42
  // ============================

  IndexType ie = 1;
  if (ici > 234) {
    ie = 2;
    ici -= 234;
  }
  // Now ici is in range 1-234 (assuming valid input in range 1-468)

  // MEij pairs...
  const IndexType station[] = {0, 1, 1, 1, 2, 2, 3, 3, 4};
  const IndexType ring[] = {0, 1, 2, 3, 1, 2, 1, 2, 1};

  // MEij preceding a given MEij matching linear index above
  const IndexType prevs[] = {0, 0, 1, 1, 1, 2, 2, 3, 3};
  const IndexType prevr[] = {0, 0, 1, 2, 3, 1, 2, 1, 2};

  IndexType is = 4;
  IndexType ir = 1;
  for (IndexType i = 2; i <= 8; ++i) {
    IndexType js = station[i];
    IndexType jr = ring[i];
    // if it's before start of MEjs/jr then it's in the previous MEis/ir
    if (ici < startChamberIndexInEndcap(ie, js, jr)) {
      is = prevs[i];
      ir = prevr[i];
      break;
    }
    // otherwise it's in ME41
  }
  IndexType ic = ici - startChamberIndexInEndcap(ie, is, ir) + 1;

  return CSCDetId(ie, is, ir, ic);
}

CSCDetId CSCIndexer::detIdFromChamberIndex(IndexType ici) const {
  // Expected range of input range argument is 1-540.
  // 1-468 for CSCs installed at 2008 start-up. 469-540 for ME42.

  IndexType ie = 1;
  if (ici > 468) {
    // ME42
    ici -= 234;       // now in range 235-306
    if (ici > 270) {  // -z
      ie = 2;
      ici -= 36;  // now in range 235-270
    }
  } else {            // in range 1-468
    if (ici > 234) {  // -z
      ie = 2;
      ici -= 234;  // now in range 1-234
    }
  }
  IndexType label = chamberLabel()[ici];
  return detIdFromChamberLabel(ie, label);
}

CSCIndexer::IndexType CSCIndexer::chamberLabelFromChamberIndex(IndexType ici) const {
  // This is just for cross-checking

  // Expected range of input range argument is 1-540.
  // 1-468 for CSCs installed at 2008 start-up. 469-540 for ME42.

  if (ici > 468) {
    // ME42
    ici -= 234;       // now in range 235-306
    if (ici > 270) {  // -z
      ici -= 36;      // now in range 235-270
    }
  } else {            // in range 1-468
    if (ici > 234) {  // -z
      ici -= 234;     // now in range 1-234
    }
  }
  return chamberLabel()[ici];
}

CSCDetId CSCIndexer::detIdFromChamberLabel(IndexType ie, IndexType label) const {
  IndexType is = label / 1000;
  label -= is * 1000;
  IndexType ir = label / 100;
  label -= ir * 100;
  IndexType ic = label;

  return CSCDetId(ie, is, ir, ic);
}

CSCDetId CSCIndexer::detIdFromLayerIndex(IndexType ili) const {
  IndexType il = (ili - 1) % 6 + 1;
  IndexType ici = (ili - 1) / 6 + 1;
  CSCDetId id = detIdFromChamberIndex(ici);

  return CSCDetId(id.endcap(), id.station(), id.ring(), id.chamber(), il);
}

std::pair<CSCDetId, CSCIndexer::IndexType> CSCIndexer::detIdFromStripChannelIndex(LongIndexType isi) const {
  const LongIndexType lastnonme42 = 217728;       // channels in 2008 installed chambers
  const LongIndexType lastplusznonme42 = 108864;  // = 217728/2
  const LongIndexType firstme13 = 34561;          // First channel of ME13
  const LongIndexType lastme13 = 48384;           // Last channel of ME13

  const IndexType lastnonme42layer = 2808;
  const IndexType lastplusznonme42layer = 1404;  // = 2808/2
  const IndexType firstme13layer = 433;          // = 72*6 + 1 (ME13 chambers are 72-108 in range 1-234)
  const IndexType lastme13layer = 648;           // = 108*6

  // All chambers but ME13 have 80 channels
  IndexType nchan = 80;

  // Set endcap to +z. This should work for ME42 channels too, since we don't need to calculate its endcap explicitly.
  IndexType ie = 1;

  LongIndexType istart = 0;
  IndexType layerOffset = 0;

  if (isi <= lastnonme42) {
    // Chambers as of 2008 Installation

    if (isi > lastplusznonme42) {
      ie = 2;
      isi -= lastplusznonme42;
    }

    if (isi > lastme13) {  // after ME13
      istart = lastme13;
      layerOffset = lastme13layer;
    } else if (isi >= firstme13) {  // ME13
      istart = firstme13 - 1;
      layerOffset = firstme13layer - 1;
      nchan = 64;
    }
  } else {
    // ME42 chambers

    istart = lastnonme42;
    layerOffset = lastnonme42layer;
  }

  isi -= istart;  // remove earlier group(s)
  IndexType ichan = (isi - 1) % nchan + 1;
  IndexType ili = (isi - 1) / nchan + 1;
  ili += layerOffset;  // add appropriate offset for earlier group(s)
  if (ie != 1)
    ili += lastplusznonme42layer;  // add offset to -z endcap; ME42 doesn't need this.

  return std::pair<CSCDetId, IndexType>(detIdFromLayerIndex(ili), ichan);
}

std::pair<CSCDetId, CSCIndexer::IndexType> CSCIndexer::detIdFromChipIndex(IndexType ici) const {
  const LongIndexType lastnonme42 = 13608;      // chips in 2008 installed chambers
  const LongIndexType lastplusznonme42 = 6804;  // = 13608/2
  const LongIndexType firstme13 = 2161;         // First channel of ME13
  const LongIndexType lastme13 = 3024;          // Last channel of ME13

  const IndexType lastnonme42layer = 2808;
  const IndexType lastplusznonme42layer = 1404;  // = 2808/2
  const IndexType firstme13layer = 433;          // = 72*6 + 1 (ME13 chambers are 72-108 in range 1-234)
  const IndexType lastme13layer = 648;           // = 108*6

  // All chambers but ME13 have 5 chips/layer
  IndexType nchipPerLayer = 5;

  // Set endcap to +z. This should work for ME42 channels too, since we don't need to calculate its endcap explicitly.
  IndexType ie = 1;

  LongIndexType istart = 0;
  IndexType layerOffset = 0;

  if (ici <= lastnonme42) {
    // Chambers as of 2008 Installation

    if (ici > lastplusznonme42) {
      ie = 2;
      ici -= lastplusznonme42;
    }

    if (ici > lastme13) {  // after ME13
      istart = lastme13;
      layerOffset = lastme13layer;
    } else if (ici >= firstme13) {  // ME13
      istart = firstme13 - 1;
      layerOffset = firstme13layer - 1;
      nchipPerLayer = 4;
    }
  } else {
    // ME42 chambers

    istart = lastnonme42;
    layerOffset = lastnonme42layer;
  }

  ici -= istart;  // remove earlier group(s)
  IndexType ichip = (ici - 1) % nchipPerLayer + 1;
  IndexType ili = (ici - 1) / nchipPerLayer + 1;
  ili += layerOffset;  // add appropriate offset for earlier group(s)
  if (ie != 1)
    ili += lastplusznonme42layer;  // add offset to -z endcap; ME42 doesn't need this.

  return std::pair<CSCDetId, IndexType>(detIdFromLayerIndex(ili), ichip);
}

int CSCIndexer::dbIndex(const CSCDetId& id, int& channel) {
  int ec = id.endcap();
  int st = id.station();
  int rg = id.ring();
  int ch = id.chamber();
  int la = id.layer();

  // The channels of ME1A are channels 65-80 of ME11
  if (st == 1 && rg == 4) {
    rg = 1;
    if (channel <= 16)
      channel += 64;  // no trapping for any bizarreness
  }
  return ec * 100000 + st * 10000 + rg * 1000 + ch * 10 + la;
}
