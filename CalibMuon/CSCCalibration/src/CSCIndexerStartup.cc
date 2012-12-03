#include <CalibMuon/CSCCalibration/interface/CSCIndexerStartup.h>

std::pair<CSCDetId, CSCIndexerBase::IndexType>  CSCIndexerStartup::detIdFromStripChannelIndex( LongIndexType isi ) const {

  const LongIndexType lastnonme42 = 217728; // channels in 2008 installed chambers
  const LongIndexType lastplusznonme42 = 108864; // = 217728/2
  const LongIndexType firstme13  = 34561; // First channel of ME13
  const LongIndexType lastme13   = 48384; // Last channel of ME13

  const IndexType lastnonme42layer = 2808;
  const IndexType lastplusznonme42layer = 1404; // = 2808/2
  const IndexType firstme13layer  = 433; // = 72*6 + 1 (ME13 chambers are 72-108 in range 1-234)
  const IndexType lastme13layer   = 648; // = 108*6

  // All chambers but ME13 have 80 channels
  IndexType nchan = 80;

  // Set endcap to +z. This should work for ME42 channels too, since we don't need to calculate its endcap explicitly.
  IndexType ie = 1;

  LongIndexType istart = 0;
  IndexType layerOffset = 0;

  if ( isi <= lastnonme42 ) {
    // Chambers as of 2008 Installation

    if ( isi > lastplusznonme42 ) {
      ie = 2;
      isi -= lastplusznonme42;
    }

    if ( isi > lastme13 ) { // after ME13
      istart = lastme13;
      layerOffset = lastme13layer;
    }
    else if ( isi >= firstme13 ) { // ME13
      istart = firstme13 - 1;
      layerOffset = firstme13layer - 1;
      nchan = 64;
    }
  }
  else {
     // ME42 chambers

    istart = lastnonme42;
    layerOffset = lastnonme42layer;
  }

   isi -= istart; // remove earlier group(s)
   IndexType ichan = (isi-1)%nchan + 1;
   IndexType ili = (isi-1)/nchan + 1;
   ili += layerOffset; // add appropriate offset for earlier group(s)
   if ( ie != 1 ) ili+= lastplusznonme42layer; // add offset to -z endcap; ME42 doesn't need this.

   return std::pair<CSCDetId, IndexType>(detIdFromLayerIndex(ili), ichan);
}

std::pair<CSCDetId, CSCIndexerBase::IndexType>  CSCIndexerStartup::detIdFromChipIndex( IndexType ici ) const {

  const LongIndexType lastnonme42 = 13608; // chips in 2008 installed chambers
  const LongIndexType lastplusznonme42 = 6804; // = 13608/2
  const LongIndexType firstme13  = 2161; // First channel of ME13
  const LongIndexType lastme13   = 3024; // Last channel of ME13

  const IndexType lastnonme42layer = 2808;
  const IndexType lastplusznonme42layer = 1404; // = 2808/2
  const IndexType firstme13layer  = 433; // = 72*6 + 1 (ME13 chambers are 72-108 in range 1-234)
  const IndexType lastme13layer   = 648; // = 108*6

  // All chambers but ME13 have 5 chips/layer
  IndexType nchipPerLayer = 5;

  // Set endcap to +z. This should work for ME42 channels too, since we don't need to calculate its endcap explicitly.
  IndexType ie = 1;

  LongIndexType istart = 0;
  IndexType layerOffset = 0;

  if ( ici <= lastnonme42 ) {
    // Chambers as of 2008 Installation

    if ( ici > lastplusznonme42 ) {
      ie = 2;
      ici -= lastplusznonme42;
    }

    if ( ici > lastme13 ) { // after ME13
      istart = lastme13;
      layerOffset = lastme13layer;
    }
    else if ( ici >= firstme13 ) { // ME13
      istart = firstme13 - 1;
      layerOffset = firstme13layer - 1;
      nchipPerLayer = 4;
    }
  }
  else {
     // ME42 chambers

    istart = lastnonme42;
    layerOffset = lastnonme42layer;
  }

   ici -= istart; // remove earlier group(s)
   IndexType ichip = (ici-1)%nchipPerLayer + 1;
   IndexType ili = (ici-1)/nchipPerLayer + 1;
   ili += layerOffset; // add appropriate offset for earlier group(s)
   if ( ie != 1 ) ili+= lastplusznonme42layer; // add offset to -z endcap; ME42 doesn't need this.

   return std::pair<CSCDetId, IndexType>(detIdFromLayerIndex(ili), ichip);
}

int CSCIndexerStartup::dbIndex(const CSCDetId & id, int & channel) const {
  int ec = id.endcap();
  int st = id.station();
  int rg = id.ring();
  int ch = id.chamber();
  int la = id.layer();

  // The channels of ME1A are channels 65-80 of ME11
  if(st == 1 && rg == 4)
    {
      rg = 1;
      if(channel <= 16) channel += 64; // no trapping for any bizarreness
    }
  return ec*100000 + st*10000 + rg*1000 + ch*10 + la;
}
