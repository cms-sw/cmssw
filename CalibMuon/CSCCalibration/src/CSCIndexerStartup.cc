#include <CalibMuon/CSCCalibration/interface/CSCIndexerStartup.h>


CSCIndexerStartup::~CSCIndexerStartup() {}


std::pair<CSCDetId, CSCIndexerBase::IndexType>  CSCIndexerStartup::detIdFromStripChannelIndex( LongIndexType isi ) const
{
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

  if ( isi <= lastnonme42 ) // Chambers as of 2008 Installation
  {
    if ( isi > lastplusznonme42 )
    {
      ie = 2;
      isi -= lastplusznonme42;
    }
    if ( isi > lastme13 ) // after ME13
    {
      istart = lastme13;
      layerOffset = lastme13layer;
    }
    else if ( isi >= firstme13 ) // ME13
    {
      istart = firstme13 - 1;
      layerOffset = firstme13layer - 1;
      nchan = 64;
    }
  }
  else // ME42 chambers
  {
    istart = lastnonme42;
    layerOffset = lastnonme42layer;
  }

  isi -= istart; // remove earlier group(s)
  IndexType ichan = (isi - 1)%nchan + 1;
  IndexType ili = (isi - 1)/nchan + 1;
  ili += layerOffset; // add appropriate offset for earlier group(s)
  if ( ie != 1 ) ili += lastplusznonme42layer; // add offset to -z endcap; ME42 doesn't need this.

  return std::pair<CSCDetId, IndexType>(detIdFromLayerIndex(ili), ichan);
}


std::pair<CSCDetId, CSCIndexerBase::IndexType>  CSCIndexerStartup::detIdFromChipIndex( IndexType ici ) const
{
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

  if ( ici <= lastnonme42 ) // Chambers as of 2008 Installation
  {
    if ( ici > lastplusznonme42 )
    {
      ie = 2;
      ici -= lastplusznonme42;
    }
    if ( ici > lastme13 ) // after ME13
    {
      istart = lastme13;
      layerOffset = lastme13layer;
    }
    else if ( ici >= firstme13 ) // ME13
    {
      istart = firstme13 - 1;
      layerOffset = firstme13layer - 1;
      nchipPerLayer = 4;
    }
  }
  else // ME42 chambers
  {
    istart = lastnonme42;
    layerOffset = lastnonme42layer;
  }

  ici -= istart; // remove earlier group(s)
  IndexType ichip = (ici - 1)%nchipPerLayer + 1;
  IndexType ili = (ici - 1)/nchipPerLayer + 1;
  ili += layerOffset; // add appropriate offset for earlier group(s)
  if ( ie != 1 ) ili += lastplusznonme42layer; // add offset to -z endcap; ME42 doesn't need this.

  return std::pair<CSCDetId, IndexType>(detIdFromLayerIndex(ili), ichip);
}


int CSCIndexerStartup::dbIndex(const CSCDetId & id, int & channel) const
{
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


CSCIndexerBase::GasGainIndexType  CSCIndexerStartup::detIdFromGasGainIndex( IndexType igg ) const
{
  const int n_types = 18;
  const IndexType type_starts[n_types] =
    {1,  1081, 4321, 6913, 8533, 13933, 15553, 20953, 22573, 23653, 26893, 29485, 31105, 36505, 38125, 43525, 45145, 50545};
  //+1/1 +1/2  +1/3  +2/1  +2/2  +3/1   +3/2   +4/1   -1/1   -1/2   -1/3   -2/1   -2/2   -3/1   -3/2   -4/1   +4/2   -4/2

  const int endcaps[n_types] =
    {1,  1,    1,    1,    1,    1,     1,     1,     2,     2,     2,     2,     2,     2,     2,     2,     1,     2};
  const int stations[n_types] =
    {1,  1,    1,    2,    2,    3,     3,     4,     1,     1,     1,     2,     2,     3,     3,     4,     4,     4};
  const int rings[n_types] =
    {1,  2,    3,    1,    2,    1,     2,     1,     1,     2,     3,     1,     2,     1,     2,     1,     2,     2};

  // determine chamber type
  std::vector<IndexType> v_type_starts(type_starts, type_starts + n_types);
  int type = int(std::upper_bound(v_type_starts.begin(), v_type_starts.end(), igg) -  v_type_starts.begin()) - 1;

  // determine factors for #HVsectors and #chips
  int sectors_per_layer = sectorsPerLayer(stations[type], rings[type]);
  int chips_per_layer = chipsPerLayer(stations[type], rings[type]);

  IndexType igg_chamber_etc = igg - type_starts[type] + 1;

  IndexType igg_chamber_and_layer = (igg_chamber_etc - 1) / sectors_per_layer + 1;

  // extract chamber & layer
  int chamber = (igg_chamber_and_layer - 1) / 6 + 1;
  int layer   = (igg_chamber_and_layer - 1) % 6 + 1;

  IndexType igg_hvseg_etc         = (igg_chamber_etc - 1) % sectors_per_layer + 1;

  // extract HVsegment and chip numbers
  IndexType hvsegment = (igg_hvseg_etc - 1) / chips_per_layer + 1;
  IndexType chip      = (igg_hvseg_etc - 1) % chips_per_layer + 1;

  CSCDetId id(endcaps[type], stations[type], rings[type], chamber, layer);
  return boost::make_tuple(id, hvsegment, chip);
}
