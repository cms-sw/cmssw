#include <CalibMuon/CSCCalibration/interface/CSCIndexerPostls1.h>


CSCIndexerPostls1::~CSCIndexerPostls1() {}


std::pair<CSCDetId, CSCIndexerBase::IndexType>  CSCIndexerPostls1::detIdFromStripChannelIndex( LongIndexType isi ) const
{
  const LongIndexType lastnonme1a       = 252288; // channels with ME42 installed
  const LongIndexType lastpluszme1a     = 262656; // last unganged ME1a +z channel = 252288 + 10368
  const LongIndexType lastnonme42 = 217728; // channels in 2008 installed chambers
  const LongIndexType lastplusznonme42 = 108864; // = 217728/2
  const LongIndexType firstme13  = 34561; // First channel of ME13
  const LongIndexType lastme13   = 48384; // Last channel of ME13

  const IndexType lastnonme42layer = 2808;
  const IndexType lastplusznonme42layer = 1404; // = 2808/2
  const IndexType firstme13layer  = 433; // = 72*6 + 1 (ME13 chambers are 72-108 in range 1-234)
  const IndexType lastme13layer   = 648; // = 108*6

  bool me1a = false;

  // Most chambers (except ME13 & ME1a) have 80 channels index width allocated
  //   unganged ME1a have 48 channels
  //   ME13 have 64 channels
  IndexType nchan = 80;

  // Set endcap to +z initially
  IndexType ie = 1;

  LongIndexType istart = 0;
  IndexType layerOffset = 0;

  if ( isi <= lastnonme42 )
  {
    // Chambers as of 2008 Installation (ME11 keeps the same #of channels 80 allocated for it in the index)
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
  else if ( isi <= lastnonme1a ) // ME42 chambers
  {
    istart = lastnonme42;
    layerOffset = lastnonme42layer;
    // don't care about ie, as ME42 stretch of indices is uniform
  }
  else // Unganged ME1a channels
  {
    me1a = true;
    if (isi > lastpluszme1a) ie = 2;
    istart = lastnonme1a;
    nchan = 48;
    // layerOffset stays 0, as we want to map them onto ME1b's layer indices
  }

  isi -= istart; // remove earlier group(s)
  IndexType ichan = (isi - 1)%nchan + 1;
  IndexType ili = (isi - 1)/nchan + 1;
  ili += layerOffset; // add appropriate offset for earlier group(s)
  if ( ie != 1 ) ili += lastplusznonme42layer; // add offset to -z endcap; ME42 doesn't need this.

  CSCDetId id = detIdFromLayerIndex(ili);

  // For unganged ME1a we need to turn this ME11 detid into an ME1a one
  if ( me1a ) id = CSCDetId( id.endcap(), 1, 4, id.chamber(), id.layer() );

  return std::make_pair(id, ichan);
}


std::pair<CSCDetId, CSCIndexerBase::IndexType>  CSCIndexerPostls1::detIdFromChipIndex( IndexType ici ) const
{
  const LongIndexType lastnonme1a       = 15768; // chips in chambers with ME42 installed
  const LongIndexType lastpluszme1a     = 16416; // last unganged ME1a +z chip = 15768 + 648 = 16416
  const LongIndexType lastnonme42 = 13608; // chips in 2008 installed chambers
  const LongIndexType lastplusznonme42 = 6804; // = 13608/2
  const LongIndexType firstme13  = 2161; // First channel of ME13
  const LongIndexType lastme13   = 3024; // Last channel of ME13

  const IndexType lastnonme42layer = 2808;
  const IndexType lastplusznonme42layer = 1404; // = 2808/2
  const IndexType firstme13layer  = 433; // = 72*6 + 1 (ME13 chambers are 72-108 in range 1-234)
  const IndexType lastme13layer   = 648; // = 108*6

  bool me1a = false;

  // Most chambers (except ME13, ME1a) have 5 chips/layer
  IndexType nchipPerLayer = 5;

  // Set endcap to +z. This should work for ME42 channels too, since we don't need to calculate its endcap explicitly.
  IndexType ie = 1;

  LongIndexType istart = 0;
  IndexType layerOffset = 0;

  if ( ici <= lastnonme42 )
  {
    // Chambers as of 2008 Installation (ME11 keeps the same #of chips 5 allocated for it in the index)
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
  else if ( ici <= lastnonme1a ) // ME42 chambers
  {
    istart = lastnonme42;
    layerOffset = lastnonme42layer;
    // don't care about ie, as ME42 stratch of indices is uniform
  }
  else  // Unganged ME1a channels
  {
    me1a = true;
    if (ici > lastpluszme1a) ie = 2;
    istart = lastnonme1a;
    nchipPerLayer = 3;
    // layerOffset stays 0, as we want to map them onto ME1b's layer indices
  }

  ici -= istart; // remove earlier group(s)
  IndexType ichip = (ici - 1)%nchipPerLayer + 1;
  IndexType ili = (ici - 1)/nchipPerLayer + 1;
  ili += layerOffset; // add appropriate offset for earlier group(s)
  if ( ie != 1 ) ili += lastplusznonme42layer; // add offset to -z endcap; ME42 doesn't need this.

  CSCDetId id = detIdFromLayerIndex(ili);

  // For unganged ME1a we need to turn this ME11 detid into an ME1a one
  if ( me1a ) id = CSCDetId( id.endcap(), 1, 4, id.chamber(), id.layer() );

  return std::make_pair(id, ichip);
}


int CSCIndexerPostls1::dbIndex(const CSCDetId & id, int & channel) const
{
  int ec = id.endcap();
  int st = id.station();
  int rg = id.ring();
  int ch = id.chamber();
  int la = id.layer();

  return ec*100000 + st*10000 + rg*1000 + ch*10 + la;
}


CSCIndexerBase::GasGainIndexType  CSCIndexerPostls1::detIdFromGasGainIndex( IndexType igg ) const
{
  const int n_types = 20;
  const IndexType type_starts[n_types] =
    {1,  1081, 4321, 6913, 8533, 13933, 15553, 20953, 22573, 23653, 26893, 29485, 31105, 36505, 38125, 43525, 45145, 50545, 55945, 56593};
  //+1/1 +1/2  +1/3  +2/1  +2/2  +3/1   +3/2   +4/1   -1/1   -1/2   -1/3   -2/1   -2/2   -3/1   -3/2   -4/1   +4/2   -4/2   +1/4   -1/4

  const int endcaps[n_types] =
    {1,  1,    1,    1,    1,    1,     1,     1,     2,     2,     2,     2,     2,     2,     2,     2,     1,     2,     1,     2};
  const int stations[n_types] =
    {1,  1,    1,    2,    2,    3,     3,     4,     1,     1,     1,     2,     2,     3,     3,     4,     4,     4,     1,     1};
  const int rings[n_types] =
    {1,  2,    3,    1,    2,    1,     2,     1,     1,     2,     3,     1,     2,     1,     2,     1,     2,     2,     4,     4};

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
