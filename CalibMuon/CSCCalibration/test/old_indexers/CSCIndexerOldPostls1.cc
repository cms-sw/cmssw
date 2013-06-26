#include "CSCIndexerOldPostls1.h"
#include <iostream>

void CSCIndexerOldPostls1::fillChamberLabel() const {
  // Fill the member vector which permits decoding of the linear chamber index
  // Logically const since initializes cache only,
  // Beware that the ME42 indices 235-270 within this vector do NOT correspond to
  // their 'real' linear indices (which are 469-504 for +z)
   chamberLabel.resize( 271 ); // one more than #chambers per endcap. Includes ME42.
   IndexType count = 0;
   chamberLabel[count] = 0;

   for ( IndexType is = 1 ; is != 5; ++is ) {
      IndexType irmax = ringsInStation(is);
      for ( IndexType ir = 1; ir != irmax+1; ++ir ) {
         IndexType icmax = chambersInRingOfStation(is, ir);
         for ( IndexType ic = 1; ic != icmax+1; ++ic ) {
	   chamberLabel[ ++count ] = is*1000 + ir*100 + ic ;
         }
      } 
   }
}


CSCIndexerOldPostls1::IndexType CSCIndexerOldPostls1::hvSegmentIndex(IndexType is, IndexType ir, IndexType iwire ) const
{
  IndexType hvSegment = 1;   // There is only one HV segment in ME1/1

  if (is > 2 && ir == 1)
  {         // HV segments are the same in ME3/1 and ME4/1
    if      ( iwire >= 33 && iwire <= 64 ) hvSegment = 2;
    else if ( iwire >= 65 && iwire <= 96 ) hvSegment = 3;
  }
  else if (is > 1 && ir == 2)
  {         // HV segments are the same in ME2/2, ME3/2, and ME4/2
    if      ( iwire >= 17 && iwire <= 28 ) hvSegment = 2;
    else if ( iwire >= 29 && iwire <= 40 ) hvSegment = 3;
    else if ( iwire >= 41 && iwire <= 52 ) hvSegment = 4;
    else if ( iwire >= 53 && iwire <= 64 ) hvSegment = 5;
  }
  else if (is == 1 && ir == 2)
  {
    if      ( iwire >= 25 && iwire <= 48 ) hvSegment = 2;
    else if ( iwire >= 49 && iwire <= 64 ) hvSegment = 3;
  }
  else if (is == 1 && ir == 3)
  {
    if      ( iwire >= 13 && iwire <= 22 ) hvSegment = 2;
    else if ( iwire >= 23 && iwire <= 32 ) hvSegment = 3;
  }
  else if (is == 2 && ir == 1)
  {
    if      ( iwire >= 45 && iwire <= 80 ) hvSegment = 2;
    else if ( iwire >= 81 && iwire <= 112) hvSegment = 3;
  }

  return hvSegment;
}


CSCDetId CSCIndexerOldPostls1::detIdFromChamberIndex_OLD( IndexType ici ) const {

  // Will not work as is for ME42
  // ============================

  IndexType ie = 1;
  if (ici > 234 ) {
     ie = 2;
     ici -= 234; 
  }
  // Now ici is in range 1-234 (assuming valid input in range 1-468)

  // MEij pairs...
  const IndexType station[] = {0,1,1,1,2,2,3,3,4};
  const IndexType ring[]    = {0,1,2,3,1,2,1,2,1};

  // MEij preceding a given MEij matching linear index above
  const IndexType prevs[] =  {0,0,1,1,1,2,2,3,3}; 
  const IndexType prevr[] =  {0,0,1,2,3,1,2,1,2};

  IndexType is = 4;
  IndexType ir = 1;
  for ( IndexType i = 2; i<=8; ++i) {
    IndexType js = station[i];
    IndexType jr = ring[i];
    // if it's before start of MEjs/jr then it's in the previous MEis/ir
      if ( ici < startChamberIndexInEndcap(ie,js,jr) ) {
	is = prevs[i];
	ir = prevr[i];
	break;
      }
      // otherwise it's in ME41
  }
  IndexType ic = ici - startChamberIndexInEndcap(ie,is,ir) + 1;

  return CSCDetId( ie, is, ir, ic );
}
 
CSCDetId CSCIndexerOldPostls1::detIdFromChamberIndex( IndexType ici ) const {
  // Expected range of input range argument is 1-540.
  // 1-468 for CSCs installed at 2008 start-up. 469-540 for ME42.

  IndexType ie = 1;
  if ( ici > 468 ) {
    // ME42
    ici -= 234; // now in range 235-306
    if ( ici > 270 ) { // -z
      ie = 2;
      ici -= 36; // now in range 235-270
    }
  }
  else { // in range 1-468
    if ( ici > 234 ) { // -z
      ie = 2;
      ici -= 234; // now in range 1-234
    }
  }
  if (chamberLabel.empty()) fillChamberLabel();
  IndexType label = chamberLabel[ici];    
  return detIdFromChamberLabel( ie, label );
}

CSCIndexerOldPostls1::IndexType CSCIndexerOldPostls1::chamberLabelFromChamberIndex( IndexType ici ) const {
  // This is just for cross-checking

  // Expected range of input range argument is 1-540.
  // 1-468 for CSCs installed at 2008 start-up. 469-540 for ME42.

  if ( ici > 468 ) {
    // ME42
    ici -= 234; // now in range 235-306
    if ( ici > 270 ) { // -z
      ici -= 36; // now in range 235-270
    }
  }
  else { // in range 1-468
    if ( ici > 234 ) { // -z
      ici -= 234; // now in range 1-234
    }
  }
  if (chamberLabel.empty()) fillChamberLabel();
  return chamberLabel[ici];  

}

CSCDetId CSCIndexerOldPostls1::detIdFromChamberLabel( IndexType ie, IndexType label ) const {

  IndexType is = label/1000;
  label -= is*1000;
  IndexType ir = label/100;
  label -= ir*100;
  IndexType ic = label;

  return CSCDetId( ie, is, ir, ic );
}

CSCDetId CSCIndexerOldPostls1::detIdFromLayerIndex( IndexType ili ) const {

  IndexType il = (ili-1)%6 + 1;
  IndexType ici = (ili-1)/6 + 1;
  CSCDetId id = detIdFromChamberIndex( ici ); 

  return CSCDetId(id.endcap(), id.station(), id.ring(), id.chamber(), il);
}

std::pair<CSCDetId, CSCIndexerOldPostls1::IndexType>  CSCIndexerOldPostls1::detIdFromStripChannelIndex( LongIndexType isi ) const {

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

  if ( isi <= lastnonme42 ) {
    // Chambers as of 2008 Installation (ME11 keeps the same #of channels 80 allocated for it in the index)

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
  else if ( isi <= lastnonme1a ) { // ME42 chambers

    istart = lastnonme42;
    layerOffset = lastnonme42layer;

    // don't care about ie, as ME42 stratch of indices is uniform
  }
  else {   // Unganged ME1a channels
    
    me1a = true;
    if (isi > lastpluszme1a) ie = 2;
    istart = lastnonme1a; 
    nchan = 48;
    // layerOffset stays 0, as we want to map them onto ME1b's layer indices
  }

  isi -= istart; // remove earlier group(s)
  IndexType ichan = (isi-1)%nchan + 1;
  IndexType ili = (isi-1)/nchan + 1;
  ili += layerOffset; // add appropriate offset for earlier group(s)
  if ( ie != 1 ) ili+= lastplusznonme42layer; // add offset to -z endcap; ME42 doesn't need this.
	
  CSCDetId id = detIdFromLayerIndex(ili);

  // For unganged ME1a we need to turn this ME11 detid into an ME1a one
  if ( me1a ) id = CSCDetId( id.endcap(), 1, 4, id.chamber(), id.layer() );
	
  return std::make_pair(id, ichan);
}


std::pair<CSCDetId, CSCIndexerOldPostls1::IndexType>  CSCIndexerOldPostls1::detIdFromChipIndex( IndexType ici ) const {

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

  if ( ici <= lastnonme42 ) {	
    // Chambers as of 2008 Installation (ME11 keeps the same #of chips 5 allocated for it in the index)

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
  else if ( ici <= lastnonme1a ) {  // ME42 chambers

    istart = lastnonme42;
    layerOffset = lastnonme42layer;
    
    // don't care about ie, as ME42 stratch of indices is uniform
  }
  else {   // Unganged ME1a channels
    
    me1a = true;
    if (ici > lastpluszme1a) ie = 2;
    istart = lastnonme1a; 
    nchipPerLayer = 3;
    // layerOffset stays 0, as we want to map them onto ME1b's layer indices
  }

   ici -= istart; // remove earlier group(s)
   IndexType ichip = (ici-1)%nchipPerLayer + 1;
   IndexType ili = (ici-1)/nchipPerLayer + 1;
   ili += layerOffset; // add appropriate offset for earlier group(s)
   if ( ie != 1 ) ili+= lastplusznonme42layer; // add offset to -z endcap; ME42 doesn't need this.
	
   CSCDetId id = detIdFromLayerIndex(ili);
   
   // For unganged ME1a we need to turn this ME11 detid into an ME1a one
   if ( me1a ) id = CSCDetId( id.endcap(), 1, 4, id.chamber(), id.layer() );
	
   return std::make_pair(id, ichip);
}


CSCIndexerOldPostls1::GasGainTuple  CSCIndexerOldPostls1::detIdFromGasGainIndex( IndexType igg ) const
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


int CSCIndexerOldPostls1::dbIndex(const CSCDetId & id, int & channel)
{
  int ec = id.endcap();
  int st = id.station();
  int rg = id.ring();
  int ch = id.chamber();
  int la = id.layer();

  return ec*100000 + st*10000 + rg*1000 + ch*10 + la;
}
