#ifndef MuonDetId_CSCIndexerOldPostls1_h
#define MuonDetId_CSCIndexerOldPostls1_h

/** \class CSCIndexerOldPostls1
 * Creates a linear index for various sublevels of the endcap muon CSC system.
 *
 * It supplies a linear index for:
 * 1. Each chamber of the CSC system: range 1-468 (CSC system as installed 2008) 469-540 for ME42 <br>
 * 2. Each layer of the CSC system: range 1-2808 (CSCs 2008) 2809-3240 for ME42 <br>
 * 3. Each strip channel of the CSC system: range 1-217728 (CSCs 2008) 217729-252288 for ME42 <br>
 * 4. Each Buckeye chip (1 for each layer in each CFEB)of the CSC system: range 1-13608 (CSCs 2008) 13609-15768 for ME42 <br>
 * 5. Each Gas Gain Sector (1 for each [CFEB*HV segment] combination in each layer): range 1-45144 (CSCs 2008) 45145-55944 for ME42 <br>
 * 6. Extended for unganged ME1a, appended to the standard list: only in the strip list
 *    252289-273024 (i.e. an extra 2 x 36 x 6 x 48 = 20736 channels)
 *
 * The chamber and layer may be specified by CSCDetId or labels for endcap, station, ring, chamber, layer.
 * The strip channel is a value 1-80 (or 64: ME13 chambers have only 64 channels.)
 * The chip number is a value 1-30 (or 24: ME13 chambers have only 24 chips.)
 *
 * The main user interface is the set of functions <br>
 *   chamberIndex(.) <br>
 *   layerIndex(.) <br>
 *   stripChannelIndex(.) <br>
 *   chipIndex(.) <br>
 * But the other functions are public since they may be useful in contexts other than for
 * Conditions Data for which the above functions are intended.
 *
 * \warning This class is hard-wired for the CSC system at start-up of CMS in 2008.
 * with rings ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41 totalling 234 chambers per endcap.
 * But ME42 is appended (to permit simulation studies), so the chamber order is <br>
 * +z ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, <br>
 * -z ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, <br>
 * +z ME42, -z ME42 <br>
 *
 * Further it is extended for hypothetical unganged ME1a strip channels.
 * The strip list then appends
 * +z ME1a, -z ME1a <br>
 *
 * \warning This intentionally has "magic numbers galore" which, supposedly, is for better performance.
 *
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <iosfwd>
#include <utility>  // for pair
#include <boost/tuple/tuple.hpp>

class CSCIndexerOldPostls1 {

public:

  typedef uint16_t IndexType;
  typedef uint32_t LongIndexType;

  typedef boost::tuple<CSCDetId,  // id
                       IndexType, // HV segment
                       IndexType  // chip
                      > GasGainTuple;

  static const IndexType MAX_CHAMBER_INDEX = 540;
  static const IndexType MAX_LAYER_INDEX = 3240;
  static const LongIndexType MAX_STRIP_CHANNEL_INDEX = 273024;
  static const IndexType MAX_CHIP_INDEX = 17064;
  static const IndexType MAX_GAS_GAIN_INDEX = 57240;

  CSCIndexerOldPostls1(){};
  ~CSCIndexerOldPostls1(){};

  /**
   * Starting index for first chamber in ring 'ir' of station 'is' in endcap 'ie',
   * in range 1-468 (CSCs 2008) or 469-540 (ME42).
   *
   * WARNING: Considers both ME1a and ME1b to be part of one ME11 chamber
   *          (result would be the same for is=1 and ir=1 or 4).
   */
  IndexType startChamberIndexInEndcap(IndexType ie, IndexType is, IndexType ir) const {
    const IndexType nschin[32] = { 1,37,73,1,        109,127,0,0,  163,181,0,0,  217,469,0,0,
                                   235,271,307,235,  343,361,0,0,  397,415,0,0,  451,505,0,0 };
    return nschin[(ie-1)*16 + (is-1)*4 + ir - 1];
  }

  /**
   * Linear index for chamber 'ic' in ring 'ir' of station 'is' in endcap 'ie',
   * in range 1-468 (CSCs 2008) or 469-540 (ME42)
   *
   * WARNING: Considers both ME1a and ME1b to be part of one ME11 chamber
   *          (result would be the same for is=1 and ir=1 or 4).
   */
  IndexType chamberIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic) const {
    return startChamberIndexInEndcap(ie,is,ir) + ic - 1; // -1 so start index _is_ ic=1
  }

  /**
   * Linear index to label each CSC in CSC system.
   * Argument is the CSCDetId of some CSCChamber.
   *
   * Output is 1-468 (CSCs 2008) 469-540 (with ME42)
   *
   * WARNING: Considers both ME1a and ME1b to be part of one ME11 chamber
   *          (result would be the same for is=1 and ir=1 or 4).
   */
  IndexType chamberIndex( const CSCDetId& id ) const {
    return chamberIndex( id.endcap(), id.station(), id.ring(), id.chamber() );
  }

  /**
   * Linear index to label each hardware layer in CSC system.
   * Argument is the CSCDetId of some CSCLayer.
   *
   * Output is 1-2808 (CSCs 2008) 2809-3240 (ME42)
   *
   * WARNING: Considers both ME1a and ME1b to share layers of single ME11 chamber
   *          (result would be the same for is=1 and ir=1 or 4).
   */
  IndexType layerIndex( const CSCDetId& id ) const {
    return layerIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer());
  }

  /**
   * Linear index for layer 'il' of chamber 'ic' in ring 'ir' of station 'is' in endcap 'ie',
   * in range 1-2808 (CSCs 2008) or 2809-3240 (ME42).
   *
   * WARNING: Considers both ME1a and ME1b to share layers of single ME11 chamber
   *          (result would be the same for is=1 and ir=1 or 4).
   */
  IndexType layerIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic, IndexType il) const {
    return (chamberIndex(ie,is,ir,ic) - 1 ) * 6 + il;
  }

  /**
   * How many (physical hardware) rings are there in station is=1, 2, 3, 4 ?
   *
   * WARNING:
   * - ME1 has only 3 physical rings (the hardware) not 4 (virtual ring 4 is used offline for ME1a).
   * - Includes ME42 so claims 2 rings in station 4.
   */
  IndexType ringsInStation( IndexType is ) const {
    const IndexType nrings[5] = { 0,3,2,2,2 }; // rings per station
    return nrings[is];
  }

  /**
   * How many offline rings are there in station is=1, 2, 3, 4 ?
   *
   * WARNING:
   * - ME1 has 4 rings in the offline notation (virtual ring 4 is used for ME1a).
   * - Includes ME42 so claims 2 rings in station 4.
   */
  IndexType offlineRingsInStation( IndexType is ) const {
    const IndexType nrings[5] = { 0,4,2,2,2 }; // rings per station
    return nrings[is];
  }

  /**
   * How many chambers are there in ring ir of station is?
   *
   * Works for ME1a (virtual ring 4 of ME1) too.
   */
  IndexType chambersInRingOfStation(IndexType is, IndexType ir) const {
    const IndexType nCinR[16] = { 36,36,36,36,  18,36,0,0,  18,36,0,0,  18,36,0,0 }; // chambers in ring
    return nCinR[(is-1)*4 + ir - 1];
  }

  /** 
   * Number of strip channels per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-4.
   * Works for ME1a input as is=1, ir=4.
   *
   * WARNING:
   * - ME1a channels are considered to be unganged i.e. 48 not the standard 16.
   * - ME1b keeps 80 channels for the indexing purposes, however channels 65-80 are ignored in the unganged case.
   */
  IndexType stripChannelsPerLayer( IndexType is, IndexType ir ) const {
    const IndexType nSCinC[16] = { 80,80,64,48,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSCinC[(is-1)*4 + ir - 1];
  }

  /**
   * Linear index for 1st strip channel in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-3.
   *
   * WARNING: ME1a channels are  NOT  considered the last 16 of the 80 total in each layer of an ME11 chamber!
   */
  LongIndexType stripChannelStart( IndexType ie, IndexType is, IndexType ir ) const {
    // These are in the ranges 1-217728 (CSCs 2008), 217729-252288 (ME42), and 252289-273024 (unganged ME1a)
    // There are 1-108884 channels per endcap (CSCs 2008), 17280 channels per endcap (ME42),
    // and 10368 channels per endcap (unganged ME1a)
    // Start of -z channels (CSCs 2008) is 108864 + 1    = 108865
    // Start of +z (ME42) is 217728 + 1                  = 217729
    // Start of -z (ME42) is 217728 + 1 + 17280          = 235009
    // Start of +z (unganged ME1a) is 252288 + 1         = 252289
    // Start of -z (unganged ME1a) is 252288 + 1 + 10368 = 262657
    const LongIndexType nStart[32] =
      {      1, 17281, 34561,252289,   48385, 57025,0,0,   74305, 82945,0,0,  100225,217729,0,0,
        108865,126145,143425,262657,  157249,165889,0,0,  183169,191809,0,0,  209089,235009,0,0 };
    return  nStart[(ie-1)*16 + (is-1)*4 + ir - 1];
  }

  /**
   * Linear index for strip channel istrip in layer 'il' of chamber 'ic' of ring 'ir'
   * in station 'is' of endcap 'ie'.
   *
   * Output is 1-217728 (CSCs 2008) or 217729-252288 (ME42) or 252289-273024 (unganged ME1a)
   * WARNING: Supplying ME1a values returns index for unganged ME1a channels, not for channels 65-80 of ME11.
   *
   * WARNING: Use at your own risk! You must input labels within hardware ranges.
   * No trapping on out-of-range values!
   */
  LongIndexType stripChannelIndex( IndexType ie, IndexType is, IndexType ir, IndexType ic, IndexType il, IndexType istrip ) const {
    return stripChannelStart(ie,is,ir)+( (ic-1)*6 + il - 1 )*stripChannelsPerLayer(is,ir) + (istrip-1);
  }

  /**
   * Linear index for strip channel 'istrip' in layer labelled by CSCDetId 'id'.
   *
   * Output is 1-217728 (CSCs 2008) or 217729-252288 (ME42) or 252289-273024 (unganged ME1a)
   *
   * WARNING: Use at your own risk! The supplied CSCDetId must be a layer id.
   * No trapping on out-of-range values!
   */
  LongIndexType stripChannelIndex( const CSCDetId& id, IndexType istrip ) const {
    return stripChannelIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer(), istrip );
  }

  /** 
   * Number of Buckeye chips per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-3.
   *
   * Works for ME1a input as is=1, ir=4
   * Considers ME42 as standard 5 chip per layer chambers.
   *
   * WARNING:
   * - ME1a channels are considered to be unganged and have their own 3 chips (ME1b has 4 chips).
   * - ME1b keeps 5 chips for the indexing purposes, however indices for the chip #5 are ignored in the unganged case.
   */
  IndexType chipsPerLayer( IndexType is, IndexType ir ) const {
    const IndexType nCinL[16] = { 5,5,4,3,  5,5,0,0,  5,5,0,0,  5,5,0,0 };
    return nCinL[(is-1)*4 + ir - 1];
  }

  /**
   * Linear index for 1st Buckey chip in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-3.
   * 
   * Works for ME1a input as is=1, ir=4
   * WARNING: ME1a channels are the last 3 of the 7 chips total in each layer of an ME11 chamber, 
   */
   IndexType chipStart( IndexType ie, IndexType is, IndexType ir ) const {

     // These are in the ranges 1-13608 (CSCs 2008) and 13609-15768 (ME42) and 15769-17064 (ME1a).
     // There are 1-6804 chips per endcap (CSCs 2008) and 1080 chips per endcap (ME42) and 648 chips per endcap (ME1a).
     // Start of -z channels (CSCs 2008) is 6804 + 1 = 6805
     // Start of +z (ME42) is 13608 + 1 = 13609
     // Start of -z (ME42) is 13608 + 1 + 1080 = 14689
     // Start of +z (ME1a) is 15768 + 1 = 15769
     // Start of -z (ME1a) is 15768 + 1 + 648 = 16417
     const IndexType nStart[32] = {1,   1081, 2161, 15769,  3025, 3565, 0,0, 4645, 5185, 0,0, 6265, 13609,0,0,
				   6805,7885, 8965, 16417,  9829, 10369,0,0, 11449,11989,0,0, 13069,14689,0,0 };
     return  nStart[(ie-1)*16 + (is-1)*4 + ir - 1];
   }

  /**
   * Linear index for Buckeye chip 'ichip' in layer 'il' of chamber 'ic' of ring 'ir'
   * in station 'is' of endcap 'ie'.
   *
   * Output is 1-13608 (CSCs 2008) or 13609-15768 (ME42) or 15769-17064 (ME1a).
   *
   * WARNING: Use at your own risk! You must input labels within hardware ranges.
   * No trapping on out-of-range values!
   */
   IndexType chipIndex( IndexType ie, IndexType is, IndexType ir, IndexType ic, IndexType il, IndexType ichip ) const {
     //printf("ME%d/%d/%d/%d layer %d  chip %d chipindex %d\n",ie,is,ir,ic,il,ichip,chipStart(ie,is,ir)+( (ic-1)*6 + il - 1 )*chipsPerLayer(is,ir) + (ichip-1));
     return chipStart(ie,is,ir)+( (ic-1)*6 + il - 1 )*chipsPerLayer(is,ir) + (ichip-1);

  }

  /**
   * Linear index for Buckeye chip 'ichip' in layer labelled by CSCDetId 'id'.
   *
   * Output is 1-13608 (CSCs 2008) or 13609-15768 (ME42) or 15769-17064 (ME1a).
   *
   * WARNING: Use at your own risk! The supplied CSCDetId must be a layer id.
   * No trapping on out-of-range values!
   */
   IndexType chipIndex( const CSCDetId& id, IndexType ichip ) const {
      return chipIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer(), ichip);
   }

  /**
   * Linear index inside of a chamber for Buckeye chip processing strip 'istrip'.
   *
   * Output is 1-5.
   *
   * WARNING: Use at your own risk! The supplied CSCDetId must be a strip id 1-80
   * ME1/1b strips must be 1-64
   * ME1/1a strips must be 1-48
   * No trapping on out-of-range values!
   */

   IndexType chipIndex( IndexType istrip ) const {
     return (istrip-1)/16 + 1;
   }

  /** 
   * Number of HV segments per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-4.
   *
   * WARNING: ME1a (ir=4), ME1b(ir=1) and whole ME11 are all assumed to have the same single HV segment
   */
  IndexType hvSegmentsPerLayer( IndexType is, IndexType ir ) const {
    const IndexType nSinL[16] = { 1,3,3,1, 3,5,0,0, 3,5,0,0, 3,5,0,0 };
    return nSinL[(is-1)*4 + ir - 1];
  }

  /** 
   * Number of Gas Gain sectors per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-4.
   */
  IndexType sectorsPerLayer( IndexType is, IndexType ir ) const {
    return chipsPerLayer(is,ir)*hvSegmentsPerLayer(is,ir);
  }

  /**
   * Linear index inside of a chamber for HV segment
   *
   * Output is 1-5.
   *
   * WARNING: Use at your own risk! The supplied CSCDetId must be chamber station, ring, and wire.
   * No trapping on out-of-range values!
   */
  IndexType hvSegmentIndex(IndexType is, IndexType ir, IndexType iwire ) const;

  /**
   * Linear index for 1st Gas gain sector in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-3.
   *
   * WARNING:
   *   current: ME1a channels are the last 1 of the 5 chips total in each layer of an ME11 chamber, 
   *            and an input ir=4 is invalid and will give nonsense.
   *   upgraded: 
   */
  IndexType sectorStart( IndexType ie, IndexType is, IndexType ir ) const {
    // There are 36 chambers * 6 layers * 5 CFEB's * 1 HV segment = 1080 gas-gain sectors in ME1/1 (non-upgraded)
    // There are 36 chambers * 6 layers * 3 CFEB's * 1 HV segment = 648 gas-gain sectors in ME1/1a (upgraded)
    // There are 36*6*5*3 = 3240 gas-gain sectors in ME1/2
    // There are 36*6*4*3 = 2592 gas-gain sectors in ME1/3
    // There are 18*6*5*3 = 1620 gas-gain sectors in ME[2-4]/1
    // There are 36*6*5*5 = 5400 gas-gain sectors in ME[2-4]/2
    // Start of -z channels (CSCs 2008) is 22572 + 1 = 22573
    // Start of +z (ME42) is 45144 + 1 = 45145
    // Start of -z (ME42) is 45144 + 1 + 5400 = 50545
    // Start of +z (ME1a) is 45144 + 1 + 2*5400 = 55945
    // Start of -z (ME42) is 45144 + 1 + 2*5400 + 648 = 56593
    const IndexType nStart[32] = {1    , 1081 ,  4321, 55945,  //ME+1/1,ME+1/2,ME+1/3,ME+1/a
				  6913 , 8533 ,     0,     0,  //ME+2/1,ME+2/2
				  13933, 15553,     0,     0,  //ME+3/1,ME+3/2
				  20953, 45145,     0,     0,  //ME+4/1,ME+4/2 (note, ME+4/2 index follows ME-4/1...)
				  22573, 23653, 26893, 56593,  //ME-1/1,ME-1/2,ME-1/3,ME+1/a
				  29485, 31105,     0,     0,  //ME-2/1,ME-2/2
				  36505, 38125,     0,     0,  //ME-3/1,ME-3/2
				  43525, 50545,     0,     0 };//ME-4/1,ME-4/2 (note, ME-4/2 index follows ME+4/2...)
    return  nStart[(ie-1)*16 + (is-1)*4 + ir - 1];
  }

  /**
   * Linear index for Gas gain sector, based on CSCDetId 'id', cathode strip 'istrip' and anode wire 'iwire' 
   *
   * Output is 1-45144 (CSCs 2008) and 45145-55944 (ME42) and 55945-57240 (ME1a)
   *
   * WARNING: Use at your own risk!  You must input labels within hardware ranges (e.g., 'id' must correspond 
   * to a specific layer 1-6). No trapping on out-of-range values!
   */
  IndexType gasGainIndex( const CSCDetId& id, IndexType istrip, IndexType iwire ) const {
    return gasGainIndex( id.endcap(), id.station(), id.ring(), id.chamber(), id.layer(),
                         hvSegmentIndex(id.station(), id.ring(), iwire), chipIndex(istrip) );
  }

  /**
   * Linear index for Gas gain sector, based on CSCDetId 'id', the HV segment# and the chip#.
   * Note: to allow method overloading, the parameters order is reversed comparing to the (id,strip,wire) method
   *
   * Output is 1-45144 (CSCs 2008) and 45145-55944 (ME42) and 55945-57240 (ME1a)
   *
   * WARNING: Use at your own risk! You must input labels within hardware ranges.
   * No trapping on out-of-range values!
   */
  IndexType gasGainIndex( IndexType ihvsegment, IndexType ichip, const CSCDetId& id ) const {
    return gasGainIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer(), ihvsegment, ichip);
  }

  /**
   * Linear index for Gas gain sector, based on the HV segment# and the chip#
   * located in layer 'il' of chamber 'ic' of ring 'ir' in station 'is' of endcap 'ie'.
   *
   * Output is 1-45144 (CSCs 2008) and 45145-55944 (ME42) and 55945-57240 (ME1a)
   *
   * WARNING: Use at your own risk! You must input labels within hardware ranges.
   * No trapping on out-of-range values!
   */
  IndexType gasGainIndex( IndexType ie, IndexType is, IndexType ir, IndexType ic, IndexType il, IndexType ihvsegment, IndexType ichip ) const {
    return sectorStart(ie,is,ir)+( (ic-1)*6 + il - 1 )*sectorsPerLayer(is,ir) + (ihvsegment-1)*chipsPerLayer(is,ir) + (ichip-1);
  }

  /**
   *  Decode CSCDetId from various indexes and labels
   */
  CSCDetId detIdFromLayerIndex( IndexType ili ) const;
  CSCDetId detIdFromChamberIndex( IndexType ici ) const;
  CSCDetId detIdFromChamberIndex_OLD( IndexType ici ) const;
  CSCDetId detIdFromChamberLabel( IndexType ie, IndexType icl ) const;
  std::pair<CSCDetId, IndexType> detIdFromStripChannelIndex( LongIndexType ichi ) const;
  std::pair<CSCDetId, IndexType> detIdFromChipIndex( IndexType ichi ) const;
  GasGainTuple detIdFromGasGainIndex( IndexType igg ) const;

  IndexType chamberLabelFromChamberIndex( IndexType ) const; // just for cross-checks

  /**
   * Build index used internally in online CSC conditions databases (the 'Igor Index')
   *
   * This is the decimal integer ie*100000 + is*10000 + ir*1000 + ic*10 + il <br>
   * (ie=1-2, is=1-4, ir=1-4, ic=1-36, il=1-6) <br>
   * Channels 1-16 in ME1A (is=1, ir=4) are NOT reset to channels 65-80 of ME11.
   * WARNING: This is now ADOPTED to unganged ME1a channels
   *          (we expect that the online conditions DB will adopt it too).
   */
  int dbIndex(const CSCDetId & id, int & channel);

private:

  void fillChamberLabel() const; // const so it can be called in const function detIdFromChamberIndex

  mutable std::vector<IndexType> chamberLabel; // mutable so can be filled by fillChamberLabel

};

#endif
