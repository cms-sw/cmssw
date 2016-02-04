#ifndef MuonDetId_CSCIndexer_h
#define MuonDetId_CSCIndexer_h

/** \class CSCIndexer
 * Creates a linear index for various sublevels of the endcap muon CSC system.
 *
 * It supplies a linear index for:
 * 1. Each chamber of the CSC system: range 1-468 (CSC system as installed 2008) 469-540 for ME42 <br>
 * 2. Each layer of the CSC system: range 1-2808 (CSCs 2008) 2809-3240 for ME42 <br>
 * 3. Each strip channel of the CSC system: range 1-217728 (CSCs 2008) 217729-252288 for ME42 <br>
 * 4. Each Buckeye chip (1 for each layer in each CFEB)of the CSC system: range 1-13608 (CSCs 2008) 13609-15768 for ME42 <br>
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
 * \warning This uses magic numbers galore!!
 *
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <iosfwd>
#include <utility>  // for pair

class CSCIndexer {

public:

  //  typedef unsigned short int IndexType;
  //  typedef unsigned       int LongIndexType;
  typedef uint16_t IndexType;
  typedef uint32_t LongIndexType;

  CSCIndexer(){};
  ~CSCIndexer(){};

  /**
   * Linear index to label each CSC in CSC system.
   * Argument is the CSCDetId of some CSCChamber.
   *
   * Output is 1-468 (CSCs 2008) 469-540 (ME42)
   *
   * WARNING: Do not input ME1a values (i.e. ring '4'): does not consider ME1a and ME1b to be separate,
   * No sanity checking on input value: if you supply an ME1a CSCDetId then you'll get nonsense.
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
   * WARNING: Do not input ME1a values (i.e. ring '4'): does not consider ME1a and ME1b to be separate,
   * No sanity checking on input value: if you supply an ME1a CSCDetId then you'll get nonsense.
   */

   IndexType layerIndex( const CSCDetId& id ) const {
     return layerIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer());
   }

   /** 
    * Starting index for first chamber in ring 'ir' of station 'is' in endcap 'ie',
    * in range 1-468 (CSCs 2008) or 469-540 (ME42).
    */
   IndexType startChamberIndexInEndcap(IndexType ie, IndexType is, IndexType ir) const {
     const IndexType nschin[24] = {1,37,73,    109,127,0, 163,181,0, 217,469,0,
                                  235,271,307, 343,361,0, 397,415,0, 451,505,0 };
     return nschin[(ie-1)*12 + (is-1)*3 + ir-1];

   }

   /**
    * Linear index for chamber 'ic' in ring 'ir' of station 'is' in endcap 'ie',
    * in range 1-468 (CSCs 2008) or 469-540 (ME42)
    */
   IndexType chamberIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic) const {
     return startChamberIndexInEndcap(ie,is,ir) + ic - 1; // -1 so start index _is_ ic=1
   }

   /**
    * Linear index for layer 'il' of chamber 'ic' in ring 'ir' of station 'is' in endcap 'ie',
    * in range 1-2808 (CSCs 2008) or 2809-3240 (ME42).
    */
   IndexType layerIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic, IndexType il) const {
    const IndexType layersInChamber = 6;
     return (chamberIndex(ie,is,ir,ic) - 1 ) * layersInChamber + il;
   }

  /**
   * How many rings are there in station is=1, 2, 3, 4 ?
   *
   * BEWARE! Includes ME42 so claims 2 rings in station 4. There is only 1 at CSC installation 2008.
   */
   IndexType ringsInStation( IndexType is ) const {
      const IndexType nrins[5] = {0,3,2,2,2}; // rings per station
      return nrins[is];
   }

  /**
   * How many chambers are there in ring ir of station is?
   *
   * Works for ME1a (ring 4 of ME1) too.
   */
   IndexType chambersInRingOfStation(IndexType is, IndexType ir) const {
      IndexType nc = 36; // most rings have 36 chambers
      if (is >1 && ir<2 ) nc = 18; // but 21, 31, 41 have 18
      return nc;
   }

  /** 
   * Number of strip channels per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-3.
   *
   * WARNING: ME1a channels are the last 16 of the 80 total in each layer of an ME11 chamber, 
   * and an input ir=4 is invalid and will give nonsense. <br>
   * Considers ME42 as standard 80-strip per layer chambers.
   */
   IndexType stripChannelsPerLayer( IndexType is, IndexType ir ) const {
     const IndexType nSCinC[12] = { 80,80,64, 80,80,0, 80,80,0, 80,80,0 };
     return nSCinC[(is-1)*3 + ir - 1];
   }

  /**
   * Linear index for 1st strip channel in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-3.
   *
   * WARNING: ME1a channels are the last 16 of the 80 total in each layer of an ME11 chamber, 
   * and an input ir=4 is invalid and will give nonsense.
   */
   LongIndexType stripChannelStart( IndexType ie, IndexType is, IndexType ir ) const {

     // These are in the ranges 1-217728 (CSCs 2008) and 217729-252288 (ME42).
     // There are 1-108884 channels per endcap (CSCs 2008) and 17280 channels per endcap (ME42).
     // Start of -z channels (CSCs 2008) is 108864 + 1 = 108865
     // Start of +z (ME42) is 217728 + 1 = 217729
     // Start of -z (ME42) is 217728 + 1 + 17280 = 235009
      const LongIndexType nStart[24] = { 1,17281,34561, 48385,57025,0, 74305,82945,0, 100225,217729,0,
                                   108865,126145,143425, 157249,165889,0, 183169,191809,0, 209089,235009,0 };
      return  nStart[(ie-1)*12 + (is-1)*3 + ir - 1];
   }

  /**
   * Linear index for strip channel istrip in layer 'il' of chamber 'ic' of ring 'ir'
   * in station 'is' of endcap 'ie'.
   *
   * Output is 1-217728 (CSCs 2008) or 217729-252288 (ME42).
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
   * Output is 1-217728 (CSCs 2008) or 217729-252288 (ME42).
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
   * WARNING: ME1a channels are the last 1 of the 5 total in each layer of an ME11 chamber, 
   * and an input ir=4 is invalid and will give nonsense. <br>
   * Considers ME42 as standard 5 chip per layer chambers.
   */
   IndexType chipsPerLayer( IndexType is, IndexType ir ) const {
     const IndexType nCinL[12] = { 5,5,4, 5,5,0, 5,5,0, 5,5,0 };
     return nCinL[(is-1)*3 + ir - 1];
   }

  /**
   * Linear index for 1st Buckey chip in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-3.
   *
   * WARNING: ME1a channels are the last 1 of the 5 chips total in each layer of an ME11 chamber, 
   * and an input ir=4 is invalid and will give nonsense.
   */
   IndexType chipStart( IndexType ie, IndexType is, IndexType ir ) const {

     // These are in the ranges 1-13608 (CSCs 2008) and 13609-15768 (ME42).
     // There are 1-6804 chips per endcap (CSCs 2008) and 1080 channels per endcap (ME42).
     // Start of -z channels (CSCs 2008) is 6804 + 1 = 6805
     // Start of +z (ME42) is 13608 + 1 = 13609
     // Start of -z (ME42) is 13608 + 1 + 1080 = 14689
     const IndexType nStart[24] = {1, 1081, 2161, 3025, 3565, 0, 4645, 5185, 0, 6265, 13609,0,
				    6805, 7885, 8965, 9829, 10369,0, 11449, 11989, 0, 13069, 14689 ,0 };
                                   
     return  nStart[(ie-1)*12 + (is-1)*3 + ir - 1];
   }

  /**
   * Linear index for Buckeye chip 'ichip' in layer 'il' of chamber 'ic' of ring 'ir'
   * in station 'is' of endcap 'ie'.
   *
   * Output is 1-13608 (CSCs 2008) or 13609-15768 (ME42).
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
   * Output is 1-13608 (CSCs 2008) or 13609-15768 (ME42).
   *
   * WARNING: Use at your own risk! The supplied CSCDetId must be a layer id.
   * No trapping on out-of-range values!
   */

   IndexType chipIndex( const CSCDetId& id, IndexType ichip ) const {
      return chipIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer(), ichip );
   }

  /**
   * Linear index for Buckeye chip processing strip 'istrip'.
   *
   * Output is 1-5.
   *
   * WARNING: Use at your own risk! The supplied CSCDetId must be a strip id 1-80
   * ME1/1a strips must be 65-80
   * No trapping on out-of-range values!
   */

   IndexType chipIndex( IndexType istrip ) const {
     return (istrip-1)/16+1;
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

  IndexType chamberLabelFromChamberIndex( IndexType ) const; // just for cross-checks

  /**
   * Build index used internally in online CSC conditions databases (the 'Igor Index')
   *
   * This is the decimal integer ie*100000 + is*10000 + ir*1000 + ic*10 + il <br>
   * (ie=1-2, is=1-4, ir=1-4, ic=1-36, il=1-6) <br>
   * Channels 1-16 in ME1A (is=1, ir=4) are reset to channels 65-80 of ME11.
   */
  int dbIndex(const CSCDetId & id, int & channel);

private:
  void fillChamberLabel() const; // const so it can be called in const function detIdFromChamberIndex

  mutable std::vector<IndexType> chamberLabel; // mutable so can be filled by fillChamberLabel


};

#endif






