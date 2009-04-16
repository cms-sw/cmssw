#ifndef MuonDetId_CSCIndexer_h
#define MuonDetId_CSCIndexer_h

/** \class CSCIndexer
 * Creates a linear index for various sublevels of the endcap muon CSC system.
 *
 * It supplies a linear index for:
 * 1. Each chamber of the CSC system: range 1-468 (CSC system as installed 2008) 469-540 for ME42 <br>
 * 2. Each layer of the CSC system: range 1-2808 (CSCs 2008) 2809-3240 for ME42 <br>
 * 3. Each strip channel of the CSC system: range 1-217728 (CSCs 2008) 217729-252288 for ME42 <br>
 * 4. Extended for hypothetical unganged ME1a, appended to the standard list: only in the strip list
 *    252289-273024 (i.e. an extra 2 x 36 x 6 x 48 = 20736 channels)
 *
 * The chamber and layer may be specified by CSCDetId or labels for endcap, station, ring, chamber, layer.
 * The strip channel is a value 1-80 (or 64: ME13 chambers have only 64 channels.)
 *
 * The main user interface is the set of functions <br>
 *   chamberIndex(.) <br>
 *   layerIndex(.) <br>
 *   stripChannelIndex(.) <br>
 * But the other functions are public since they may be useful in contexts other than for
 * Conditions Data for which the above functions are intended.
 *
 * \warning This class is hard-wired for the CSC system at start-up of LHC, extended for hypothetical
 * unganged ME1a strip channels.
 * Rings ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41 totalling 234 chambers per endcap.
 * But ME42 is appended (to permit simulation studies), so the chamber order is <br>
 * +z ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, <br>
 * -z ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, <br>
 * +z ME42, -z ME42 <br>
 * The strip list then appends
 * +z ME1a, -z ME1a <br>
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
   * WARNING: Considers both ME1a and ME1b to be part of ME11 - the same values result for is=1, ir=1 or 4.
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
   * WARNING: Considers both ME1a and ME1b to be part of ME11 - the same values result for is=1, ir=1 or 4.
   */

   IndexType layerIndex( const CSCDetId& id ) const {
     return layerIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer());
   }

   /** 
    * Starting index for first chamber in ring 'ir' of station 'is' in endcap 'ie',
    * in range 1-468 (CSCs 2008) or 469-540 (ME42).
    *
    * WARNING: Considers both ME1a and ME1b to be part of ME11 - the same values result for is=1, ir=1 or 4.
    */
   IndexType startChamberIndexInEndcap(IndexType ie, IndexType is, IndexType ir) const {
     const IndexType nschin[32] = {1,37,73,1,    109,127,0,0, 163,181,0,0, 217,469,0,0,
        		        235,271,307,235, 343,361,0,0, 397,415,0,0, 451,505,0,0 };
     return nschin[(ie-1)*16 + (is-1)*4 + ir - 1];

   }

   /**
    * Linear index for chamber 'ic' in ring 'ir' of station 'is' in endcap 'ie',
    * in range 1-468 (CSCs 2008) or 469-540 (ME42)
    *
    * WARNING: Considers both ME1a and ME1b to be part of ME11 - the same values result for is=1, ir=1 or 4.
    */
   IndexType chamberIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic) const {
     return startChamberIndexInEndcap(ie,is,ir) + ic - 1; // -1 so start index _is_ ic=1
   }

   /**
    * Linear index for layer 'il' of chamber 'ic' in ring 'ir' of station 'is' in endcap 'ie',
    * in range 1-2808 (CSCs 2008) or 2809-3240 (ME42).
    *
    * WARNING: Considers both ME1a and ME1b to be part of ME11 - the same values result for is=1, ir=1 or 4.
    */
   IndexType layerIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic, IndexType il) const {
     const IndexType layersInChamber = 6;
     return (chamberIndex(ie,is,ir,ic) - 1 ) * layersInChamber + il;
   }

  /**
   * How many (hardware) rings are there in station is=1, 2, 3, 4 ?
   *
   * WARNING <BR>
   * - Includes ME42 so claims 2 rings in station 4.  <BR>
   * - ME1 has only 3 rings (the hardware) not 4 (virtual ring 4 is used offline for ME1a).
   */
   IndexType ringsInStation( IndexType is ) const {
     const IndexType nRinS[5] = {3,2,2,2,0}; // rings per station
     return nRinS[is-1];
   }

  /**
   * How many chambers are there in ring ir of station is?
   *
   * Works for ME1a (virtual ring 4 of ME1) too.
   */
   IndexType chambersInRingOfStation(IndexType is, IndexType ir) const {
     const IndexType nCinR[16] = {36,36,36,36, 18,36,0,0, 18,36,0,0, 18,36,0,0}; // chambers in ring
     return nCinR[(is-1)*4 + ir - 1];
   }

  /** 
   * Number of strip channels per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-4.
   * Works for ME1a input as is=1, ir=4
   * WARNING: ME1a channels are considered to be unganged i.e. 48 not the standard 16.
   */
   IndexType stripChannelsPerLayer( IndexType is, IndexType ir ) const {
     const IndexType nSCinC[16] = { 80,80,64,48, 80,80,0,0, 80,80,0,0, 80,80,0,0 };
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
         {      1, 17281, 34561,252289,  48385, 57025,0,0,  74305, 82945,0,0, 100225,217729,0,0,
           108865,126145,143425,262657, 157249,165889,0,0, 183169,191809,0,0, 209089,235009,0,0 };
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
   *  Decode CSCDetId from various indexes and labels
   */
  CSCDetId detIdFromLayerIndex( IndexType ili ) const;
  CSCDetId detIdFromChamberIndex( IndexType ici ) const;
  CSCDetId detIdFromChamberIndex_OLD( IndexType ici ) const;
  CSCDetId detIdFromChamberLabel( IndexType ie, IndexType icl ) const;
  std::pair<CSCDetId, IndexType> detIdFromStripChannelIndex( LongIndexType ichi ) const;

  IndexType chamberLabelFromChamberIndex( IndexType ) const; // just for cross-checks

  /**
   * Build index used internally in online CSC conditions databases (the 'Igor Index')
   *
   * This is the decimal integer ie*100000 + is*10000 + ir*1000 + ic*10 + il <br>
   * (ie=1-2, is=1-4, ir=1-4, ic=1-36, il=1-6) <br>
   * Channels 1-16 in ME1a (is=1, ir=4) are reset to channels 65-80 of ME11.
   * WARNING: This is  NOT  adapted to unganged ME1a channels.
   */
  int dbIndex(const CSCDetId & id, int & channel);

private:
  void fillChamberLabel() const; // const so it can be called in const function detIdFromChamberIndex

  mutable std::vector<IndexType> chamberLabel; // mutable so can be filled by fillChamberLabel


};

#endif






