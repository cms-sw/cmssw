#ifndef MuonDetId_CSCIndexer_h
#define MuonDetId_CSCIndexer_h

/** \class CSCIndexer
 * Creates a linear index for various sublevels of the endcap muon CSC system.
 *
 * It supplies a linear index for:
 * 1. Each chamber of the CSC system: range 1-468 <br>
 * 2. Each layer of the CSC system: range 1-2808 <br>
 * 3. Each strip channel of the CSC system: range 1-217728 <br>
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
 * \warning This class is hard-wired for the CSC system at start-up of CMS in 2008.
 * with rings ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41 totalling 234 chambers per endcap.
 *
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <iosfwd>
#include <utility>  // for pair

class CSCIndexer {

  //  typedef unsigned short int IndexType;
  //  typedef unsigned       int LongIndexType;
  typedef uint16_t IndexType;
  typedef uint32_t LongIndexType;

public:
  CSCIndexer(){};
  ~CSCIndexer(){};

  /**
   * Linear index to label each CSC in CSC system.
   * Argument is the CSCDetId of some CSCChamber.
   *
   * Output is 1 to 468.
   *
   * WARNING: Do not input ME1a values (i.e. ring '4'): does not consider ME1a and ME1b to be separate,
   * No sanity checking on input value: if you supply an ME1a CSCDetId then you'll get nonsense.
   */

   IndexType chamberIndex( CSCDetId& id ) const {
     return chamberIndex( id.endcap(), id.station(), id.ring(), id.chamber() );
   }

  /**
   * Linear index to label each hardware layer in CSC system.
   * Argument is the CSCDetId of some CSCLayer.
   *
   * Output is 1 to 2808.
   *
   * WARNING: Do not input ME1a values (i.e. ring '4'): does not consider ME1a and ME1b to be separate,
   * No sanity checking on input value: if you supply an ME1a CSCDetId then you'll get nonsense.
   */

   IndexType layerIndex( CSCDetId& id ) const {
     return layerIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer());
   }

   /** 
    * Starting index in range 1-234 for first chamber in ring ir of station is
    */
   IndexType startChamberIndexInEndcap(IndexType is, IndexType ir) const {
     const IndexType nschin[12] = {1,37,73, 109,127,0, 163,181,0, 217,0,0};
     return nschin[(is-1)*3+ir-1];
   }

   /** 
    * Starting index in range 1-468 for first chamber in ring ir of station is in endcap ie
    */
   IndexType startChamberIndex(IndexType ie, IndexType is, IndexType ir) const {
     const IndexType nchpere = 234;
     return (ie-1)*nchpere + startChamberIndexInEndcap(is, ir);
   }

   /**
    *  Linear index in range 1-468 for chamber ic in ring ir of station is in endcap ie
    */
   IndexType chamberIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic) const {
     return startChamberIndex(ie,is,ir) + ic - 1; // -1 so start index _is_ ic=1
   }

   /**
    * Linear index in 1-2808 for layer il of chamber ic in ring ir of station is in endcap ie
    */
   IndexType layerIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic, IndexType il) const {
    const IndexType layersInChamber = 6;
     return (chamberIndex(ie,is,ir,ic) - 1 ) * layersInChamber + il;
   }

  /**
   * How many rings are there in station is=1, 2, 3, 4 ?
   */
   IndexType ringsInStation( IndexType is ) const {
      const IndexType nrins[5] = {0,3,2,2,1}; // rings per station
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
   * and an input ir=4 is invalid and will give nonsense.
   */
   IndexType stripChannelsPerLayer( IndexType is, IndexType ir ) const {
     const IndexType nSCinC[12] = { 80,80,64, 80,80,0, 80,80,0, 80,0,0 };
     return nSCinC[(is-1)*3 + ir - 1];
   }

  /**
   * Linear index for 1st strip channel in ring ir of station is in endcap ie.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-3.
   *
   * WARNING: ME1a channels are the last 16 of the 80 total in each layer of an ME11 chamber, 
   * and an input ir=4 is invalid and will give nonsense.
   */
   LongIndexType stripChannelStart( IndexType ie, IndexType is, IndexType ir ) const {
      const LongIndexType stripChannelsPerEndcap = 108864;
      const LongIndexType nStart[12] = { 1,17281,34561, 48385,57025,0, 74305,82945,0, 100225,0,0 };
      return (ie-1)*stripChannelsPerEndcap + nStart[(is-1)*3 + ir - 1];
   }

  /**
   * Linear index for strip channel istrip in layer il of chamber ic of ring ir
   * in station is of endcap ie.
   *
   * Output is 1-217728.
   *
   * WARNING: Use at your own risk! You must input labels within hardware ranges.
   * No trapping on out-of-range values!
   */
   LongIndexType stripChannelIndex( IndexType ie, IndexType is, IndexType ir, IndexType ic, IndexType il, IndexType istrip ) const {
      return stripChannelStart(ie,is,ir)+( (ic-1)*6 + il - 1 )*stripChannelsPerLayer(is,ir) + (istrip-1);
   }

  /**
   * Linear index for strip channel istrip in layer labelled by CSCDetId id.
   *
   * Output is 1-217728.
   *
   * WARNING: Use at your own risk! The supplied CSCDetId must be a layer id.
   * No trapping on out-of-range values!
   */

   LongIndexType stripChannelIndex( CSCDetId& id, IndexType istrip ) const {
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

  IndexType checkLabel( IndexType ) const;

private:
  void fillChamberLabel() const; // const so it can be called in const function detIdFromChamberIndex

  mutable std::vector<IndexType> chamberLabel; // mutable so can be filled by fillChamberLabel


};

#endif

