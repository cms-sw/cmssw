#ifndef MuonDetId_CSCIndexer_h
#define MuonDetId_CSCIndexer_h

/** \class CSCIndexer
 * Handle creation and access of a linear index for each layer 
 * and each strip channel of CSC chamber system
 * 
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <iosfwd>

class CSCIndexer {

public:
	CSCIndexer();
	~CSCIndexer(){};

  /**
   * Linear index to label each layer of each hardware chamber.
   * Arguments are the usual csc labels for endcap, station, ring, chamber, layer.
   *
   * Output is 1 to 2808
   *
   * WARNING: Do not input ME1a values (i.e. ring 4): does not consider ME1a and ME1b to be separate,
   * No sanity checking on input values!
   */
   unsigned int layerIndex( unsigned short int ie, unsigned short int is, unsigned short int ir, unsigned short int ic, unsigned short int il) const {
    const unsigned short int chambersInEndcap = 234;
    const unsigned short int layersInChamber = 6;
    return layersInChamber*( (ie-1)*chambersInEndcap + chambersUpToStation(is-1)
	       + chambersInStationUpToRing(is, ir-1) + (ic-1) )+ il;
  }

  /**
   * Linear index to label each layer in CSC subdetectors.
   * Argument is a CSCDetId.
   *
   * Output is 1 to 2808
   *
   * WARNING: Do not input ME1a values (i.e. ring 4): does not consider ME1a and ME1b to be separate,
   * No sanity checking on input value: if you supply an ME1a CSCDetId then you'll get nonsense.
   */
   unsigned int layerIndex( CSCDetId& id ) const {
    return layerIndex(id.endcap(), id.station(), id.ring(), id.chamber(), id.layer());
  }

	/**
	 * How many rings are there in station is=1, 2, 3, 4 ?
	*/
	unsigned short int ringsInStation( unsigned short int is ){
		const unsigned short int nrins[5]={0,3,2,2,1}; // rings per station
		return nrins[is];
	}

  /**
   * How many chambers are there in a station is=1, 2, 3, 4 ?
   *
   * Does not consider ME1a and ME1b to be separate.
   */
   unsigned short int chambersInStation( unsigned short int is ) const {
      const unsigned short int ncins[5]={0,108,54,54,18}; // chambers per station (-) 1 2 3 4
      return ncins[is];
    }

  /**
   * How many chambers in stations UP TO AND INCLUDING  is=1, 2, 3, 4 ?
   *
   * Does not consider ME1a and ME1b to be separate.
   */
   unsigned short int chambersUpToStation( unsigned short int is ) const {
      const unsigned short int ncins[5]={0, 108,162,216,234}; // chambers in: 0, 1, 1+2, 1+2+3, 1+2+3+4
      return ncins[is];
    }

  /**
   * How many chambers are there in ring ir of station is?
   *
   * Works for ME1a (ring 4 of ME1) too.
   */
   unsigned short int chambersInRingOfStation(unsigned short int is, unsigned short int ir) const {
    short int nc = 36; // most rings have 36 chambers
    if (is >1 && ir<2 ) nc = 18; // but 21, 31, 41 have 18
    return nc;
  }

  /**
   * How many chambers are there in a station is=1, 2, 3, 4, UP TO AND INCLUDING ring ir?
   *
   * WARNING: Do not input ME1a values (i.e. ring 4): does not consider ME1a and ME1b to be separate,
   * No sanity checking on input values!
   */
   unsigned short int chambersInStationUpToRing( unsigned short int is, unsigned short int ir) const {
    //             ir=     1  2    3   
    // station 1 rings (0),1,1+2,1+2+3  
    // station 2 rings (0),1,1+2,1+2 (only 2 rings)
    // station 3 rings (0),1,1+2,1+2 (only 2 rings)
    // station 4 rings (0),1, 1,  1 (only 1 ring)
    const unsigned short int nCinSuptoR[16]={0,36,72,108,  0,18,54,54,  0,18,54,54,  0,18,18,18};
    return nCinSuptoR[(is-1)*4+ir];
   }

  /** 
   * Number of strip channels per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-3.
   *
   * WARNING: ME1a channels are the last 16 of the 80 total in each layer of an ME11 chamber, 
   * and an input ir=4 is invalid and will give nonsense.
   */
   unsigned short int stripChannelsPerLayer( unsigned short int is, unsigned short int ir ) const {
    const unsigned short int nSCinC[12] = { 80,80,64, 80,80,0, 80,80,0, 80,0,0 };
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
	unsigned int stripChannelStart( unsigned short int ie, unsigned short int is, unsigned short int ir ) const {
		const unsigned int stripChannelsPerEndcap = 108864;
		const unsigned int nStart[12] = { 1,17281,34561, 48385,57025,0, 74305,82945,0, 100225,0,0 };
		return (ie-1)*stripChannelsPerEndcap + nStart[(is-1)*3 + ir - 1];
	}

  /**
   * Linear index for strip channel istrip in layer il of chamber ic of ring ir
   * in station is of endcap ie.
   *
   * Use at your own risk! You must input labels within hardware ranges.
   * No trapping on out-of-range values!
   */
	unsigned int stripChannelIndex( unsigned short int ie, unsigned short int is, unsigned short int ir, unsigned short int ic, unsigned short int il, unsigned short int istrip ) const {
		return stripChannelStart(ie,is,ir)+( (ic-1)*6 + il - 1 )*stripChannelsPerLayer(is,ir) + (istrip-1);
  }

  /**
   *  Decode CSCDetId from layer linear index
   */
	CSCDetId detIdFromLayer( unsigned int lin ) const;

  /**
   * Decode CSCDetId from strip channel linear index
   */
	CSCDetId detIdFromStrip( unsigned int sin ) const;
  
  /**
   * Decode strip channel index from overall strip channel linear index
   */
	unsigned short int stripChannel( unsigned int sin ) const;

private:
	std::vector<unsigned int> igorIndex;

};

#endif

