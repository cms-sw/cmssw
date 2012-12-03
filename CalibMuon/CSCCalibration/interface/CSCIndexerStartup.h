#ifndef CSCIndexerStartup_H
#define CSCIndexerStartup_H

/** \class CSCIndexerStartup
 * Concrete CSCIndexer class appropriate for CSC Conditions Data access for CMS startup (2008-2013)
 * in which most ME4/2 chambers do not exist and the 48 ME1/1A strips are ganged into 16 channels. <br>
 * The conditions data are stored in an order based on the hadrware channel order so this class
 * has to jump through hoops in order to map between that order and a CSCDetID order offline.
 *
 * See documentation in base class CSCIndexerBase for more information.
 *
 * \warning This class is hard-wired for the CSC system at start-up of CMS in 2008.
 * with rings ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41 totalling 234 chambers per endcap.
 * But ME42 is appended (to permit simulation studies), so the chamber order is <br>
 * +z ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, <br>
 * -z ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, <br>
 * +z ME42, -z ME42 <br>
 *
 *  CSCIndexerBase::stripChannelIndex returns <br>
 *      1-217728 (CSCs 2008), 217729-252288 (ME42) (and ME1a channels are always channels 65-80 of 1-80 in ME11)
 *
 *  CSCIndexerBase::chipIndex returns <br>
 *      1-13608 (CSCs 2008), 13609-15768 (ME42)
 *
 *  CSCIndexerBase::gasGainIndex returns <br>
 *     1-45144 (CSCs 2008), 45145-55944 (ME42)
 *
 * \warning This uses magic numbers galore!!
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 */

#include <CalibMuon/CSCCalibration/interface/CSCIndexerBase.h>

class CSCIndexerStartup : public CSCIndexerBase {

public:

  CSCIndexerStartup(){};
  ~CSCIndexerStartup(){};

  std::string name() const {return "CSCIndexerStartup";}

   /**
   * Number of strip channels per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-4.
   * Works for ME1a input as is=1, ir=4.
   *
   * WARNING: <br>
   * - ME1a channels are ganged i.e. 16 <br>
   * - ME1b has 64 channels <br>
   * - ME11 is the 80 channels 1-64 for ME1b and 65-80 for ME1a.
   */
  IndexType stripChannelsPerLayer( IndexType is, IndexType ir ) const {
    const IndexType nSCinC[16] = { 64,80,64,16,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSCinC[(is-1)*4 + ir - 1];
  }

   /**
   * Number of strip channels between one layer and the next in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-4.
   * Works for ME1a input as is=1, ir=4.
   *
   * WARNING:
   * Although ME1a has only 16 channels, we must still skip 80 to get to the next ME1a channel
   * because these 16 channels are hardware channels 65-80 of ME11 and we must also skip over the
   * 64 of ME1b which are 1-64 of ME11.
   */
  IndexType stripChannelsToNextLayer( IndexType is, IndexType ir ) const {
    const IndexType nSCtoN[16] = { 80,80,64,80,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSCtoN[(is-1)*4 + ir - 1];
  }

  /**
   * Linear index for 1st strip channel in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-4.
   *
   * WARNING: ME1a channels are the last 16 of the 80 total in each layer of an ME11 chamber,
   * but this version has been extended to work for input ir=4 too.
   * 
   */
   LongIndexType stripChannelStart( IndexType ie, IndexType is, IndexType ir ) const {

     // These are in the ranges 1-217728 (CSCs 2008) and 217729-252288 (ME42).
     // There are 1-108884 channels per endcap (CSCs 2008) and 17280 channels per endcap (ME42).
     // Start of -z channels (CSCs 2008) is 108864 + 1 = 108865
     // Start of +z (ME42) is 217728 + 1 = 217729
     // Start of -z (ME42) is 217728 + 1 + 17280 = 235009

      const LongIndexType nStart[32] = { 1,17281,34561,65,  48385,57025,0,0,  74305,82945,0,0,  100225,217729,0,0, 
					 108865,126145,143425,108929,  157249,165889,0,0,  183169,191809,0,0,  209089,235009,0,0 };
      return  nStart[(ie-1)*16 + (is-1)*4 + ir - 1];

   }

  /**
   * Number of Buckeye chips per layer in a chamber in ring ir of station is.
   *
   * Station label range 1-4, Ring label range 1-4 (4=ME1a)
   *
   * WARNING: ME1a channels are the last 1 of the 5 total in each layer of an ME11 chamber,
   * and an input ir=4 is will just return 0.
   *
   * Considers ME42 as standard 5 chip per layer chambers.
   */
  IndexType chipsPerLayer( IndexType is, IndexType ir ) const {
    const IndexType nCinL[16] = { 5,5,4,0,  5,5,0,0,  5,5,0,0,  5,5,0,0 };
    return nCinL[(is-1)*4 + ir - 1];
  }

  /**
   * Linear index for 1st Buckeye chip in ring 'ir' of station 'is' in endcap 'ie'.
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
   * Linear index for 1st Gas gain sector in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-3.
   *
   * WARNING: ME1a channels are the last 1 of the 5 chips total in each layer of an ME11 chamber, 
   * and an input ir=4 is invalid and will give nonsense.
   */
  IndexType sectorStart( IndexType ie, IndexType is, IndexType ir ) const {
    // There are 36 chambers * 6 layers * 5 CFEB's * 1 HV segment = 1080 gas-gain sectors in ME1/1
    // There are 36*6*5*3 = 3240 gas-gain sectors in ME1/2
    // There are 36*6*4*3 = 2592 gas-gain sectors in ME1/3
    // There are 18*6*5*3 = 1620 gas-gain sectors in ME[2-4]/1
    // There are 36*6*5*5 = 5400 gas-gain sectors in ME[2-4]/2
    // Start of -z channels (CSCs 2008) is 22572 + 1 = 22573
    // Start of +z (ME42) is 45144 + 1 = 45145
    // Start of -z (ME42) is 45144 + 1 + 5400 = 50545
    const IndexType nStart[24] = {1    ,1081 , 4321,   //ME+1/1,ME+1/2,ME+1/3
				  6913 ,8533 ,    0,   //ME+2/1,ME+2/2,ME+2/3
				  13933,15553,    0,   //ME+3/1,ME+3/2,ME+3/3
				  20953,45145,    0,   //ME+4/1,ME+4/2,ME+4/3 (note, ME+4/2 index follows ME-4/1...)
				  22573,23653,26893,   //ME-1/1,ME-1/2,ME-1/3
				  29485,31105    ,0,   //ME-2/1,ME-2/2,ME-2/3
				  36505,38125,    0,   //ME-3/1,ME-3/2,ME-3/3
				  43525,50545,    0};  //ME-4/1,ME-4/2,ME-4/3 (note, ME-4/2 index follows ME+4/2...)
    return  nStart[(ie-1)*12 + (is-1)*3 + ir - 1];
  }

  /**
   *  Decode CSCDetId from various indexes and labels
   */
  std::pair<CSCDetId, IndexType> detIdFromStripChannelIndex( LongIndexType ichi ) const;
  std::pair<CSCDetId, IndexType> detIdFromChipIndex( IndexType ichi ) const;

  /**
   * Build index used internally in online CSC conditions databases (the 'Igor Index')
   *
   * This is the decimal integer ie*100000 + is*10000 + ir*1000 + ic*10 + il <br>
   * (ie=1-2, is=1-4, ir=1-4, ic=1-36, il=1-6) <br>
   * Channels 1-16 in ME1A (is=1, ir=4) are reset to channels 65-80 of ME11.
   */
  int dbIndex(const CSCDetId & id, int & channel) const;

};

#endif






