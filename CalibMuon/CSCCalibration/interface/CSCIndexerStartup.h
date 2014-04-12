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

  ~CSCIndexerStartup();

  std::string name() const {return "CSCIndexerStartup";}

  /// \name maxIndexMethods
  //@{
  LongIndexType maxStripChannelIndex() const { return 252288; }
  IndexType maxChipIndex() const             { return 15768; }
  IndexType maxGasGainIndex() const          { return 55944; }
  //@}


  /// \name nonIndexCountingMethods
  //@{
  /**
   * How many online rings are there in station 'is'=1, 2, 3, 4 ?
   *
   * \warning: ME1a + ME1b are considered as single ME1/1
   * 'online' rings for the startup
   */
  IndexType onlineRingsInStation( IndexType is ) const
  {
    const IndexType nrings[5] = { 0, 3, 2, 2, 2 };
    return nrings[is];
  }

  /**
   * Number of strip readout channels per layer in an offline chamber
   * with ring 'ir' and station 'is'.
   * Works for ME1a (ring 4 of ME1) too.
   *
   * Assume ME1a has 16 ganged readout channels.
   */
  IndexType stripChannelsPerOfflineLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nSC[16] = { 64,80,64,16,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSC[(is-1)*4 + ir - 1];
  }

  /**
   * Number of strip readout channels per layer in an online chamber
   * with ring 'ir' and station 'is'.
   * Works for ME1a (ring 4 of ME1) too.
   *
   * Assume ME1a has 16 ganged readout channels. Online chamber has 64+16=80 channels.
   */
  IndexType stripChannelsPerOnlineLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nSC[16] = { 80,80,64,80,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSC[(is-1)*4 + ir - 1];
  }

  /**
   * Number of Buckeye chips per layer in an online chamber
   * in ring 'ir' of station 'is'.
   * Works for ME1a (ring 4 of ME1) too.
   *
   * 'Online' ME11 for the startup is considered as a single chamber with 5 chips
   */
  IndexType chipsPerOnlineLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nCinL[16] = { 5,5,4,5,  5,5,0,0,  5,5,0,0,  5,5,0,0 };
    return nCinL[(is - 1)*4 + ir - 1];
  }

  //@}


  /// \name stripIndexMethods
  //@{
  /**
   * Number of strip channel indices for a layer in a chamber
   * defined by station number 'is' and ring number 'ir'.
   *
   * Station label range 1-4, Ring label range 1-4 (4=ME1a)
   *
   * This depends on the ordering of the channels in the database.
   * In startup scenario there are 80 indices allocated per ME1/1 layer with 1-64 belonging to ME1b
   * and 65-80 belonging to ME1a. So the ME1/a database indices are mapped to extend the ME1/b index ranges,
   * which is how the raw hardware channels numbering is implemented.
   */
  IndexType stripChannelsPerLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nSCinC[16] = { 80,80,64,80,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSCinC[(is - 1)*4 + ir - 1];
  }

  /**
   * Linear index for 1st strip channel in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-4.
   *
   * WARNING: while ME1a channels are the last 16 of the 80 total in each layer of an ME11 chamber,
   * their start index here defaults to the start index of ME1a.
   */
  LongIndexType stripChannelStart( IndexType ie, IndexType is, IndexType ir ) const
  {
    // These are in the ranges 1-217728 (CSCs 2008) and 217729-252288 (ME42).
    // There are 1-108884 channels per endcap (CSCs 2008) and 17280 channels per endcap (ME42).
    // Start of -z channels (CSCs 2008) is 108864 + 1 = 108865
    // Start of +z (ME42) is 217728 + 1 = 217729
    // Start of -z (ME42) is 217728 + 1 + 17280 = 235009
    const LongIndexType nStart[32] =
      {      1, 17281, 34561,     1,   48385, 57025,0,0,   74305, 82945,0,0,  100225,217729,0,0,
        108865,126145,143425,108865,  157249,165889,0,0,  183169,191809,0,0,  209089,235009,0,0 };
     return  nStart[(ie - 1)*16 + (is - 1)*4 + ir - 1];
  }
  //@}


  /// \name chipIndexMethods
  //@{
  /**
   * Number of Buckeye chips indices per layer in a chamber
   * in offline ring 'ir' of station 'is'.
   *
   * Station label range 1-4, Ring label range 1-4 (4=ME1a)
   *
   * \warning: the ME1a CFEB is just the last 1 of the 5 total in each layer of an ME11 chamber.
   * So, the input of ir=4 is will just return the same 5 total chips per whole ME11.
   *
   * Considers ME42 as standard 5 chip per layer chambers.
   */
  IndexType chipsPerLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nCinL[16] = { 5,5,4,5,  5,5,0,0,  5,5,0,0,  5,5,0,0 };
    return nCinL[(is - 1)*4 + ir - 1];
  }

  /**
   * Linear index for 1st Buckeye chip in offline ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-3.
   *
   * \warning: while ME1a chip is the last 1 of the 5 chips total in each layer of an ME11 chamber,
   * here the ME1a input ir=4 defaults to the ME1b start index (ir=4 <=> ir=1).
   */
  IndexType chipStart( IndexType ie, IndexType is, IndexType ir ) const
  {
    // These are in the ranges 1-13608 (CSCs 2008) and 13609-15768 (ME42).
    // There are 1-6804 chips per endcap (CSCs 2008) and 1080 channels per endcap (ME42).
    // Start of -z channels (CSCs 2008) is 6804 + 1 = 6805
    // Start of +z (ME42) is 13608 + 1 = 13609
    // Start of -z (ME42) is 13608 + 1 + 1080 = 14689
    const IndexType nStart[32] =
      {   1, 1081, 2161,    1,   3025, 3565, 0,0,   4645,  5185, 0,0,   6265, 13609,0,0,
       6805, 7885, 8965, 6805,   9829, 10369,0,0,  11449, 11989, 0,0,  13069, 14689,0,0 };
    return  nStart[(ie - 1)*16 + (is - 1)*4 + ir - 1];
  }
  //@}


  /// \name gasGainIndexMethods
  //@{
  /**
   * Linear index for 1st Gas gain sector in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-4 (4=ME1a)
   *
   * \warning: ME1a chip is the last 1 of the 5 chips total in each layer of an ME11 chamber,
   *            and an input ir=4 in this case would give the same result as ir=1
   */
  IndexType sectorStart( IndexType ie, IndexType is, IndexType ir ) const
  {
    // There are 36 chambers * 6 layers * 5 CFEB's * 1 HV segment = 1080 gas-gain sectors in ME1/1
    // There are 36*6*5*3 = 3240 gas-gain sectors in ME1/2
    // There are 36*6*4*3 = 2592 gas-gain sectors in ME1/3
    // There are 18*6*5*3 = 1620 gas-gain sectors in ME[2-4]/1
    // There are 36*6*5*5 = 5400 gas-gain sectors in ME[2-4]/2
    // Start of -z channels (CSCs 2008) is 22572 + 1 = 22573
    // Start of +z (ME42) is 45144 + 1 = 45145
    // Start of -z (ME42) is 45144 + 1 + 5400 = 50545
    const IndexType nStart[32] =
      {1    ,1081 , 4321,  1, //ME+1/1,ME+1/2,ME+1/3,ME+1/4
       6913 ,8533 ,    0,  0, //ME+2/1,ME+2/2,
       13933,15553,    0,  0, //ME+3/1,ME+3/2,
       20953,45145,    0,  0, //ME+4/1,ME+4/2,ME+4/3 (note, ME+4/2 index follows ME-4/1...)
       22573,23653,26893,  22573, //ME-1/1,ME-1/2,ME-1/3, ME-1/4
       29485,31105,    0,  0, //ME-2/1,ME-2/2,ME-2/3
       36505,38125,    0,  0, //ME-3/1,ME-3/2,ME-3/3
       43525,50545,    0,  0};//ME-4/1,ME-4/2,ME-4/3 (note, ME-4/2 index follows ME+4/2...)
    return  nStart[(ie-1)*16 + (is-1)*4 + ir - 1];
  }
  //@}


  /**
   *  Decode CSCDetId from various indexes and labels
   */
  std::pair<CSCDetId, IndexType> detIdFromStripChannelIndex( LongIndexType ichi ) const;
  std::pair<CSCDetId, IndexType> detIdFromChipIndex( IndexType ichi ) const;
  GasGainIndexType detIdFromGasGainIndex( IndexType igg ) const;

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
