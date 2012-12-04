#ifndef CSCIndexerPostls1_H
#define CSCIndexerPostls1_H

/** \class CSCIndexerPostls1
 * Concrete CSCIndexer class appropriate for CSC Conditions Data access after CMS long shutdown 1 (2013-2014)
 * in which most ME4/2 rings are complete and the 48 strips in ME1/1A are unganged and so have 48 channels. <br>
 * The conditions data are stored in an order based on the hadrware channel order so this class
 * has to jump through hoops in order to map between that order and a CSCDetID order offline.
 *
 * See documentation in base class CSCIndexerBase for more information.
 *
 * \warning This class is hard-wired for the CSC system expected after long shutdown 2013-2014 (LS1) of CMS.<br>
 * The basic order is as for startup (pre-LS1)
 * with rings ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41 totalling 234 chambers per endcap.
 * Then ME42 is appended, so the chamber order is <br>
 * +z ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, <br>
 * -z ME11, ME12, ME13, ME21, ME22, ME31, ME32, ME41, <br>
 * +z ME42, -z ME42 <br>
 *
 * It is further extended for unganged ME1a strip channels by appending +z ME1a, -z ME1a.
 *
 *  CSCIndexerBase::stripChannelIndex returns <br>
 *      1-217728 (CSCs 2008), 217729-252288 (ME42), 252289-273024 (unganged ME1a)
 *
 *  CSCIndexerBase::chipIndex returns <br>
 *      1-13608 (CSCs 2008), 13609-15768 (ME42), 15769-17064 (unganged ME1a).
 *
 *  CSCIndexerBase::gasGainIndex returns <br>
 *     1-45144 (CSCs 2008), 45145-55944 (ME42), 55945-57240 (unganged ME1a)
 *
 * \warning This has "magic numbers galore".
 * \warning EVERY LABEL COUNTS FROM ONE NOT ZERO.
 *
 */

#include <CalibMuon/CSCCalibration/interface/CSCIndexerBase.h>

class CSCIndexerPostls1 : public CSCIndexerBase
{
public:

  ~CSCIndexerPostls1();

  std::string name() const { return "CSCIndexerPostls1"; }


  /// \name maxIndexMethods
  //@{
  LongIndexType maxStripChannelIndex() const { return 273024; }
  IndexType maxChipIndex() const             { return 17064; }
  IndexType maxGasGainIndex() const          { return 57240; }
  //@}


  /// \name nonIndexCountingMethods
  //@{
  /**
   * How many online rings are there in station 'is'=1, 2, 3, 4 ?
   *
   * \warning: ME1a and ME1b are considered as two separate
   * 'online' rings for the upgrade
   */
  IndexType onlineRingsInStation( IndexType is ) const
  {
    const IndexType nrings[5] = { 0, 4, 2, 2, 2 };
    return nrings[is];
  }

  /**
   * Number of strip readout channels per layer in an offline chamber
   * with ring 'ir' and station 'is'.
   *
   * Assume ME1a has 48 unganged readout channels.
   */
  IndexType stripChannelsPerOfflineLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nSC[16] = { 64,80,64,48,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSC[(is-1)*4 + ir - 1];
  }

  /**
   * Number of strip readout channels per layer in an online chamber
   * with ring 'ir' and station 'is'.
   *
   * Assume ME1a has 48 unganged readout channels.
   * Online chambers ME1a and ME1b are separate.
   */
  IndexType stripChannelsPerOnlineLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nSC[16] = { 64,80,64,48,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSC[(is-1)*4 + ir - 1];
  }

  /**
   * Number of Buckeye chips per layer in an online chamber
   * in ring 'ir' of station 'is'.
   * Works for ME1a (ring 4 of ME1) too.
   *
   * 'Online' ME11 for the upgrade is considered as split into 1a and 1b
   * chambers with 3 and 4 CFEBs respectively
   */
  IndexType chipsPerOnlineLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nCinL[16] = { 4,5,4,3,  5,5,0,0,  5,5,0,0,  5,5,0,0 };
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
   * In the currently implemented upgrade scenario, the ME1b still keeps the
   * 80 indices wide ranges inherited from Startup (with the last 65-80 indices remaining unused),
   * while the ME1/1A unganged channels get their own 48 indices wide ranges.
   */
  IndexType stripChannelsPerLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nSCinC[16] = { 80,80,64,48,  80,80,0,0,  80,80,0,0,  80,80,0,0 };
    return nSCinC[(is - 1)*4 + ir - 1];
  }

  /**
   * Linear index for 1st strip channel in ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-4 (4=ME1a)
   *
   * WARNING: ME1a channels are  NOT  considered the last 16 of the 80 total in each layer of an ME11 chamber!
   */
  LongIndexType stripChannelStart( IndexType ie, IndexType is, IndexType ir ) const
  {
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
   * Works for ME1a input as is=1, ir=4
   * Considers ME42 as standard 5 chip per layer chambers.
   *
   * WARNING:
   * - ME1a channels are considered to be unganged and have their own 3 chips (ME1b has 4 chips).
   * - ME1b keeps 5 chips for the indexing purposes, however indices for the chip #5 are ignored in the unganged case.
   */
  IndexType chipsPerLayer( IndexType is, IndexType ir ) const
  {
    const IndexType nCinL[16] = { 5,5,4,3,  5,5,0,0,  5,5,0,0,  5,5,0,0 };
    return nCinL[(is - 1)*4 + ir - 1];
  }

  /**
   * Linear index for 1st Buckeye chip in offline ring 'ir' of station 'is' in endcap 'ie'.
   *
   * Endcap label range 1-2, Station label range 1-4, Ring label range 1-4 (4=ME1a)
   * Works for ME1a input as is=1, ir=4
   *
   * \warning: ME1a chips are the last 3 of the 7 chips total in each layer of an ME11 chamber,
   */
  IndexType chipStart( IndexType ie, IndexType is, IndexType ir ) const
  {
    // These are in the ranges 1-13608 (CSCs 2008) and 13609-15768 (ME42) and 15769-17064 (ME1a).
    // There are 1-6804 chips per endcap (CSCs 2008) and 1080 chips per endcap (ME42) and 648 chips per endcap (ME1a).
    // Start of -z channels (CSCs 2008) is 6804 + 1 = 6805
    // Start of +z (ME42) is 13608 + 1 = 13609
    // Start of -z (ME42) is 13608 + 1 + 1080 = 14689
    // Start of +z (ME1a) is 15768 + 1 = 15769
    // Start of -z (ME1a) is 15768 + 1 + 648 = 16417
    const IndexType nStart[32] =
      {1,   1081, 2161, 15769,   3025, 3565, 0,0,  4645, 5185, 0,0,  6265, 13609,0,0,
       6805,7885, 8965, 16417,   9829, 10369,0,0,  11449,11989,0,0,  13069,14689,0,0 };
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
   * \warning:  unganged ME1a has 3 own chips, which are currently appended to the end of the index range,
   *            ME1b still keeps 5 chips with the chip #5 index being unused.
   */
  IndexType sectorStart( IndexType ie, IndexType is, IndexType ir ) const
  {
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
    const IndexType nStart[32] =
      {1    , 1081 ,  4321, 55945,  //ME+1/1,ME+1/2,ME+1/3,ME+1/a
       6913 , 8533 ,     0,     0,  //ME+2/1,ME+2/2
       13933, 15553,     0,     0,  //ME+3/1,ME+3/2
       20953, 45145,     0,     0,  //ME+4/1,ME+4/2 (note, ME+4/2 index follows ME-4/1...)
       22573, 23653, 26893, 56593,  //ME-1/1,ME-1/2,ME-1/3,ME+1/a
       29485, 31105,     0,     0,  //ME-2/1,ME-2/2
       36505, 38125,     0,     0,  //ME-3/1,ME-3/2
       43525, 50545,     0,     0 };//ME-4/1,ME-4/2 (note, ME-4/2 index follows ME+4/2...)
    return  nStart[(ie-1)*16 + (is-1)*4 + ir - 1];
  }
  //@}

  /**
   *  Decode CSCDetId from various indexes and labels
   */
  std::pair<CSCDetId, IndexType> detIdFromStripChannelIndex( LongIndexType ichi ) const;
  std::pair<CSCDetId, IndexType> detIdFromChipIndex( IndexType ichi ) const;
  CSCIndexerBase::GasGainIndexType detIdFromGasGainIndex( IndexType igg ) const;

  /**
   * Build index used internally in online CSC conditions databases (the 'Igor Index')
   *
   * This is the decimal integer ie*100000 + is*10000 + ir*1000 + ic*10 + il <br>
   * (ie=1-2, is=1-4, ir=1-4, ic=1-36, il=1-6) <br>
   * Channels 1-16 in ME1A (is=1, ir=4) are NOT reset to channels 65-80 of ME11.
   * WARNING: This is now ADAPTED for unganged ME1a channels
   *          (we expect that the online conditions DB will adopt it too).
   */
  int dbIndex(const CSCDetId & id, int & channel) const;

};

#endif
