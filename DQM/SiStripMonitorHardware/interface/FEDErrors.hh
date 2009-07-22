// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      FEDErrors
// 
/**\class FEDErrors DQM/SiStripMonitorHardware/interface/FEDErrors.hh

 Description: class summarising FED errors
*/
//
// Original Author:  Nicholas Cripps in plugin file
//         Created:  2008/09/16
// Modified by    :  Anne-Marie Magnan, code copied from plugin to this class
//

#ifndef DQM_SiStripMonitorHardware_FEDErrors_HH
#define DQM_SiStripMonitorHardware_FEDErrors_HH

#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"


class FEDErrors {

public:
  
  struct FEDCounters {
    unsigned int nFEDErrors;
    unsigned int nDAQProblems;
    unsigned int nFEDsWithFEProblems;
    unsigned int nCorruptBuffers;
    unsigned int nBadChannels;
    unsigned int nBadActiveChannels;
    unsigned int nFEDsWithFEOverflows;
    unsigned int nFEDsWithFEBadMajorityAddresses;
    unsigned int nFEDsWithMissingFEs;
    unsigned int nTotalBadChannels;
    unsigned int nTotalBadActiveChannels;
  };

  struct ChannelCounters {
    unsigned int nNotConnected;
    unsigned int nUnlocked;
    unsigned int nOutOfSync;
    unsigned int nAPVStatusBit;
    unsigned int nAPVError;
    unsigned int nAPVAddressError;
  };

  struct FECounters {
    unsigned int nFEOverflows; 
    unsigned int nFEBadMajorityAddresses; 
    unsigned int nFEMissing;
  };

  struct FEDLevelErrors {
    bool HasCabledChannels;
    bool DataPresent;
    bool DataMissing;
    bool InvalidBuffers;
    bool BadFEDCRCs;
    bool BadDAQCRCs;
    bool BadIDs;
    bool BadDAQPacket;
    bool CorruptBuffer;
    bool FEsOverflow;
    bool FEsMissing;
    bool FEsBadMajorityAddress;
    bool BadChannelStatusBit;
    bool BadActiveChannelStatusBit;
  };

  struct FELevelErrors {
    unsigned short FeID;
    bool Overflow;
    bool Missing;
    bool BadMajorityAddress;
  };

  struct ChannelLevelErrors {
    unsigned int ChannelID;
    bool Connected;
    bool IsActive;
    bool Unlocked;
    bool OutOfSync;
    bool operator <(const ChannelLevelErrors & aErr) const;
  };

  struct APVLevelErrors {
    unsigned int APVID;
    unsigned int ChannelID;
    bool Connected;
    bool IsActive;
    bool APVStatusBit;
    bool APVError;
    bool APVAddressError;

    bool operator <(const APVLevelErrors & aErr) const;
  };


  FEDErrors();

  ~FEDErrors();

  void initialise(const unsigned int aFedID,
		  const SiStripFedCabling* aCabling);

  //return false if no data, with or without cabled channels.
  bool checkDataPresent(const FEDRawData& aFedData);

  //perform a sanity check with unpacking code check
  bool failUnpackerFEDCheck(const FEDRawData & fedData);


  //return true if there were no errors at the level they are analysing
  //ie analyze FED returns true if there were no FED level errors which prevent the whole FED being unpacked
  //fill errors: define the order of importance.
  bool fillFEDErrors(const FEDRawData& aFedData,
		     bool & aFullDebug,
		     const unsigned int aPrintDebug
		     );

  bool fillFEErrors(const sistrip::FEDBuffer* aBuffer);

  bool fillChannelErrors(const sistrip::FEDBuffer* aBuffer,
			 bool & aFullDebug,
			 const unsigned int aPrintDebug
			 );

  //1--Add all channels of a FED if anyFEDErrors or corruptBuffer
  //2--if aFillAll = true, add all channels anyway with 0 if no errors, so TkHistoMap is filled for all valid channels ...
  void fillBadChannelList(std::map<unsigned int,std::pair<unsigned short,unsigned short> > & aMap,
			  const SiStripFedCabling* aCabling,
			  unsigned int & aNBadChannels,
			  unsigned int & aNBadActiveChannels,
			  const bool aFillAll);

  //bool foundFEDErrors();

  const bool failMonitoringFEDCheck();

  const bool anyDAQProblems();
  
  const bool anyFEDErrors();

  const bool anyFEProblems();

  const bool printDebug();

  const unsigned int fedID();

  static FEDCounters & getFEDErrorsCounters();

  static ChannelCounters & getChannelErrorsCounters();

  FECounters & getFEErrorsCounters();

  FEDLevelErrors & getFEDLevelErrors();
  
  std::vector<FELevelErrors> & getFELevelErrors();

  std::vector<ChannelLevelErrors> & getChannelLevelErrors();

  std::vector<APVLevelErrors> & getAPVLevelErrors();

  std::vector<std::pair<unsigned int, bool> > & getBadChannels();

  void addBadFE(const FELevelErrors & aFE);

  void addBadChannel(const ChannelLevelErrors & aChannel);

  void addBadAPV(const APVLevelErrors & aAPV, bool & aFirst);

  void incrementFEDCounters();

  void incrementChannelCounters(const FEDErrors::ChannelLevelErrors & aChannel);

  void incrementAPVCounters(const FEDErrors::APVLevelErrors & aAPV);

  void print(const FEDCounters & aFEDCounter, std::ostream & aOs = std::cout);
  void print(const FECounters & aFECounter, std::ostream & aOs = std::cout);
  void print(const FEDLevelErrors & aFEDErr, std::ostream & aOs = std::cout);
  void print(const FELevelErrors & aFEErr, std::ostream & aOs = std::cout);
  void print(const ChannelLevelErrors & aErr, std::ostream & aOs = std::cout);
  void print(const APVLevelErrors & aErr, std::ostream & aOs = std::cout);


protected:
  
private:

  unsigned int fedID_;

  bool connected_[sistrip::FEDCH_PER_FED];

  FECounters feCounter_;
  FEDLevelErrors fedErrors_;
  std::vector<FELevelErrors> feErrors_;
  std::vector<ChannelLevelErrors> chErrorsDetailed_;
  std::vector<APVLevelErrors> apvErrors_;
  std::vector<std::pair<unsigned int,bool> > chErrors_;

  bool failUnpackerFEDCheck_;

};//class


#endif //DQM_SiStripMonitorHardware_FEDErrors_HH
