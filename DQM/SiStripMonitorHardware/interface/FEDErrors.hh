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

class FEDErrors {

public:
  
  struct FEDCounters {
    unsigned int nFEDErrors;
    unsigned int nDAQProblems;
    unsigned int nFEDsWithFEProblems;
    unsigned int nCorruptBuffers;
    unsigned int nBadActiveChannels;
    unsigned int nFEDsWithFEOverflows;
    unsigned int nFEDsWithFEBadMajorityAddresses;
    unsigned int nFEDsWithMissingFEs;
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
    bool IsActive;
    bool Unlocked;
    bool OutOfSync;

    bool operator <(const ChannelLevelErrors & aErr) const;
  };

  struct APVLevelErrors {
    unsigned int APVID;
    unsigned int ChannelID;
    bool IsActive;
    bool APVStatusBit;
    bool APVError;
    bool APVAddressError;

    bool operator <(const APVLevelErrors & aErr) const;
  };


  FEDErrors();

  ~FEDErrors();

  void initialise(const unsigned int aFedID);

  void hasCabledChannels(const bool isCabled);

  //bool foundFEDErrors();

  const bool anyDAQProblems();
  
  const bool anyFEDErrors();

  const bool anyFEProblems();

  const bool printDebug();

  const unsigned int fedID();

  static FEDCounters & getFEDErrorsCounters();

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

  void print(const FEDCounters & aFEDCounter, std::ostream & aOs = std::cout);
  void print(const FECounters & aFECounter, std::ostream & aOs = std::cout);
  void print(const FEDLevelErrors & aFEDErr, std::ostream & aOs = std::cout);
  void print(const FELevelErrors & aFEErr, std::ostream & aOs = std::cout);
  void print(const ChannelLevelErrors & aErr, std::ostream & aOs = std::cout);
  void print(const APVLevelErrors & aErr, std::ostream & aOs = std::cout);


protected:
  
private:

  unsigned int fedID_;

  FECounters feCounter_;
  FEDLevelErrors fedErrors_;
  std::vector<FELevelErrors> feErrors_;
  std::vector<ChannelLevelErrors> chErrorsDetailed_;
  std::vector<APVLevelErrors> apvErrors_;
  std::vector<std::pair<unsigned int,bool> > chErrors_;


};//class


#endif //DQM_SiStripMonitorHardware_FEDErrors_HH
