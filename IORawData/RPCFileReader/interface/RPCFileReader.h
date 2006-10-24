#ifndef RPCFILEREADER_RPCFILEREADER_H
#define RPCFILEREADER_RPCFILEREADER_H

/** \class RPCFileReader
 *
 *  Read PAC data from ASCII files convert them and write as FEDRawData
 *
 *  $Date: 2006/10/18 16:09:10 $
 *  $Revision: 1.8 $
 * \author Michal Bluj - SINS, Warsaw
*/
#include <memory>
#include <string>
#include <cctype>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ExternalInputSource.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IORawData/RPCFileReader/interface/RPCPacData.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"


class RPCFileReader : public edm::ExternalInputSource {

 public:
  explicit RPCFileReader(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc);

  ~RPCFileReader();
  
  virtual bool produce(edm::Event &);
  virtual void setRunAndEventInfo();

 private:

  //typedefs
  typedef struct{unsigned int year; std::string month; unsigned int day, hour, min, sec;} Time;
  typedef struct{int ptCode, quality, sign;} LogCone;
  typedef unsigned short Word16;
  typedef unsigned long long Word64;

  //data members
  int run_, event_, bxn_;
  Time timeStamp_;
  std::vector<std::vector<LogCone> > theLogCones_;//(12)
  std::vector<std::vector<std::vector<std::vector<RPCPacData> > > > linkData_;//(3,18,3)

  bool isOpen_, noMoreData_;
  int eventPos_[2];
  int fileCounter_, eventCounter_;

  bool debug_,saveOutOfTime_,pacTrigger_,maskChannels_;

  unsigned int triggerFedId_;//, tbNum_;
  std::vector<unsigned int> tbNums_;

  // consts
  //FIXME: Checked
  const static int RPC_PAC_TRIGGER_DELAY=11;
  const static int RPC_PAC_L1ACCEPT_BX=2;
  
  const static int LAST_BX=6; 
  const static int FIRST_BX=-2; 

  std::vector<std::vector<bool> > maskedChannels;

  // methods
  void readDataFromAsciiFile(std::string fileName, int *pos);
  
  Word16 buildCDWord(RPCPacData linkData);
  Word16 buildSLDWord(unsigned int tbNum, unsigned int linkNum);
  Word16 buildSBXDWord(unsigned int bxn);
  Word16 buildEmptyWord();
    
  FEDRawData* rpcDataFormatter();

  bool isHexNumber(std::string str){//utility to check if string is hex
    for(unsigned int i=0; i<str.size(); i++)
      if(! isxdigit(str[i])) return false;
    return true;
  }

};

#endif //RPCFILEREADER_RPCFILEREADER_H
