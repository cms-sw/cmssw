#ifndef RPCFILEREADER_RPCFILEREADER_H
#define RPCFILEREADER_RPCFILEREADER_H

/** \class RPCFileReader
 *
 *  Read PAC data from ASCII files convert them and write as FEDRawData
 *
 *  $Date: $
 *  $Revision: $
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

using namespace std;
using namespace edm;

class RPCFileReader : public ExternalInputSource {

 public:
  explicit RPCFileReader(ParameterSet const& pset, InputSourceDescription const& desc);

  ~RPCFileReader();
  
  virtual bool produce(Event &);
  virtual void setRunAndEventInfo();

  int run(){ return run_; }
  int event(){ return event_; }
  int bxn(){ return bxn_; }

  void SetRun(int runNo){ run_=runNo; }
  void SetEvent(int eventNo){ event_=eventNo; }
  void SetBxn(int bxnNo){ bxn_=bxnNo; }

 private:

  //typedefs
  typedef struct{unsigned int year; string month; unsigned int day, hour, min, sec;} Time;
  typedef struct{int ptCode, quality, sign;} LogCone;
  typedef unsigned short Word16;
  typedef unsigned long long Word64;

  //data members
  int run_, event_, bxn_;
  Time timeStamp_;
  std::vector<LogCone> theLogCones_;//(12)
  std::vector<std::vector<RPCPacData> > linkData_;//(3,18)

  bool isOpen_, noMoreData_;
  int eventPos_[2];
  int fileCounter_, eventCounter_;

  bool debug_;

  unsigned int triggerFedId_, tbNum_;

  // consts
  //FIXME: At the moment PACTrigger delay isn't known
  const static unsigned int RPC_PAC_TRIGGER_DELAY=10;
  const static unsigned int RPC_PAC_L1ACCEPT_BX=1;
  
  // methods
  void ReadDataFromAsciiFile(string fileName, int *pos);
  
  unsigned short BuildCDWord(RPCPacData linkData);
  unsigned short BuildSLDWord(unsigned int tbNum, unsigned int linkNum);
  unsigned short BuildSBXDWord(unsigned int bxn);
  unsigned short BuildEmptyWord();
    
  FEDRawData* RPCDataFormatter();

  bool IsHexNumber(string str){//utility to check if string is hex
    for(unsigned int i=0; i<str.size(); i++)
      if(! isxdigit(str[i])) return false;
    return true;
  }

};

#endif //RPCFILEREADER_RPCFILEREADER_H
