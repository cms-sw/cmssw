#ifndef DTHVStatusHandler_H
#define DTHVStatusHandler_H
/** \class DTHVStatusHandler
 *
 *  Description: Class to copy HV status via PopCon
 *
 *
 *  $Date: 2009/12/08 16:11:34 $
 *  $Revision: 1.3 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "CondCore/PopCon/interface/PopConSourceHandler.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include <string>

namespace coral {
  class ISessionProxy;
  class TimeStamp;
}
class DTHVAbstractCheck;

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTHVStatusHandler: public popcon::PopConSourceHandler<DTHVStatus> {

 public:

  /** Constructor
   */
  DTHVStatusHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTHVStatusHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

 private:

  typedef std::pair<long long int,float> timedMeasurement;
  typedef std::pair<int,float> channelValue;

  void checkNewData();

  void getChannelMap();
  void getLayerSplit();
  void getChannelSplit();
  void dumpHVAliases();

  void createSnapshot();
  int recoverSnapshot( std::map<int,timedMeasurement>& snapshotValues );
  cond::Time_t recoverLastTime();
  void   dumpSnapshot( const coral::TimeStamp& time,
                       std::map<int,timedMeasurement>& snapshotValues );
  void updateHVStatus();
  int  checkForPeriod( cond::Time_t condSince,
                       cond::Time_t condUntil,
                       std::map<int,timedMeasurement>& snapshotValues,
                       int& missingChannels,
                       bool copyOffline );

  void copyHVData( std::map<int,timedMeasurement>& snapshotValues );
  DTHVStatus* offlineList( std::map<int,timedMeasurement>& snapshotValues );
  void setFlags( DTHVStatus* hv, int type, int rawId, float value );
  void setChannelFlag( DTHVStatus* hv,
                       int whe, int sta, int sec, int qua, int lay, int l_p,
                       char cht, int err );

//  int checkCurrentStatus( int chan, int type, float value );
  int checkStatusChange( int type, float oldValue, float newValue );

  static coral::TimeStamp coralTime( const  cond::Time_t&    time );
  static  cond::Time_t     condTime( const coral::TimeStamp& time );
  static  cond::Time_t     condTime( long long int           time );

  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  std::string bufferConnect;
  DTHVStatus* lastStatus;

  int ySince;
  int mSince;
  int dSince;
  int hSince;
  int pSince;
  int sSince;
  int yUntil;
  int mUntil;
  int dUntil;
  int hUntil;
  int pUntil;
  int sUntil;
  long long int bwdTime;
  long long int fwdTime;
  long long int minTime;

  DTHVAbstractCheck* hvChecker;
//  float* minHV;
//  float* maxHV;
//  float maxCurrent;

  cond::Time_t procSince;
  cond::Time_t procUntil;
  cond::Time_t lastFound;
  cond::Time_t nextFound;
  cond::Time_t timeLimit;
  long long int lastStamp;
  int maxPayload;

  coral::ISessionProxy* omds_s_proxy;
  coral::ISessionProxy* buff_s_proxy;

  std::string mapVersion;
  std::string splitVersion;
  std::map<int,int> aliasMap;
  std::map<int,int> layerMap;
  std::map<int,int> laySplit;
  std::map< int, std::vector<int>* > channelSplit;

};


#endif // DTHVStatusHandler_H






