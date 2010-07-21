#ifndef DTHVStatusHandler_H
#define DTHVStatusHandler_H
/** \class DTHVStatusHandler
 *
 *  Description: Class to copy HV status via PopCon
 *
 *
 *  $Date: 2010/04/02 10:30:59 $
 *  $Revision: 1.5 $
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
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include <string>

namespace coral {
  class TimeStamp;
}
#include "CondTools/DT/interface/DTHVAbstractCheck.h"

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

  typedef DTHVAbstractCheck::timedMeasurement timedMeasurement;
  typedef std::pair<int,float> channelValue;

  void checkNewData();

  void getChannelMap();
  void getLayerSplit();
  void getChannelSplit();
  void dumpHVAliases();

  void createSnapshot();
  int recoverSnapshot();
  cond::Time_t recoverLastTime();
  void   dumpSnapshot( const coral::TimeStamp& time );
  void updateHVStatus();
  int  checkForPeriod( cond::Time_t condSince,
                       cond::Time_t condUntil,
                       int& missingChannels,
                       bool copyOffline );

  void copyHVData();
  DTHVStatus* offlineList();
  void setFlags( DTHVStatus* hv, int rawId, int flag );
  void setChannelFlag( DTHVStatus* hv,
                       int whe, int sta, int sec, int qua, int lay, int l_p,
                       char cht, int err );

  int checkStatusChange( int type, float oldValue, float newValue );
  void filterData();

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

  std::map<int,timedMeasurement> snapshotValues;
  DTHVAbstractCheck* hvChecker;

  cond::Time_t procSince;
  cond::Time_t procUntil;
  cond::Time_t lastFound;
  cond::Time_t nextFound;
  cond::Time_t timeLimit;
  long long int lastStamp;
  int maxPayload;
  
  cond::DbConnection connection;
  cond::DbSession omds_session;
  cond::DbSession buff_session;

  std::string mapVersion;
  std::string splitVersion;
  std::map<int,int> aliasMap;
  std::map<int,int> layerMap;
  std::map<int,int> laySplit;
  std::map< int, std::vector<int>* > channelSplit;
  std::vector< std::pair<DTHVStatus*, cond::Time_t> > tmpContainer;
  bool switchOff;

};


#endif // DTHVStatusHandler_H






