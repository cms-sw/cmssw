#ifndef DTHVStatusHandler_H
#define DTHVStatusHandler_H
/** \class DTHVStatusHandler
 *
 *  Description: Class to copy HV status via PopCon
 *
 *
 *  $Date: 2008/11/25 11:00:00 $
 *  $Revision: 1.1 $
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
}

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

  void getChannelMap();
  void getLayerSplit();

//  void updateHVStatus( int run );
  void updateHVStatus( cond::Time_t time );
  void dumpHVAliases();

  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  std::string bufferConnect;
  DTHVStatus* hvStatus;

  int ySince;
  int mSince;
  int dSince;
  int yUntil;
  int mUntil;
  int dUntil;

  coral::ISessionProxy* omds_s_proxy;
  coral::ISessionProxy* buff_s_proxy;

  std::string mapVersion;
  std::map<int,int> aliasMap;
  std::map<int,int> laySplit;

};


#endif // DTHVStatusHandler_H






