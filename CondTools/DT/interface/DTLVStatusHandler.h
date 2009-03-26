#ifndef DTLVStatusHandler_H
#define DTLVStatusHandler_H
/** \class DTLVStatusHandler
 *
 *  Description: Class to copy CCB DCS-status via PopCon
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
#include "CondFormats/DTObjects/interface/DTLVStatus.h"
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

class DTLVStatusHandler: public popcon::PopConSourceHandler<DTLVStatus> {

 public:

  /** Constructor
   */
  DTLVStatusHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTLVStatusHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

 private:

  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  std::string bufferConnect;
  DTLVStatus* ccbStatus;

  coral::ISessionProxy* omds_s_proxy;
  coral::ISessionProxy* buff_s_proxy;

};


#endif // DTLVStatusHandler_H






