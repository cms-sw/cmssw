#ifndef DTLVStatusHandler_H
#define DTLVStatusHandler_H
/** \class DTLVStatusHandler
 *
 *  Description: Class to copy CCB DCS-status via PopCon
 *
 *
 *  $Date: 2010/07/21 16:06:53 $
 *  $Revision: 1.2 $
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
#include "CondCore/CondDB/interface/Session.h"
#include "CondFormats/DTObjects/interface/DTLVStatus.h"
#include <string>


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

  cond::persistency::Session omds_session;
  cond::persistency::Session buff_session;

};


#endif // DTLVStatusHandler_H






