#ifndef DTLVStatusValidateHandler_H
#define DTLVStatusValidateHandler_H
/** \class DTLVStatusValidateHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2010/01/20 18:20:09 $
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

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTLVStatusValidateHandler: public popcon::PopConSourceHandler<DTLVStatus> {

 public:

  /** Constructor
   */
  DTLVStatusValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTLVStatusValidateHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

 private:

  int firstRun;
  int  lastRun;
  std::string dataVersion;
  std::string dataFileName;
  std::string elogFileName;
  void addNewObject( int runNumber );

};


#endif // DTLVStatusValidateHandler_H






