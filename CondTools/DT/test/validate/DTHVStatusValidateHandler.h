#ifndef DTHVStatusValidateHandler_H
#define DTHVStatusValidateHandler_H
/** \class DTHVStatusValidateHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2010/01/20 17:29:26 $
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

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTHVStatusValidateHandler: public popcon::PopConSourceHandler<DTHVStatus> {

 public:

  /** Constructor
   */
  DTHVStatusValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTHVStatusValidateHandler();

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


#endif // DTHVStatusValidateHandler_H






