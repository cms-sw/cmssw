#ifndef DTTtrigValidateHandler_H
#define DTTtrigValidateHandler_H
/** \class DTTtrigValidateHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2008/03/25 16:19:57 $
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
#include "CondFormats/DTObjects/interface/DTTtrig.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTtrigValidateHandler: public popcon::PopConSourceHandler<DTTtrig> {

 public:

  /** Constructor
   */
  DTTtrigValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTTtrigValidateHandler();

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


#endif // DTTtrigValidateHandler_H






