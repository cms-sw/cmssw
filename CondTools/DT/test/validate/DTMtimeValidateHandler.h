#ifndef DTMtimeValidateHandler_H
#define DTMtimeValidateHandler_H
/** \class DTMtimeValidateHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2008/03/21 15:12:29 $
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
#include "CondFormats/DTObjects/interface/DTMtime.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTMtimeValidateHandler: public popcon::PopConSourceHandler<DTMtime> {

 public:

  /** Constructor
   */
  DTMtimeValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTMtimeValidateHandler();

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


#endif // DTMtimeValidateHandler_H






