#ifndef DTT0ValidateHandler_H
#define DTT0ValidateHandler_H
/** \class DTT0ValidateHandler
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
#include "CondFormats/DTObjects/interface/DTT0.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTT0ValidateHandler: public popcon::PopConSourceHandler<DTT0> {

 public:

  /** Constructor
   */
  DTT0ValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTT0ValidateHandler();

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


#endif // DTT0ValidateHandler_H






