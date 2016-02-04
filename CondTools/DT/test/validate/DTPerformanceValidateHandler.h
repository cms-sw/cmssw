#ifndef DTPerformanceValidateHandler_H
#define DTPerformanceValidateHandler_H
/** \class DTPerformanceValidateHandler
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
#include "CondFormats/DTObjects/interface/DTPerformance.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTPerformanceValidateHandler: public popcon::PopConSourceHandler<DTPerformance> {

 public:

  /** Constructor
   */
  DTPerformanceValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTPerformanceValidateHandler();

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


#endif // DTPerformanceValidateHandler_H






