#ifndef DTRangeT0ValidateHandler_H
#define DTRangeT0ValidateHandler_H
/** \class DTRangeT0ValidateHandler
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
#include "CondFormats/DTObjects/interface/DTRangeT0.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTRangeT0ValidateHandler: public popcon::PopConSourceHandler<DTRangeT0> {

 public:

  /** Constructor
   */
  DTRangeT0ValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTRangeT0ValidateHandler();

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


#endif // DTRangeT0ValidateHandler_H






