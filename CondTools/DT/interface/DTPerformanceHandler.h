#ifndef DTPerformanceHandler_H
#define DTPerformanceHandler_H
/** \class DTPerformanceHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2008/02/15 18:15:02 $
 *  $Revision: 1.3 $
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
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTPerformanceHandler: public popcon::PopConSourceHandler<DTPerformance> {

 public:

  /** Constructor
   */
  DTPerformanceHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTPerformanceHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

 private:

  std::string dataTag;
  std::string fileName;
  unsigned int runNumber;

};


#endif // DTPerformanceHandler_H






