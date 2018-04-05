#ifndef DTPerformanceHandler_H
#define DTPerformanceHandler_H
/** \class DTPerformanceHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:22 $
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
  ~DTPerformanceHandler() override;

  /** Operations
   */
  /// 
  void getNewObjects() override;
  std::string id() const override;

 private:

  std::string dataTag;
  std::string fileName;
  unsigned int runNumber;

};


#endif // DTPerformanceHandler_H






