#ifndef DTTPGParametersHandler_H
#define DTTPGParametersHandler_H
/** \class DTTPGParametersHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2009/02/16 13:21:15 $
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
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"

//---------------
// C++ Headers --
//---------------
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTPGParametersHandler: public popcon::PopConSourceHandler<DTTPGParameters> {

 public:

  /** Constructor
   */
  DTTPGParametersHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTTPGParametersHandler();

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


#endif // DTTPGParametersHandler_H






