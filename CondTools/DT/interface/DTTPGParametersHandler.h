#ifndef DTTPGParametersHandler_H
#define DTTPGParametersHandler_H
/** \class DTTPGParametersHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2009/01/27 12:07:41 $
 *  $Revision: 1.1.2.1 $
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

class DTTPGParametersHandler : public popcon::PopConSourceHandler<DTTPGParameters> {
public:
  /** Constructor
   */
  DTTPGParametersHandler(const edm::ParameterSet& ps);

  /** Destructor
   */
  ~DTTPGParametersHandler() override;

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

#endif  // DTTPGParametersHandler_H
