#ifndef DTTPGParametersValidateHandler_H
#define DTTPGParametersValidateHandler_H
/** \class DTTPGParametersValidateHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2009/12/22 16:30:00 $
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
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTPGParametersValidateHandler : public popcon::PopConSourceHandler<DTTPGParameters> {
public:
  /** Constructor
   */
  DTTPGParametersValidateHandler(const edm::ParameterSet& ps);

  /** Destructor
   */
  virtual ~DTTPGParametersValidateHandler();

  /** Operations
   */
  ///
  void getNewObjects();
  std::string id() const;

private:
  int firstRun;
  int lastRun;
  std::string dataVersion;
  std::string dataFileName;
  std::string elogFileName;
  void addNewObject(int runNumber);
};

#endif  // DTTPGParametersValidateHandler_H
