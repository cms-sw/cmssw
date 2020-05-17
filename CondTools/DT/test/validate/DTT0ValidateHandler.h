#ifndef DTT0ValidateHandler_H
#define DTT0ValidateHandler_H
/** \class DTT0ValidateHandler
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
#include "CondFormats/DTObjects/interface/DTT0.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTT0ValidateHandler : public popcon::PopConSourceHandler<DTT0> {
public:
  /** Constructor
   */
  DTT0ValidateHandler(const edm::ParameterSet& ps);

  /** Destructor
   */
  ~DTT0ValidateHandler() override;

  /** Operations
   */
  ///
  void getNewObjects() override;
  std::string id() const override;

private:
  int firstRun;
  int lastRun;
  std::string dataVersion;
  std::string dataFileName;
  std::string elogFileName;
  void addNewObject(int runNumber);
};

#endif  // DTT0ValidateHandler_H
