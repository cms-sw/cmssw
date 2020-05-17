#ifndef DTStatusFlagValidateHandler_H
#define DTStatusFlagValidateHandler_H
/** \class DTStatusFlagValidateHandler
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
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTStatusFlagValidateHandler : public popcon::PopConSourceHandler<DTStatusFlag> {
public:
  /** Constructor
   */
  DTStatusFlagValidateHandler(const edm::ParameterSet& ps);

  /** Destructor
   */
  ~DTStatusFlagValidateHandler() override;

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

#endif  // DTStatusFlagValidateHandler_H
