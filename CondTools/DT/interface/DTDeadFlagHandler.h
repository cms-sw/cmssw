#ifndef DTDeadFlagHandler_H
#define DTDeadFlagHandler_H
/** \class DTDeadFlagHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:19 $
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
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTDeadFlagHandler : public popcon::PopConSourceHandler<DTDeadFlag> {
public:
  /** Constructor
   */
  DTDeadFlagHandler(const edm::ParameterSet& ps);

  /** Destructor
   */
  ~DTDeadFlagHandler() override;

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

#endif  // DTDeadFlagHandler_H
