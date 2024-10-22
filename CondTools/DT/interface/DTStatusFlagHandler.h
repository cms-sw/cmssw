#ifndef DTStatusFlagHandler_H
#define DTStatusFlagHandler_H
/** \class DTStatusFlagHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:27 $
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
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTStatusFlagHandler : public popcon::PopConSourceHandler<DTStatusFlag> {
public:
  /** Constructor
   */
  DTStatusFlagHandler(const edm::ParameterSet& ps);

  /** Destructor
   */
  ~DTStatusFlagHandler() override;

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

#endif  // DTStatusFlagHandler_H
