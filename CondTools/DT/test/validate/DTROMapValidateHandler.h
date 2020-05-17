#ifndef DTROMapValidateHandler_H
#define DTROMapValidateHandler_H
/** \class DTROMapValidateHandler
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
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTROMapValidateHandler : public popcon::PopConSourceHandler<DTReadOutMapping> {
public:
  /** Constructor
   */
  DTROMapValidateHandler(const edm::ParameterSet& ps);

  /** Destructor
   */
  ~DTROMapValidateHandler() override;

  /** Operations
   */
  ///
  void getNewObjects() override;
  std::string id() const override;

private:
  std::string dataVersion;
  std::string dataFileName;
  std::string elogFileName;
};

#endif  // DTROMapValidateHandler_H
