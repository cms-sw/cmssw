#ifndef DTDeadFlagHandler_H
#define DTDeadFlagHandler_H
/** \class DTDeadFlagHandler
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
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"

//---------------
// C++ Headers --
//---------------
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTDeadFlagHandler: public popcon::PopConSourceHandler<DTDeadFlag> {

 public:

  /** Constructor
   */
  DTDeadFlagHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTDeadFlagHandler();

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


#endif // DTDeadFlagHandler_H

