#ifndef DTStatusFlagHandler_H
#define DTStatusFlagHandler_H
/** \class DTStatusFlagHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2008/02/15 18:15:03 $
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
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

//---------------
// C++ Headers --
//---------------
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTStatusFlagHandler: public popcon::PopConSourceHandler<DTStatusFlag> {

 public:

  /** Constructor
   */
  DTStatusFlagHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTStatusFlagHandler();

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


#endif // DTStatusFlagHandler_H






