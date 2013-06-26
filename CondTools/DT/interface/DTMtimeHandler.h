#ifndef DTMtimeHandler_H
#define DTMtimeHandler_H
/** \class DTMtimeHandler
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
#include "CondFormats/DTObjects/interface/DTMtime.h"

//---------------
// C++ Headers --
//---------------
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTMtimeHandler: public popcon::PopConSourceHandler<DTMtime> {

 public:

  /** Constructor
   */
  DTMtimeHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTMtimeHandler();

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


#endif // DTMtimeHandler_H






