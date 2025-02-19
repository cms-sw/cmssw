#ifndef DTTtrigHandler_H
#define DTTtrigHandler_H
/** \class DTTtrigHandler
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
#include "CondFormats/DTObjects/interface/DTTtrig.h"

//---------------
// C++ Headers --
//---------------
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTtrigHandler: public popcon::PopConSourceHandler<DTTtrig> {

 public:

  /** Constructor
   */
  DTTtrigHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTTtrigHandler();

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


#endif // DTTtrigHandler_H






