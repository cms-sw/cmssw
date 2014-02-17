#ifndef DTT0Handler_H
#define DTT0Handler_H
/** \class DTT0Handler
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
#include "CondFormats/DTObjects/interface/DTT0.h"

//---------------
// C++ Headers --
//---------------
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTT0Handler: public popcon::PopConSourceHandler<DTT0> {

 public:

  /** Constructor
   */
  DTT0Handler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTT0Handler();

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


#endif // DTT0Handler_H






