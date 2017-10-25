#ifndef DTT0Handler_H
#define DTT0Handler_H
/** \class DTT0Handler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:29 $
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
  ~DTT0Handler() override;

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


#endif // DTT0Handler_H






