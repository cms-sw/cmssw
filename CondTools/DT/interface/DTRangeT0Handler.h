#ifndef DTRangeT0Handler_H
#define DTRangeT0Handler_H
/** \class DTRangeT0Handler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:22 $
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
#include "CondFormats/DTObjects/interface/DTRangeT0.h"

//---------------
// C++ Headers --
//---------------
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTRangeT0Handler: public popcon::PopConSourceHandler<DTRangeT0> {

 public:

  /** Constructor
   */
  DTRangeT0Handler( const edm::ParameterSet& ps );

  /** Destructor
   */
  ~DTRangeT0Handler() override;

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


#endif // DTRangeT0Handler_H






