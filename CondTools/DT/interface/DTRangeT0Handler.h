#ifndef DTRangeT0Handler_H
#define DTRangeT0Handler_H
/** \class DTRangeT0Handler
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
  virtual ~DTRangeT0Handler();

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


#endif // DTRangeT0Handler_H






