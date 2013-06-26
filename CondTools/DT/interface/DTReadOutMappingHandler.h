#ifndef DTReadOutMappingHandler_H
#define DTReadOutMappingHandler_H
/** \class DTReadOutMappingHandler
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
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

//---------------
// C++ Headers --
//---------------
#include <string>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTReadOutMappingHandler: public popcon::PopConSourceHandler<DTReadOutMapping> {

 public:

  /** Constructor
   */
  DTReadOutMappingHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTReadOutMappingHandler();

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


#endif // DTReadOutMappingHandler_H






