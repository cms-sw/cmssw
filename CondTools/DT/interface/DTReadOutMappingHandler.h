#ifndef DTReadOutMappingHandler_H
#define DTReadOutMappingHandler_H
/** \class DTReadOutMappingHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:25 $
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
  ~DTReadOutMappingHandler() override;

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


#endif // DTReadOutMappingHandler_H






