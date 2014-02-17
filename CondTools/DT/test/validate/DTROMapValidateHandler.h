#ifndef DTROMapValidateHandler_H
#define DTROMapValidateHandler_H
/** \class DTROMapValidateHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2008/03/25 16:19:57 $
 *  $Revision: 1.1 $
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


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTROMapValidateHandler: public popcon::PopConSourceHandler<DTReadOutMapping> {

 public:

  /** Constructor
   */
  DTROMapValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTROMapValidateHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

 private:

  std::string dataVersion;
  std::string dataFileName;
  std::string elogFileName;

};


#endif // DTROMapValidateHandler_H






