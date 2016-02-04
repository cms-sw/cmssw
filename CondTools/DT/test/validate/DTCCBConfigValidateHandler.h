#ifndef DTCCBConfigValidateHandler_H
#define DTCCBConfigValidateHandler_H
/** \class DTCCBConfigValidateHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2010/01/20 17:29:26 $
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
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTCCBConfigValidateHandler: public popcon::PopConSourceHandler<DTCCBConfig> {

 public:

  /** Constructor
   */
  DTCCBConfigValidateHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTCCBConfigValidateHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

 private:

  int firstRun;
  int  lastRun;
  std::string dataVersion;
  std::string dataFileName;
  std::string elogFileName;
  void addNewObject( int runNumber );
  static bool cfrDiff( const std::vector<int>& l_conf,
                       const std::vector<int>& r_conf );

};


#endif // DTCCBConfigValidateHandler_H






