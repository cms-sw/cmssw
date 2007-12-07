#ifndef DTCCBConfigAnalyzer_H
#define DTCCBConfigAnalyzer_H
/** \class DTCCBConfigAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/11/24 12:29:52 $
 *  $Revision: 1.1.2.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "CondCore/PopCon/interface/PopConAnalyzer.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTCCBConfigAnalyzer: public popcon::PopConAnalyzer<DTCCBConfig> {

 public:

  /** Constructor
   */
  DTCCBConfigAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTCCBConfigAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  std::string offlineAuthentication;
  std::string listToken;

};


#endif // DTCCBConfigAnalyzer_H






