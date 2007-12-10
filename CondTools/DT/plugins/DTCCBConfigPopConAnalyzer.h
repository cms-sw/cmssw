#ifndef DTCCBConfigPopConAnalyzer_H
#define DTCCBConfigPopConAnalyzer_H
/** \class DTCCBConfigPopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:47 $
 *  $Revision: 1.2 $
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

class DTCCBConfigPopConAnalyzer: public popcon::PopConAnalyzer<DTCCBConfig> {

 public:

  /** Constructor
   */
  DTCCBConfigPopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTCCBConfigPopConAnalyzer();

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


#endif // DTCCBConfigPopConAnalyzer_H






