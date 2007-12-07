#ifndef DTMtimeAnalyzer_H
#define DTMtimeAnalyzer_H
/** \class DTMtimeAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/11/24 12:29:53 $
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
#include "CondFormats/DTObjects/interface/DTMtime.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTMtimeAnalyzer: public popcon::PopConAnalyzer<DTMtime> {

 public:

  /** Constructor
   */
  DTMtimeAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTMtimeAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTMtimeAnalyzer_H






