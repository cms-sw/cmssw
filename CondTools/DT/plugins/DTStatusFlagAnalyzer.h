#ifndef DTStatusFlagAnalyzer_H
#define DTStatusFlagAnalyzer_H
/** \class DTStatusFlagAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/11/24 12:29:54 $
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
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTStatusFlagAnalyzer: public popcon::PopConAnalyzer<DTStatusFlag> {

 public:

  /** Constructor
   */
  DTStatusFlagAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTStatusFlagAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTStatusFlagAnalyzer_H






