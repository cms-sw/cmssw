#ifndef DTDeadFlagAnalyzer_H
#define DTDeadFlagAnalyzer_H
/** \class DTDeadFlagAnalyzer
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
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTDeadFlagAnalyzer: public popcon::PopConAnalyzer<DTDeadFlag> {

 public:

  /** Constructor
   */
  DTDeadFlagAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTDeadFlagAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};

#endif // DTDeadFlagAnalyzer_H






