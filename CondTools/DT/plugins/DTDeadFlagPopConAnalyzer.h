#ifndef DTDeadFlagPopConAnalyzer_H
#define DTDeadFlagPopConAnalyzer_H
/** \class DTDeadFlagPopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:13:00 $
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
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTDeadFlagPopConAnalyzer: public popcon::PopConAnalyzer<DTDeadFlag> {

 public:

  /** Constructor
   */
  DTDeadFlagPopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTDeadFlagPopConAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};

#endif // DTDeadFlagPopConAnalyzer_H






