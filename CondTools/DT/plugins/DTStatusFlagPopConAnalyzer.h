#ifndef DTStatusFlagPopConAnalyzer_H
#define DTStatusFlagPopConAnalyzer_H
/** \class DTStatusFlagPopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:13:13 $
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
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTStatusFlagPopConAnalyzer: public popcon::PopConAnalyzer<DTStatusFlag> {

 public:

  /** Constructor
   */
  DTStatusFlagPopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTStatusFlagPopConAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTStatusFlagPopConAnalyzer_H






