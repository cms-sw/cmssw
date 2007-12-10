#ifndef DTTtrigPopConAnalyzer_H
#define DTTtrigPopConAnalyzer_H
/** \class DTTtrigPopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:13:16 $
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
#include "CondFormats/DTObjects/interface/DTTtrig.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTtrigPopConAnalyzer: public popcon::PopConAnalyzer<DTTtrig> {

 public:

  /** Constructor
   */
  DTTtrigPopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTTtrigPopConAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTTtrigPopConAnalyzer_H






