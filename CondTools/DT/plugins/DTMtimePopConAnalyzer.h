#ifndef DTMtimePopConAnalyzer_H
#define DTMtimePopConAnalyzer_H
/** \class DTMtimePopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:13:02 $
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
#include "CondFormats/DTObjects/interface/DTMtime.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTMtimePopConAnalyzer: public popcon::PopConAnalyzer<DTMtime> {

 public:

  /** Constructor
   */
  DTMtimePopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTMtimePopConAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTMtimePopConAnalyzer_H






