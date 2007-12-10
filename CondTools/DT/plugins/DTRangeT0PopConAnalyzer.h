#ifndef DTRangeT0PopConAnalyzer_H
#define DTRangeT0PopConAnalyzer_H
/** \class DTRangeT0PopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:13:09 $
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
#include "CondFormats/DTObjects/interface/DTRangeT0.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTRangeT0PopConAnalyzer: public popcon::PopConAnalyzer<DTRangeT0> {

 public:

  /** Constructor
   */
  DTRangeT0PopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTRangeT0PopConAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTRangeT0PopConAnalyzer_H






