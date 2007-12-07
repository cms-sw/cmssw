#ifndef DTRangeT0Analyzer_H
#define DTRangeT0Analyzer_H
/** \class DTRangeT0Analyzer
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
#include "CondFormats/DTObjects/interface/DTRangeT0.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTRangeT0Analyzer: public popcon::PopConAnalyzer<DTRangeT0> {

 public:

  /** Constructor
   */
  DTRangeT0Analyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTRangeT0Analyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTRangeT0Analyzer_H






