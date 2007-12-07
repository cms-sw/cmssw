#ifndef DTT0Analyzer_H
#define DTT0Analyzer_H
/** \class DTT0Analyzer
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
#include "CondFormats/DTObjects/interface/DTT0.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTT0Analyzer: public popcon::PopConAnalyzer<DTT0> {

 public:

  /** Constructor
   */
  DTT0Analyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTT0Analyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTT0Analyzer_H






