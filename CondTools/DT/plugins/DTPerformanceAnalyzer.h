#ifndef DTPerformanceAnalyzer_H
#define DTPerformanceAnalyzer_H
/** \class DTPerformanceAnalyzer
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
#include "CondFormats/DTObjects/interface/DTPerformance.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTPerformanceAnalyzer: public popcon::PopConAnalyzer<DTPerformance> {

 public:

  /** Constructor
   */
  DTPerformanceAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTPerformanceAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTPerformanceAnalyzer_H






