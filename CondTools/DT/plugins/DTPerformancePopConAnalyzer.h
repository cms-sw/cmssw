#ifndef DTPerformancePopConAnalyzer_H
#define DTPerformancePopConAnalyzer_H
/** \class DTPerformancePopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:13:06 $
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
#include "CondFormats/DTObjects/interface/DTPerformance.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTPerformancePopConAnalyzer: public popcon::PopConAnalyzer<DTPerformance> {

 public:

  /** Constructor
   */
  DTPerformancePopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTPerformancePopConAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTPerformancePopConAnalyzer_H






