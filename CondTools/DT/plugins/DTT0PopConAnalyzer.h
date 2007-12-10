#ifndef DTT0PopConAnalyzer_H
#define DTT0PopConAnalyzer_H
/** \class DTT0PopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:13:15 $
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
#include "CondFormats/DTObjects/interface/DTT0.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTT0PopConAnalyzer: public popcon::PopConAnalyzer<DTT0> {

 public:

  /** Constructor
   */
  DTT0PopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTT0PopConAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTT0PopConAnalyzer_H






