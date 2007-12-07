#ifndef DTReadOutMappingAnalyzer_H
#define DTReadOutMappingAnalyzer_H
/** \class DTReadOutMappingAnalyzer
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
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTReadOutMappingAnalyzer: public popcon::PopConAnalyzer<DTReadOutMapping> {

 public:

  /** Constructor
   */
  DTReadOutMappingAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTReadOutMappingAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTReadOutMappingAnalyzer_H






