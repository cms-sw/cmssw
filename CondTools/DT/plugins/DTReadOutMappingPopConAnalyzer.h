#ifndef DTReadOutMappingPopConAnalyzer_H
#define DTReadOutMappingPopConAnalyzer_H
/** \class DTReadOutMappingPopConAnalyzer
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:13:12 $
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
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTReadOutMappingPopConAnalyzer: public popcon::PopConAnalyzer<DTReadOutMapping> {

 public:

  /** Constructor
   */
  DTReadOutMappingPopConAnalyzer( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTReadOutMappingPopConAnalyzer();

  /** Operations
   */
  /// 
  void initSource( const edm::Event& evt, const edm::EventSetup& est );

 private:

  std::string dataTag;
  std::string fileName;

};


#endif // DTReadOutMappingPopConAnalyzer_H






