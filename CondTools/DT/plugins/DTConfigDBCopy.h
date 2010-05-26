#ifndef DTConfigDBCopy_H
#define DTConfigDBCopy_H
/** \class DTConfigDBCopy
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:50 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "FWCore/Framework/interface/EDAnalyzer.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigDBCopy: public edm::EDAnalyzer {

 public:

  /** Constructor
   */
  explicit DTConfigDBCopy( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTConfigDBCopy();

  /** Operations
   */
  /// 
  virtual void beginJob();
  virtual void analyze( const edm::Event& e, const edm::EventSetup& c );

 private:

  std::string sourceContact;
  std::string sourceCatalog;
  std::string sourceToken;
  std::string sourceAuthPath;
  std::string targetContact;
  std::string targetCatalog;
  std::string targetToken;
  std::string targetAuthPath;

};


#endif // DTConfigDBCopy_H






