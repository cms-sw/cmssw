#ifndef DTConfigDBDump_H
#define DTConfigDBDump_H
/** \class DTConfigDBDump
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

class DTConfigDBDump: public edm::EDAnalyzer {

 public:

  /** Constructor
   */
  explicit DTConfigDBDump( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTConfigDBDump();

  /** Operations
   */
  /// 
  virtual void beginJob( edm::EventSetup const& c );
  virtual void analyze( const edm::Event& e, const edm::EventSetup& c );

 private:

  std::string contact;
  std::string catalog;
  std::string token;
  std::string authPath;

};


#endif // DTConfigDBDump_H






