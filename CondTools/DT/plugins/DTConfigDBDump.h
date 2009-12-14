#ifndef DTConfigDBDump_H
#define DTConfigDBDump_H
/** \class DTConfigDBDump
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:54 $
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
  virtual void beginJob();
  virtual void analyze( const edm::Event& e, const edm::EventSetup& c );

 private:

  std::string contact;
  std::string catalog;
  std::string token;
  std::string authPath;

};


#endif // DTConfigDBDump_H






