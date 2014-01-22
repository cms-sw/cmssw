#ifndef DTKeyedConfigDBDump_H
#define DTKeyedConfigDBDump_H
/** \class DTKeyedConfigDBDump
 *
 *  Description: 
 *
 *
 *  $Date: 2010/03/18 16:07:59 $
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

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/DTObjects/interface/Serialization.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTKeyedConfigDBDump: public edm::EDAnalyzer {

 public:

  /** Constructor
   */
  explicit DTKeyedConfigDBDump( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTKeyedConfigDBDump();

  /** Operations
   */
  /// 
  virtual void beginJob();
  virtual void analyze( const edm::Event& e, const edm::EventSetup& c );

 private:

};


#endif // DTKeyedConfigDBDump_H

