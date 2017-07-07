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
  ~DTKeyedConfigDBDump() override;

  /** Operations
   */
  /// 
  void beginJob() override;
  void analyze( const edm::Event& e, const edm::EventSetup& c ) override;

 private:

};


#endif // DTKeyedConfigDBDump_H

