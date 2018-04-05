#ifndef DTKeyedConfigDBInit_H
#define DTKeyedConfigDBInit_H
/** \class DTKeyedConfigDBInit
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

class DTKeyedConfigDBInit: public edm::EDAnalyzer {

 public:

  /** Constructor
   */
  explicit DTKeyedConfigDBInit( const edm::ParameterSet& ps );

  /** Destructor
   */
  ~DTKeyedConfigDBInit() override;

  /** Operations
   */
  /// 
  void beginJob() override;
  void analyze( const edm::Event& e, const edm::EventSetup& c ) override;
  void endJob() override;

 private:

  std::string container;
  std::string iov;

};


#endif // DTKeyedConfigDBInit_H

