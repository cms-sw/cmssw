#ifndef DTKeyedConfigDBInit_H
#define DTKeyedConfigDBInit_H
/** \class DTKeyedConfigDBInit
 *
 *  Description: 
 *
 *
 *  $Date: 2010/05/14 11:43:08 $
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

class DTKeyedConfigDBInit: public edm::EDAnalyzer {

 public:

  /** Constructor
   */
  explicit DTKeyedConfigDBInit( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTKeyedConfigDBInit();

  /** Operations
   */
  /// 
  virtual void beginJob();
  virtual void analyze( const edm::Event& e, const edm::EventSetup& c );
  virtual void endJob();

 private:

  std::string container;
  std::string iov;

};


#endif // DTKeyedConfigDBInit_H

