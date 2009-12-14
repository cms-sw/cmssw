#ifndef DTConfigDBInit_H
#define DTConfigDBInit_H
/** \class DTConfigDBInit
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:56 $
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

class DTConfigDBInit: public edm::EDAnalyzer {

 public:

  /** Constructor
   */
  explicit DTConfigDBInit( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTConfigDBInit();

  /** Operations
   */
  /// 
  virtual void beginJob();
  virtual void analyze( const edm::Event& e, const edm::EventSetup& c );

 private:

  std::string name;
  std::string contact;
  std::string catalog;
  std::string authPath;

};


#endif // DTConfigDBInit_H






