//-------------------------------------------------
//
/**  \class DTTFParametersTester
 *
 *   L1 DT Track Finder Parameters Tester
 *
 *
 *   $Date: 2009/05/04 09:26:09 $
 *   $Revision: 1.1 $
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTTFParametersTester_h
#define DTTFParametersTester_h

#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"


class DTTFParametersTester : public edm::EDAnalyzer {
 public:

  DTTFParametersTester(const edm::ParameterSet& ps);

  ~DTTFParametersTester();
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

 private:

};

#endif
