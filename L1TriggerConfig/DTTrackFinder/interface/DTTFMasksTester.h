//-------------------------------------------------
//
/**  \class DTTFMasksTester
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
#ifndef DTTFMasksTester_h
#define DTTFMasksTester_h

#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"


class DTTFMasksTester : public edm::EDAnalyzer {
 public:

  DTTFMasksTester(const edm::ParameterSet& ps);

  ~DTTFMasksTester();
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

 private:

};

#endif
