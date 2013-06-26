//-------------------------------------------------
//
/**  \class DTQualPatternLutTester
 *
 *   L1 DT Track Finder Quality Pattern Lut Tester
 *
 *
 *   $Date: 2009/05/04 09:26:09 $
 *   $Revision: 1.1 $
 *
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTQualPatternLutTester_h
#define DTQualPatternLutTester_h

#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1MuDTQualPatternLut.h"
#include "CondFormats/DataRecord/interface/L1MuDTQualPatternLutRcd.h"


class DTQualPatternLutTester : public edm::EDAnalyzer {
 public:

  DTQualPatternLutTester(const edm::ParameterSet& ps);

  ~DTQualPatternLutTester();
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

 private:

};

#endif
