/** \file
 *
 *  $Date: 2007/09/24 13:54:37 $
 *  $Revision: 1.1 $
 *  \author A. Tumanov - Rice
 */

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
class CSCCorrelatedLCTDigi;

class CSCDigiToPattern : public edm::EDAnalyzer {
public:
  explicit CSCDigiToPattern(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);

  //virtual void endJob();
private:
  // variables persistent across events should be declared here.
  //
};



