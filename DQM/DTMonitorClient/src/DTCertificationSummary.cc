
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/02/17 16:22:37 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */


#include "DQM/DTMonitorClient/src/DTCertificationSummary.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using namespace std;
using namespace edm;



DTCertificationSummary::DTCertificationSummary(const ParameterSet& pset) {}




DTCertificationSummary::~DTCertificationSummary() {}



void DTCertificationSummary::beginJob(const EventSetup& setup){
  // get the DQMStore
  theDbe = Service<DQMStore>().operator->();
  
  // book the ME
  theDbe->setCurrentFolder("DT/EventInfo/CertificationContents");
  // global fraction
  totalCertFraction = theDbe->bookFloat("DTCertificationSummary");  
  totalCertFraction->Fill(-1);
  // Wheel "fractions" -> will be 0 or 1
  for(int wheel = -2; wheel != 3; ++wheel) {
    stringstream streams;
    streams << "DT_Wheel" << wheel;
    certFractions[wheel] = theDbe->bookFloat(streams.str());
    certFractions[wheel]->Fill(-1);
  }

  //

}



void DTCertificationSummary::beginLuminosityBlock(const LuminosityBlock& lumi, const  EventSetup& setup) {
}




void DTCertificationSummary::endLuminosityBlock(const LuminosityBlock&  lumi, const  EventSetup& setup){}



void DTCertificationSummary::endJob() {}



void DTCertificationSummary::analyze(const Event& event, const EventSetup& setup){}



