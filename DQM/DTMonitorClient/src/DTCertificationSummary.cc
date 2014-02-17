
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/22 15:32:04 $
 *  $Revision: 1.5 $
 *  \author G. Cerminara - INFN Torino
 */


#include "DQM/DTMonitorClient/src/DTCertificationSummary.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;



DTCertificationSummary::DTCertificationSummary(const ParameterSet& pset) {}




DTCertificationSummary::~DTCertificationSummary() {}



void DTCertificationSummary::beginJob(){
  // get the DQMStore
  theDbe = Service<DQMStore>().operator->();
  
  // book the ME
  theDbe->setCurrentFolder("DT/EventInfo");
  // global fraction
  totalCertFraction = theDbe->bookFloat("CertificationSummary");  
  totalCertFraction->Fill(-1);

  // certification map
  certMap = theDbe->book2D("CertificationSummaryMap","DT Certification Summary Map",12,1,13,5,-2,3);
  certMap->setAxisTitle("sector",1);
  certMap->setAxisTitle("wheel",2);

  theDbe->setCurrentFolder("DT/EventInfo/CertificationContents");
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




void DTCertificationSummary::beginRun(const Run& run, const  EventSetup& setup) {
}




void DTCertificationSummary::endLuminosityBlock(const LuminosityBlock&  lumi, const  EventSetup& setup){
}



void DTCertificationSummary::endRun(const Run& run, const  EventSetup& setup){
  // get the relevant summary histos
  MonitorElement* effSummary = theDbe->get("DT/05-ChamberEff/EfficiencyGlbSummary");
  MonitorElement* resSummary = theDbe->get("DT/02-Segments/ResidualsGlbSummary");
  MonitorElement* segQualSummary = theDbe->get("DT/02-Segments/segmentSummary");

  // check that all needed histos are there
  if(effSummary == 0 || resSummary == 0 || segQualSummary == 0) {
    LogWarning("DQM|DTMonitorClient|DTCertificationSummary") << "*** Warning: not all needed summaries are present!" << endl;
    return;
  }

  // reset the MEs
  totalCertFraction->Fill(0.);
  certFractions[-2]->Fill(0.);
  certFractions[-1]->Fill(0.);
  certFractions[-0]->Fill(0.);
  certFractions[1]->Fill(0.);
  certFractions[2]->Fill(0.);
  certMap->Reset();

  // loop over all sectors and wheels
  for(int wheel = -2; wheel != 3; ++wheel) {
    for(int sector = 1; sector != 13; ++sector) {
      double eff = effSummary->getBinContent(sector, wheel+3);
      double res = resSummary->getBinContent(sector, wheel+3);
      double segQual = segQualSummary->getBinContent(sector, wheel+3);
      
      double total = 0;
      if(segQual != 0) {
	total = min(res,eff);
      } else {
	total = eff;
      }
      
      certMap->Fill(sector,wheel,total);      
      // can use variable weight depending on the sector
      double weight = 1./12.;
      certFractions[wheel]->Fill(certFractions[wheel]->getFloatValue() + weight*total);
      double totalWeight = 1./60.;
      totalCertFraction->Fill(totalCertFraction->getFloatValue() + totalWeight*total);
    }
  }

}



void DTCertificationSummary::endJob() {}



void DTCertificationSummary::analyze(const Event& event, const EventSetup& setup){}



