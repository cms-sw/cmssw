/*
 *  See header file for a description of this class.
 *
 *  \author M. Pelliccioni - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 */

#include <DQM/DTMonitorClient/src/DTChamberEfficiencyClient.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMStore.h>

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

//two words about conventions: "All" histograms are those made for all segments
//while "Qual" histograms are those for segments with at least 12 hits

DTChamberEfficiencyClient::DTChamberEfficiencyClient(const ParameterSet& pSet)
{

  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "DTChamberEfficiencyClient: Constructor called";

  prescaleFactor = pSet.getUntrackedParameter<int>("diagnosticPrescale", 1);

}

DTChamberEfficiencyClient::~DTChamberEfficiencyClient()
{
   LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
     << "DTChamberEfficiencyClient: Destructor called";
}

void DTChamberEfficiencyClient::beginRun(const edm::Run& run, const edm::EventSetup& setup) {

  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(muonGeom);

}


void DTChamberEfficiencyClient::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                                                         edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & setup)
{
  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "DTChamberEfficiencyClient: endluminosityBlock";
}  

void DTChamberEfficiencyClient::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) 
{
  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "DTChamberEfficiencyClient: endRun";

  bookHistos(ibooker);  

  // reset the global summary
  globalEffSummary->Reset();

  //Loop over the wheels
  for(int wheel=-2;wheel<=2;wheel++){
    stringstream wheel_str; wheel_str << wheel;

    // Get the ME produced by EfficiencyTask Source
    // All means no selection on segments, Qual means segments with at least 12 hits
	
    MonitorElement* MECountAll = igetter.get("DT/05-ChamberEff/Task/hCountSectVsChamb_All_W" + wheel_str.str());	
    MonitorElement* MECountQual = igetter.get("DT/05-ChamberEff/Task/hCountSectVsChamb_Qual_W" + wheel_str.str());	
    MonitorElement* MEExtrap = igetter.get("DT/05-ChamberEff/Task/hExtrapSectVsChamb_W" + wheel_str.str());	

    //get the TH2F
    if(!MECountAll || !(MECountAll->getTH2F())) {
      edm::LogWarning("DTChamberEfficiencyClient") << "ME not available" << std::endl;
      return;
    }
      
    TH2F* hCountAll = MECountAll->getTH2F();
    TH2F* hCountQual = MECountQual->getTH2F();
    TH2F* hExtrap = MEExtrap->getTH2F();



    const int nBinX = summaryHistos[wheel+2][0]->getNbinsX();
    const int nBinY = summaryHistos[wheel+2][0]->getNbinsY();

    for(int j=1;j<=nBinX;j++){
      for(int k=1;k<=nBinY;k++){
	summaryHistos[wheel+2][0]->setBinContent(j,k,0.);
	summaryHistos[wheel+2][1]->setBinContent(j,k,0.);

	const float numerAll = hCountAll->GetBinContent(j,k);
	const float numerQual = hCountQual->GetBinContent(j,k);
	const float denom = hExtrap->GetBinContent(j,k);

	if(denom != 0.){
	  const float effAll= numerAll/denom;
	  const float eff_error_All = sqrt((effAll+effAll*effAll)/denom);

	  const float effQual= numerQual/denom;
	  const float eff_error_Qual = sqrt((effQual+effQual*effQual)/denom);

	  summaryHistos[wheel+2][0]->setBinContent(j,k,effAll);
	  summaryHistos[wheel+2][0]->setBinError(j,k,eff_error_All);

	  summaryHistos[wheel+2][1]->setBinContent(j,k,effQual);
	  summaryHistos[wheel+2][1]->setBinError(j,k,eff_error_Qual);

        // Fill 1D eff distributions
        globalEffDistr -> Fill(effAll);
        EffDistrPerWh[wheel+2] -> Fill(effAll);

	}
      }
    }
  }

  // fill the global eff. summary
  // problems at a granularity smaller than the chamber are ignored
  for(int wheel=-2; wheel<=2; wheel++) { // loop over wheels
    // retrieve the chamber efficiency summary
    MonitorElement * segmentWheelSummary = summaryHistos[wheel+2][0];
    if(segmentWheelSummary != 0) {

      for(int sector=1; sector<=12; sector++) { // loop over sectors
        float nFailingChambers = 0.;

	double meaneff = 0.;
	double errorsum = 0.;

	for(int station = 1; station != 5; ++station) { // loop over stations
	  
	  const double tmpefficiency = segmentWheelSummary->getBinContent(sector, station);
	  const double tmpvariance = pow(segmentWheelSummary->getBinError(sector, station),2);
	  
	  if(tmpefficiency < 0.2 || tmpvariance == 0){
	    nFailingChambers++;
	    continue;
	  }

	  meaneff += tmpefficiency/tmpvariance;
	  errorsum += 1./tmpvariance;

	  LogTrace("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
	    << "Wheel: " << wheel << " Stat: " << station
	    << " Sect: " << sector << " status: " << meaneff/errorsum << endl;
	}

	if(sector == 4 || sector == 10) {
	  int whichSector = (sector == 4) ? 13 : 14;

	  const double tmpefficiency = segmentWheelSummary->getBinContent(whichSector, 4);
	  const double tmpvariance = pow(segmentWheelSummary->getBinError(whichSector, 4),2);

	  if(tmpefficiency > 0.2 && tmpvariance != 0) {
	    meaneff += tmpefficiency/tmpvariance;
	    errorsum += 1./tmpvariance;
	  }
	  else nFailingChambers++;

	}

	double eff_result = 0;
        if(errorsum != 0) eff_result = meaneff/errorsum;

	if(nFailingChambers != 0) {
	  if(sector != 4 && sector != 10) eff_result = eff_result*(4.-nFailingChambers)/4.;
	  else eff_result = eff_result*(5.-nFailingChambers)/5.;
	}

	if(eff_result > 0.7) globalEffSummary->Fill(sector,wheel,1.);
	else if(eff_result < 0.7 && eff_result > 0.5) globalEffSummary->Fill(sector,wheel,0.6);
	else if(eff_result < 0.5 && eff_result > 0.3) globalEffSummary->Fill(sector,wheel,0.4);
	else if(eff_result < 0.3 && eff_result > 0.) globalEffSummary->Fill(sector,wheel,0.15);

      }
    }
  }
  return;
}

void DTChamberEfficiencyClient::bookHistos(DQMStore::IBooker & ibooker)
{

  ibooker.setCurrentFolder("DT/05-ChamberEff");

  globalEffSummary = ibooker.book2D("EfficiencyGlbSummary","Efficiency Summary",12,1,13,5,-2,3);
  globalEffSummary->setAxisTitle("sector",1);
  globalEffSummary->setAxisTitle("wheel",2);

  globalEffDistr = ibooker.book1D("TotalEfficiency","Total efficiency",51,0.,1.02);
  globalEffDistr -> setAxisTitle("Eff",1);

  for(int wh=-2; wh<=2; wh++){
    stringstream wheel; wheel << wh;
    string histoNameAll =  "EfficiencyMap_All_W" + wheel.str();
    string histoTitleAll =  "Efficiency map for all segments for wheel " + wheel.str();

    string histoNameQual =  "EfficiencyMap_Qual_W" + wheel.str();
    string histoTitleQual =  "Efficiency map for quality segments for wheel " + wheel.str();

    string histoNameEff =  "Efficiency_W" + wheel.str();
    string histoTitleEff =  "Segment efficiency, wheel " + wheel.str();

    ibooker.setCurrentFolder("DT/05-ChamberEff");

    summaryHistos[wh+2][0] = ibooker.book2D(histoNameAll.c_str(),histoTitleAll.c_str(),14,1.,15.,4,1.,5.);
    summaryHistos[wh+2][0]->setAxisTitle("Sector",1);
    summaryHistos[wh+2][0]->setBinLabel(1,"MB1",2);
    summaryHistos[wh+2][0]->setBinLabel(2,"MB2",2);
    summaryHistos[wh+2][0]->setBinLabel(3,"MB3",2);
    summaryHistos[wh+2][0]->setBinLabel(4,"MB4",2);

    EffDistrPerWh[wh+2] = ibooker.book1D(histoNameEff.c_str(),histoTitleEff.c_str(),51,0.,1.02);
    EffDistrPerWh[wh+2] -> setAxisTitle("Eff",1);

    ibooker.setCurrentFolder("DT/05-ChamberEff/HighQual");

    summaryHistos[wh+2][1] = ibooker.book2D(histoNameQual.c_str(),histoTitleQual.c_str(),14,1.,15.,4,1.,5.);
    summaryHistos[wh+2][1]->setAxisTitle("Sector",1);
    summaryHistos[wh+2][1]->setBinLabel(1,"MB1",2);
    summaryHistos[wh+2][1]->setBinLabel(2,"MB2",2);
    summaryHistos[wh+2][1]->setBinLabel(3,"MB3",2);
    summaryHistos[wh+2][1]->setBinLabel(4,"MB4",2);
   
  }

  return;
}
