/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/08 11:39:40 $
 *  $Revision: 1.1 $
 *  \author M. Pelliccioni - INFN Torino
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

  dbe = Service<DQMStore>().operator->();

  prescaleFactor = pSet.getUntrackedParameter<int>("diagnosticPrescale", 1);
}

DTChamberEfficiencyClient::~DTChamberEfficiencyClient()
{
   LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
     << "DTChamberEfficiencyClient: Destructor called";
}

void DTChamberEfficiencyClient::beginJob(const EventSetup& context)
{
  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "DTChamberEfficiencyClient: BeginJob";

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  bookHistos();

  return;
}

void DTChamberEfficiencyClient::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context)
{
  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "[DTChamberEfficiencyClient]: Begin of LS transition";

  return;
}

void DTChamberEfficiencyClient::beginRun(const Run& run, const EventSetup& setup)
{
  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "DTChamberEfficiencyClient: beginRun";

  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(muonGeom);

  return;
}

void DTChamberEfficiencyClient::analyze(const Event& e, const EventSetup& context)
{

  nevents++;
  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "[DTChamberEfficiencyClient]: " << nevents << " events";
  return;
}

void DTChamberEfficiencyClient::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context)
{
  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "DTChamberEfficiencyClient: endRun";

  //Loop over the wheels
  for(int wheel=-2;wheel<=2;wheel++){
    stringstream wheel_str; wheel_str << wheel;

    // Get the ME produced by EfficiencyTask Source
    // All means no selection on segments, Qual means segments with at least 12 hits
    MonitorElement* MECountAll = dbe->get("DT/05-ChamberEff/Task/hCountSectVsChamb_All_W" + wheel_str.str());	
    MonitorElement* MECountQual = dbe->get("DT/05-ChamberEff/Task/hCountSectVsChamb_Qual_W" + wheel_str.str());	
    MonitorElement* MEExtrap = dbe->get("DT/05-ChamberEff/Task/hExtrapSectVsChamb_W" + wheel_str.str());	

    //get the TH2F
    TH2F* hCountAll = MECountAll->getTH2F();
    TH2F* hCountQual = MECountQual->getTH2F();
    TH2F* hExtrap = MEExtrap->getTH2F();

    int nBinX = summaryHistos[wheel+2][0]->getNbinsX();
    int nBinY = summaryHistos[wheel+2][0]->getNbinsY();

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
	}
      }
    }
  }

  return;
}

void DTChamberEfficiencyClient::endJob()
{
  LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyClient")
    << "DTChamberEfficiencyClient: endJob";
  return;
}

void DTChamberEfficiencyClient::bookHistos()
{
  for(int wh=-2; wh<=2; wh++){
    stringstream wheel; wheel << wh;
    string histoNameAll =  "EfficiencyMap_All_W" + wheel.str();
    string histoTitleAll =  "Efficiency map for all segments for wheel " + wheel.str();

    string histoNameQual =  "EfficiencyMap_Qual_W" + wheel.str();
    string histoTitleQual =  "Efficiency map for quality segments for wheel " + wheel.str();


    dbe->setCurrentFolder("DT/05-ChamberEff");

    summaryHistos[wh+2][0] = dbe->book2D(histoNameAll.c_str(),histoTitleAll.c_str(),14,1.,15.,4,1.,5.);
    summaryHistos[wh+2][0]->setAxisTitle("Sector",1);
    summaryHistos[wh+2][0]->setBinLabel(1,"MB1",2);
    summaryHistos[wh+2][0]->setBinLabel(2,"MB2",2);
    summaryHistos[wh+2][0]->setBinLabel(3,"MB3",2);
    summaryHistos[wh+2][0]->setBinLabel(4,"MB4",2);

    dbe->setCurrentFolder("DT/05-ChamberEff/HighQual");

    summaryHistos[wh+2][1] = dbe->book2D(histoNameQual.c_str(),histoTitleQual.c_str(),14,1.,15.,4,1.,5.);
    summaryHistos[wh+2][1]->setAxisTitle("Sector",1);
    summaryHistos[wh+2][1]->setBinLabel(1,"MB1",2);
    summaryHistos[wh+2][1]->setBinLabel(2,"MB2",2);
    summaryHistos[wh+2][1]->setBinLabel(3,"MB3",2);
    summaryHistos[wh+2][1]->setBinLabel(4,"MB4",2);
  }

  return;
}
