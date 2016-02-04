
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/06/21 10:47:53 $
 *  $Revision: 1.17 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTNoiseAnalysisTest.h>

// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>



using namespace edm;
using namespace std;


DTNoiseAnalysisTest::DTNoiseAnalysisTest(const edm::ParameterSet& ps){
  LogTrace("DTDQM|DTMonitorClient|DTNoiseAnalysisTest") << "[DTNoiseAnalysisTest]: Constructor";

  dbe = edm::Service<DQMStore>().operator->();

  // get the cfi parameters
  noisyCellDef = ps.getUntrackedParameter<int>("noisyCellDef",500);

  // switch on/off the summaries for the Synchronous noise
  doSynchNoise = ps.getUntrackedParameter<bool>("doSynchNoise", false);
  detailedAnalysis = ps.getUntrackedParameter<bool>("detailedAnalysis",false);
  maxSynchNoiseRate =  ps.getUntrackedParameter<double>("maxSynchNoiseRate", 0.001);
  nMinEvts  = ps.getUntrackedParameter<int>("nEventsCert", 5000);

}


DTNoiseAnalysisTest::~DTNoiseAnalysisTest(){
  LogTrace("DTDQM|DTMonitorClient|DTNoiseAnalysisTest") << "DTNoiseAnalysisTest: analyzed " << nevents << " events";
}


void DTNoiseAnalysisTest::beginJob(){
  LogTrace("DTDQM|DTMonitorClient|DTNoiseAnalysisTest") <<"[DTNoiseAnalysisTest]: BeginJob"; 

  nevents = 0;

  // book the histos
  bookHistos();

}

void DTNoiseAnalysisTest::beginRun(Run const& run, EventSetup const& context) {

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTNoiseAnalysisTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  LogTrace("DTDQM|DTMonitorClient|DTNoiseAnalysisTest") <<"[DTNoiseAnalysisTest]: Begin of LS transition";
}


void DTNoiseAnalysisTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  if(nevents%1000 == 0) LogTrace("DTDQM|DTMonitorClient|DTNoiseAnalysisTest")
    << "[DTNoiseAnalysisTest]: "<<nevents<<" events";

}

void DTNoiseAnalysisTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  LogVerbatim ("DTDQM|DTMonitorClient|DTNoiseAnalysisTest")
    <<"[DTNoiseAnalysisTest]: End of LS transition, performing the DQM client operation";

  // Reset the summary plots
  for(map<int, MonitorElement* >::iterator plot =  noiseHistos.begin();
      plot != noiseHistos.end(); ++plot) {
    (*plot).second->Reset();
  }

  for(map<int,  MonitorElement* >::iterator plot = noisyCellHistos.begin();
      plot != noisyCellHistos.end(); ++plot) {
    (*plot).second->Reset();
  }

  summaryNoiseHisto->Reset();



  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  LogTrace ("DTDQM|DTMonitorClient|DTNoiseAnalysisTest")
    <<"[DTNoiseAnalysisTest]: Fill the summary histos";

  for (; ch_it != ch_end; ++ch_it) { // loop over chambers
    DTChamberId chID = (*ch_it)->id();

    MonitorElement * histo = dbe->get(getMEName(chID));

    if(histo) { // check the pointer

      TH2F * histo_root = histo->getTH2F();

      for(int sl = 1; sl != 4; ++sl) { // loop over SLs
        // skip theta SL in MB4 chambers
        if(chID.station() == 4 && sl == 2) continue;

        int binYlow = ((sl-1)*4)+1;

        for(int layer = 1; layer <= 4; ++layer) { // loop over layers

          // Get the layer ID
          DTLayerId layID(chID,sl,layer);

          int nWires = muonGeom->layer(layID)->specificTopology().channels();
          int firstWire = muonGeom->layer(layID)->specificTopology().firstChannel();

          int binY = binYlow+(layer-1);

          for(int wire = firstWire; wire != (nWires+firstWire); wire++){ // loop over wires

            double noise = histo_root->GetBinContent(wire, binY);
            // fill the histos
            noiseHistos[chID.wheel()]->Fill(noise);
            noiseHistos[3]->Fill(noise);
            int sector = chID.sector();
            if(noise>noisyCellDef) {
              if(sector == 13) {
                sector = 4;
              } else if(sector == 14) {
                sector = 10;
              }
              noisyCellHistos[chID.wheel()]->Fill(sector,chID.station());
              summaryNoiseHisto->Fill(sector,chID.wheel());
            }
          }
        }
      }
    }
  }

  if(detailedAnalysis) {
    threshChannelsHisto->Reset();
    TH1F * histo = noiseHistos[3]->getTH1F();
    for(int step = 0; step != 15; step++) {
      int threshBin = step + 1;
      int minBin = 26 + step*5;
      int nNoisyCh = histo->Integral(minBin,101);
      threshChannelsHisto->setBinContent(threshBin,nNoisyCh);
    }
  }

  // build the summary of synch noise


  if(doSynchNoise) {
    LogTrace("DTDQM|DTMonitorClient|DTNoiseAnalysisTest")
      << "[DTNoiseAnalysisTest]: fill summaries for synch noise" << endl;
    summarySynchNoiseHisto->Reset();
    glbSummarySynchNoiseHisto->Reset();
    for(int wheel = -2; wheel != 3; ++wheel) {
      // Get the histo produced by DTDigiTask
      MonitorElement * histoNoiseSynch = dbe->get(getSynchNoiseMEName(wheel));
      if(histoNoiseSynch != 0) {
        for(int sect = 1; sect != 13; ++sect) { // loop over sectors
          TH2F * histo = histoNoiseSynch->getTH2F();
          float maxSectRate = 0;
          for(int sta = 1; sta != 5; ++sta) {
            float chRate = histo->GetBinContent(sect, sta)/(float)nevents;
            LogTrace("DTDQM|DTMonitorClient|DTNoiseAnalysisTest")
              << "   Wheel: " << wheel << " sect: " << sect
              << " station: " << sta
              << " rate is: " << chRate << endl;
            if (chRate > maxSectRate)
              maxSectRate = chRate;
          }
          summarySynchNoiseHisto->Fill(sect,wheel,
              maxSectRate > maxSynchNoiseRate ? 1 : 0);
          float glbBinValue = 1 - 0.15*maxSectRate/maxSynchNoiseRate;
          glbSummarySynchNoiseHisto->Fill(sect,wheel,glbBinValue>0 ? glbBinValue : 0);

        }
      } else {
        LogWarning("DTDQM|DTMonitorClient|DTNoiseAnalysisTest")
          << "   Histo: " << getSynchNoiseMEName(wheel) << " not found!" << endl;
      }
    }

  }

  string nEvtsName = "DT/EventInfo/Counters/nProcessedEventsNoise";
  MonitorElement * meProcEvts = dbe->get(nEvtsName);

  if (meProcEvts) {
    int nProcEvts = meProcEvts->getFloatValue();
    glbSummarySynchNoiseHisto->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
    summarySynchNoiseHisto->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
  } else {
    glbSummarySynchNoiseHisto->setEntries(nMinEvts +1);
    summarySynchNoiseHisto->setEntries(nMinEvts + 1);
    LogVerbatim ("DTDQM|DTMonitorClient|DTnoiseAnalysisTest") << "[DTNoiseAnalysisTest] ME: "
      <<  nEvtsName << " not found!" << endl;
  }


}	       


string DTNoiseAnalysisTest::getMEName(const DTChamberId & chID) {

  stringstream wheel; wheel << chID.wheel();	
  stringstream station; station << chID.station();	
  stringstream sector; sector << chID.sector();	

  string folderName = 
    "DT/05-Noise/Wheel" +  wheel.str() +
    //     "/Station" + station.str() +
    "/Sector" + sector.str() + "/";

  string histoname = folderName + string("NoiseRate")  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str();

  return histoname;

}


void DTNoiseAnalysisTest::bookHistos() {

  dbe->setCurrentFolder("DT/05-Noise");
  string histoName;

  for(int wh=-2; wh<=2; wh++){
    stringstream wheel; wheel << wh;
    histoName =  "NoiseRateSummary_W" + wheel.str();
    noiseHistos[wh] = dbe->book1D(histoName.c_str(),histoName.c_str(),100,0,2000);
    noiseHistos[wh]->setAxisTitle("rate (Hz)",1);
    noiseHistos[wh]->setAxisTitle("entries",2);
  }
  histoName =  "NoiseRateSummary";
  noiseHistos[3] = dbe->book1D(histoName.c_str(),histoName.c_str(),100,0,2000);
  noiseHistos[3]->setAxisTitle("rate (Hz)",1);
  noiseHistos[3]->setAxisTitle("entries",2);


  for(int wh=-2; wh<=2; wh++){
    stringstream wheel; wheel << wh;
    histoName =  "NoiseSummary_W" + wheel.str();
    noisyCellHistos[wh] = dbe->book2D(histoName.c_str(),"# of noisy channels",12,1,13,4,1,5);
    noisyCellHistos[wh]->setBinLabel(1,"MB1",2);
    noisyCellHistos[wh]->setBinLabel(2,"MB2",2);
    noisyCellHistos[wh]->setBinLabel(3,"MB3",2);
    noisyCellHistos[wh]->setBinLabel(4,"MB4",2);  
    noisyCellHistos[wh]->setAxisTitle("Sector",1);
  }

  histoName =  "NoiseSummary";
  summaryNoiseHisto =  dbe->book2D(histoName.c_str(),"# of noisy channels",12,1,13,5,-2,3);
  summaryNoiseHisto->setAxisTitle("Sector",1);
  summaryNoiseHisto->setAxisTitle("Wheel",2);

  if(detailedAnalysis) {
    histoName = "NoisyChannels";
    threshChannelsHisto = dbe->book1D(histoName.c_str(),"# of noisy channels vs threshold",15,500,2000);
    threshChannelsHisto->setAxisTitle("threshold",1);
    threshChannelsHisto->setAxisTitle("# noisy channels",2);
  }

  if(doSynchNoise) {
    dbe->setCurrentFolder("DT/05-Noise/SynchNoise/");
    histoName =  "SynchNoiseSummary";
    summarySynchNoiseHisto = dbe->book2D(histoName.c_str(),"Summary Synch. Noise",12,1,13,5,-2,3);
    summarySynchNoiseHisto->setAxisTitle("Sector",1);
    summarySynchNoiseHisto->setAxisTitle("Wheel",2);
    histoName =  "SynchNoiseGlbSummary";
    glbSummarySynchNoiseHisto = dbe->book2D(histoName.c_str(),"Summary Synch. Noise",12,1,13,5,-2,3);
    glbSummarySynchNoiseHisto->setAxisTitle("Sector",1);
    glbSummarySynchNoiseHisto->setAxisTitle("Wheel",2);
  }

}



string DTNoiseAnalysisTest::getSynchNoiseMEName(int wheelId) const {

  stringstream wheel; wheel << wheelId;	
  string folderName = 
    "DT/05-Noise/SynchNoise/";
  string histoname = folderName + string("SyncNoiseEvents")  
    + "_W" + wheel.str();

  return histoname;

}

