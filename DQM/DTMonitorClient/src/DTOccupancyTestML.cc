/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - University and INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */


#include <DQM/DTMonitorClient/src/DTOccupancyTestML.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TMath.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <vector>

using namespace edm;
using namespace std;




DTOccupancyTestML::DTOccupancyTestML(const edm::ParameterSet& ps){
  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTestML") << "[DTOccupancyTestML]: Constructor";

  // Get the DQM service

  lsCounter = 0;

  writeRootFile  = ps.getUntrackedParameter<bool>("writeRootFile", false);
  if(writeRootFile) {
    rootFile = new TFile("MLDTOccupancyTest.root","RECREATE");
    ntuple = new TNtuple("OccupancyNtuple", "OccupancyNtuple", "ls:wh:st:se:lay1MeanCell:lay1RMS:lay2MeanCell:lay2RMS:lay3MeanCell:lay3RMS:lay4MeanCell:lay4RMS:lay5MeanCell:lay5RMS:lay6MeanCell:lay6RMS:lay7MeanCell:lay7RMS:lay8MeanCell:lay8RMS:lay9MeanCell:lay9RMS:lay10MeanCell:lay10RMS:lay11MeanCell:lay11RMS:lay12MeanCell:lay12RMS");
  }
  
  // switch on the mode for running on test pulses (different top folder)
  tpMode = ps.getUntrackedParameter<bool>("testPulseMode", false);
  
  runOnAllHitsOccupancies =  ps.getUntrackedParameter<bool>("runOnAllHitsOccupancies", true);
  runOnNoiseOccupancies =  ps.getUntrackedParameter<bool>("runOnNoiseOccupancies", false);
  runOnInTimeOccupancies = ps.getUntrackedParameter<bool>("runOnInTimeOccupancies", false);
  nMinEvts  = ps.getUntrackedParameter<int>("nEventsCert", 5000);
  nMinEvtsPC  = ps.getUntrackedParameter<int>("nEventsMinPC", 2200);
  nZeroEvtsPC  = ps.getUntrackedParameter<int>("nEventsZeroPC", 30);

  bookingdone = false;

  // Event counter
  nevents = 0;

}

DTOccupancyTestML::~DTOccupancyTestML(){
  LogVerbatim ("DTDQM|DTMonitorClient|MLDTOccupancyTest") << " destructor called" << endl;
}


void DTOccupancyTestML::beginRun(const edm::Run& run, const EventSetup& context){

  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTestML") << "[DTOccupancyTestML]: BeginRun";

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}

  void DTOccupancyTestML::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,
                                edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {
  if (!bookingdone) {

  // Book the summary histos
  //   - one summary per wheel
  for(int wh = -2; wh <= 2; ++wh) { // loop over wheels
    bookHistos(ibooker,wh, string("MLDTOccupancies"), "MLOccupancySummary");
    }

  ibooker.setCurrentFolder(topFolder(true));
  string title = "Occupancy Summary";
  if(tpMode) {
    title = "Test Pulse Occupancy Summary";
    }
  //   - global summary with alarms
  summaryHisto = ibooker.book2D("MLOccupancySummary",title.c_str(),12,1,13,5,-2,3);
  summaryHisto->setAxisTitle("sector",1);
  summaryHisto->setAxisTitle("wheel",2);
  
  //   - global summary with percentages
  glbSummaryHisto = ibooker.book2D("MLOccupancyGlbSummary",title.c_str(),12,1,13,5,-2,3);
  glbSummaryHisto->setAxisTitle("sector",1);
  glbSummaryHisto->setAxisTitle("wheel",2);


  // assign the name of the input histogram
  if(runOnAllHitsOccupancies) {
    nameMonitoredHisto = "OccupancyAllHits_perCh";
    } else if(runOnNoiseOccupancies) {
    nameMonitoredHisto = "OccupancyNoise_perCh";
    } else if(runOnInTimeOccupancies) {
    nameMonitoredHisto = "OccupancyInTimeHits_perCh";
    } else { // default is AllHits histo
    nameMonitoredHisto = "OccupancyAllHits_perCh";
    }

  }
  bookingdone = true; 


  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTestML")
    <<"[DTOccupancyTestML]: End of LS transition, performing the DQM client operation";
  lsCounter++;



  // Reset the global summary
  summaryHisto->Reset();
  glbSummaryHisto->Reset();

  nChannelTotal = 0;
  nChannelDead = 0;

  // Get all the DT chambers
  vector<const DTChamber*> chambers = muonGeom->chambers();

  // Load graph
  tensorflow::setLogging("3");
  edm::FileInPath modelFilePath("DQM/DTMonitorClient/data/occupancy_cnn_v1.pb");
  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(modelFilePath.fullPath());

  // Create session
  tensorflow::Session* session = tensorflow::createSession(graphDef);

  for(vector<const DTChamber*>::const_iterator chamber = chambers.begin();
      chamber != chambers.end(); ++chamber) {  // Loop over all chambers
    DTChamberId chId = (*chamber)->id();

    MonitorElement * chamberOccupancyHisto = igetter.get(getMEName(nameMonitoredHisto, chId));	

    // Run the tests on the plot for the various granularities
    if(chamberOccupancyHisto != nullptr) {
      // Get the 2D histo
      TH2F *histo = chamberOccupancyHisto->getTH2F();

      float chamberPercentage = 1.;
      int result = runOccupancyTest(histo, chId, chamberPercentage, graphDef, session);
      int sector = chId.sector();

      if(sector == 13) {
	sector = 4;
	float resultSect4 = wheelHistos[chId.wheel()]->getBinContent(sector, chId.station());
	if(resultSect4 > result) {
	  result = (int)resultSect4;
	}
      } else if(sector == 14) {
	sector = 10;
	float resultSect10 = wheelHistos[chId.wheel()]->getBinContent(sector, chId.station());
	if(resultSect10 > result) {
	  result = (int)resultSect10;
	}
      }
      
      // the 2 MB4 of Sect 4 and 10 count as half a chamber
      if((sector == 4 || sector == 10) && chId.station() == 4) 
	chamberPercentage = chamberPercentage/2.;

      wheelHistos[chId.wheel()]->setBinContent(sector, chId.station(),result);
      if(result > summaryHisto->getBinContent(sector, chId.wheel()+3)) {
	summaryHisto->setBinContent(sector, chId.wheel()+3, result);
      }
      glbSummaryHisto->Fill(sector, chId.wheel(), chamberPercentage*1./4.);
    } else {
      LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTestML") << "[DTOccupancyTestML] ME: "
				      << getMEName(nameMonitoredHisto, chId) << " not found!" << endl;
    }

  }

  // Clean up neural network graph
  tensorflow::closeSession(session);
  delete graphDef;

  string nEvtsName = "DT/EventInfo/Counters/nProcessedEventsDigi";

  MonitorElement * meProcEvts = igetter.get(nEvtsName);

  if (meProcEvts) {
    int nProcEvts = meProcEvts->getFloatValue();
    glbSummaryHisto->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
    summaryHisto->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
  } else {
    glbSummaryHisto->setEntries(nMinEvts +1);
    summaryHisto->setEntries(nMinEvts + 1);
    LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTestML") << "[DTOccupancyTestML] ME: "
		       <<  nEvtsName << " not found!" << endl;
  }

  // Fill the global summary
  // Check for entire sectors off and report them on the global summary
  //FIXME: TODO

  if(writeRootFile) ntuple->AutoSave("SaveSelf");

}

void DTOccupancyTestML::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTestML") << "[DTOccupancyTestML] endjob called!";
  if(writeRootFile) {
    rootFile->cd();
    ntuple->Write();
    rootFile->Close();
  }
}


  
// --------------------------------------------------

void DTOccupancyTestML::bookHistos(DQMStore::IBooker & ibooker, const int wheelId, 
                                                      string folder, string histoTag) {
  // Set the current folder
  stringstream wheel; wheel << wheelId;	

  ibooker.setCurrentFolder(topFolder(true));

  // build the histo name
  string histoName = histoTag + "_W" + wheel.str(); 
  
  
  LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTestML") <<"[DTOccupancyTestML]: booking wheel histo:"
							<< histoName 
							<< " (tag "
							<< histoTag
							<< ") in: "
							<< topFolder(true) + "Wheel"+ wheel.str() + "/" + folder << endl;
  
  string histoTitle = "Occupancy summary WHEEL: "+wheel.str();
  if(tpMode) {
    histoTitle = "TP Occupancy summary WHEEL: "+wheel.str();
  }

  wheelHistos[wheelId] = ibooker.book2D(histoName,histoTitle,12,1,13,4,1,5);
  wheelHistos[wheelId]->setBinLabel(1,"MB1",2);
  wheelHistos[wheelId]->setBinLabel(2,"MB2",2);
  wheelHistos[wheelId]->setBinLabel(3,"MB3",2);
  wheelHistos[wheelId]->setBinLabel(4,"MB4",2);
  wheelHistos[wheelId]->setAxisTitle("sector",1);
}



string DTOccupancyTestML::getMEName(string histoTag, const DTChamberId& chId) {

  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();
  stringstream sector; sector << chId.sector();


  string folderRoot = topFolder(false) + "Wheel" + wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/";

  // build the histo name
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str();

  string histoname = folderRoot + histoName;

  return histoname;
}


int DTOccupancyTestML::getIntegral(TH2F *histo, int firstBinX, int lastBinX, int firstBinY, int lastBinY, bool doall) {

  int sum = 0;
    for (Int_t i = firstBinX; i < lastBinX+1; i++) {
      for (Int_t j = firstBinY; j < lastBinY+1; j++) {
        
        if (histo->GetBinContent(i,j) >0){
          if (!doall) return 1;
	  sum += histo->GetBinContent(i,j);
	}
      }
    }

    return sum;
}

int DTOccupancyTestML::runOccupancyTest(TH2F *histo, const DTChamberId& chId,
                                        float& chamberPercentage,
                                        tensorflow::GraphDef *graphDef,
                                        tensorflow::Session *session) {

  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTestML")
           << "--- Occupancy test for chamber: " << chId << endl;

  // Initialize counters
  int totalLayers = 0;
  int badLayers = 0;

  // Loop over the super layers
  for (int superLayer = 1; superLayer <= 3; superLayer++) {
    int binYlow = ((superLayer-1)*4)+1;

    // Skip for non-existent super layers
    if (chId.station() == 4 && superLayer == 2) continue;

    // Loop over layers
    for (int layer = 1; layer <= 4; layer++) {
      DTLayerId layID(chId, superLayer, layer);
      int firstWire = muonGeom->layer(layID)->
                                             specificTopology().firstChannel();
      int nWires = muonGeom->layer(layID)->specificTopology().channels();
      int binY = binYlow+(layer-1);
      std::vector<float> layerOccupancy(nWires);
      int channelId = 0;

      // Loop over cells within  a layer
      for (int cell = firstWire; cell != (nWires + firstWire); cell++) {
        double cellOccupancy = histo->GetBinContent(cell, binY);
        layerOccupancy.at(channelId) = cellOccupancy;
        channelId++;
      }

      int targetSize = 47;
      std::vector<float> reshapedLayerOccupancy = interpolateLayers(layerOccupancy,
                                                                    nWires,
                                                                    targetSize);

      // Scale occupancy
      float maxOccupancyInLayer =
        *std::max_element(reshapedLayerOccupancy.begin(),
                          reshapedLayerOccupancy.end());

      if (maxOccupancyInLayer != 0) {
          for (auto &val : reshapedLayerOccupancy)
            val /= maxOccupancyInLayer;
      }

      // Define input
      tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, targetSize});
      for (int i = 0; i < targetSize; i++)
        input.matrix<float>()(0, i) = reshapedLayerOccupancy[i];

      std::vector<tensorflow::Tensor> outputs;
      tensorflow::run(session, { { "input_cnn_input", input } },
                      { "output_cnn/Softmax" }, &outputs);

      totalLayers++;
      bool isBad = outputs[0].matrix<float>()(0, 1) > 0.95;
      if (isBad) badLayers++;
    }
  }

  // Calculate a fraction of good layers
  chamberPercentage = 1.0 - static_cast<float>(badLayers)/totalLayers;

  if (badLayers > 8) return 3; // 3 SL
  if (badLayers > 4) return 2; // 2 SL
  if (badLayers > 0) return 1; // 1 SL
  return 0;
}

std::vector<float> DTOccupancyTestML::interpolateLayers(std::vector<float> const& inputs, int size, int targetSize) {
  // Reshape layers with linear interpolation
  int interpolationFloor = 0;
  float interpolationPoint = 0.;
  float step = static_cast<float>(size) / targetSize;
  std::vector<float> reshapedInput(targetSize);

  for (int i = 0; i < targetSize; i++) {
    interpolationFloor = static_cast<int>(std::floor(interpolationPoint));
    // Interpolating here
    reshapedInput.at(i) = (interpolationPoint - interpolationFloor) *
     (inputs[interpolationFloor + 1] - inputs[interpolationFloor]) +
     inputs[interpolationFloor];
    interpolationPoint = step + interpolationPoint;
  }
  return reshapedInput;
}

string DTOccupancyTestML::topFolder(bool isBooking) const {
  if(tpMode) return string("DT/10-TestPulses/");
  if(isBooking) return string("DT/01-Digi/ML");
  return string("DT/01-Digi/");
}
