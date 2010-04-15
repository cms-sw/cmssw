/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/10/01 13:05:46 $
 *  $Revision: 1.10 $
 *  \author G. Mila - INFN Torino
 */


#include "CalibMuon/DTCalibration/plugins/DTNoiseCalibration.h"


// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EventSetup.h>

// Geometry
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// Digis
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

// Database
#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "TH2F.h"
#include "TFile.h"

using namespace edm;
using namespace std;


DTNoiseCalibration::DTNoiseCalibration(const edm::ParameterSet& ps){

  cout << "[DTNoiseCalibration]: Constructor" <<endl;

  // Get the debug parameter for verbose output
  debug = ps.getUntrackedParameter<bool>("debug");

  // Get the label to retrieve digis from the event
  digiLabel = ps.getUntrackedParameter<string>("digiLabel");

  // The analysis type
  fastAnalysis = ps.getUntrackedParameter<bool>("fastAnalysis", true);
  // The wheel & sector interested for the time-dependent analysis
  wh = ps.getUntrackedParameter<int>("wheel", 0);
  sect = ps.getUntrackedParameter<int>("sector", 6);

  // The trigger mode
  cosmicRun = ps.getUntrackedParameter<bool>("cosmicRun", false);

  // The trigger width (if noise run)
  TriggerWidth = ps.getUntrackedParameter<int>("TriggerWidth");

  //get the offset to look for the noise
  theOffset = ps.getUntrackedParameter<double>("theOffset",500.);

  // The root file which will contain the histos
  string rootFileName = ps.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  parameters=ps;

}

void DTNoiseCalibration::beginRun(const edm::Run& run, const edm::EventSetup& setup ) {

  cout <<"[DTNoiseCalibration]: BeginJob"<<endl; 
  nevents = 0;
  counter = 0;

  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // tTrig 
  if (parameters.getUntrackedParameter<bool>("readDB", true)) 
    setup.get<DTTtrigRcd>().get(tTrigMap);

  // TDC time distribution
  int numBin = (TriggerWidth*(32/25))/50;
  hTDCTriggerWidth = new TH1F("TDC_Time_Distribution", "TDC_Time_Distribution", numBin, 0, TriggerWidth*(32/25));

}


void DTNoiseCalibration::analyze(const edm::Event& e, const edm::EventSetup& context){
  nevents++;
  if(debug)
    cout<<"nevents: "<<nevents<<endl;
  
  // Get the digis from the event
  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel(digiLabel, dtdigis);

  TH1F *hOccupancyHisto;
  TH2F *hEvtPerWireH;
  string Histo2Name;

  // LOOP OVER ALL THE DIGIS OF THE EVENT
  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){

      //Check the TDC trigger width
      int tdcTime = (*digiIt).countsTDC();
      if(!cosmicRun){	
	if(debug)
	  cout<<"tdcTime (ns): "<<(tdcTime*25)/32<<endl;
	if(((tdcTime*25)/32)>TriggerWidth){
	  cout<<"***Error*** : your digi has a tdcTime (ns) higher than the TDC trigger width :"<<(tdcTime*25)/32<<endl;
	  abort();
	}
      }	

      if((!fastAnalysis &&
	 (*dtLayerId_It).first.superlayerId().chamberId().wheel()==wh &&
	 (*dtLayerId_It).first.superlayerId().chamberId().sector()==sect) ||
	 fastAnalysis)
	hTDCTriggerWidth->Fill(tdcTime);

      // Set the window of interest if the run is triggered by cosmics
      if ( parameters.getUntrackedParameter<bool>("readDB", true) ) {
	tTrigMap->get( ((*dtLayerId_It).first).superlayerId(), tTrig, tTrigRMS, kFactor, DTTimeUnits::counts );

	upperLimit = tTrig - theOffset;
      }
      else { 
	tTrig = parameters.getUntrackedParameter<int>("defaultTtrig", 4000);
	upperLimit = tTrig - theOffset;
      }
	
      if((cosmicRun && (*digiIt).countsTDC()<upperLimit) || (!cosmicRun) ){

	if(debug && cosmicRun)
	  cout<<"tdcTime (ns): "<<((*digiIt).countsTDC()*25)/32<<" --- TriggerWidth (ns): "<<(upperLimit*25)/32<<endl;

	// Get the number of wires
	const  DTLayerId dtLId = (*dtLayerId_It).first;
	const DTTopology& dtTopo = dtGeom->layer(dtLId)->specificTopology();
	const int nWires = dtTopo.channels();
	const int firstWire = dtTopo.firstChannel();
	const int lastWire = dtTopo.lastChannel();
	
	// book the occupancy histos
	theFile->cd();
	if((!fastAnalysis &&
	   dtLId.superlayerId().chamberId().wheel()==wh &&
	   dtLId.superlayerId().chamberId().sector()==sect) ||
	   fastAnalysis){
	  hOccupancyHisto = theHistoOccupancyMap[dtLId];
	  if(hOccupancyHisto == 0) {
	    string HistoName = "DigiOccupancy_" + getLayerName(dtLId);
	    theFile->cd();
	    hOccupancyHisto = new TH1F(HistoName.c_str(), HistoName.c_str(), nWires, firstWire, lastWire+1);
	    if(debug)
	      cout << "  New Occupancy Histo: " << hOccupancyHisto->GetName() << endl;
	    theHistoOccupancyMap[dtLId] = hOccupancyHisto;
	  }
	  hOccupancyHisto->Fill((*digiIt).wire());
	}

	// book the digi event plot every 1000 events if the analysis is not "fast" and if is the correct sector
	if(!fastAnalysis &&
	   dtLId.superlayerId().chamberId().wheel()==wh &&
	   dtLId.superlayerId().chamberId().sector()==sect) {
	  if(theHistoEvtPerWireMap.find(dtLId) == theHistoEvtPerWireMap.end() ||
	     (theHistoEvtPerWireMap.find(dtLId) != theHistoEvtPerWireMap.end() &&
	      skippedPlot[dtLId] != counter)){ 
	    skippedPlot[dtLId] = counter;
	    stringstream toAppend; toAppend << counter;
	    Histo2Name = "DigiPerWirePerEvent_" + getLayerName(dtLId) + "_" + toAppend.str();
	    theFile->cd();
	    hEvtPerWireH = new TH2F(Histo2Name.c_str(), Histo2Name.c_str(), 1000,0.5,1000.5,nWires, firstWire, lastWire+1);
	    if(hEvtPerWireH){
	      if(debug)
		cout << "  New Histo with the number of digi per evt per wire: " << hEvtPerWireH->GetName() << endl;
	      theHistoEvtPerWireMap[dtLId]=hEvtPerWireH;
	    }
	  }
	}
      }
    }
  }
    
  //Fill the plot of the number of digi per event per wire
  std::map<int,int > DigiPerWirePerEvent;
  // LOOP OVER ALL THE CHAMBERS
  vector<DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = dtGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId ch = (*ch_it)->id();
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();
    // Loop over the SLs
    for(; sl_it != sl_end; ++sl_it) {
      DTSuperLayerId sl = (*sl_it)->id();
      vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
      vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
      // Loop over the Ls
      for(; l_it != l_end; ++l_it) {
	DTLayerId layerId = (*l_it)->id();
	
	// Get the number of wires
	const DTTopology& dtTopo = dtGeom->layer(layerId)->specificTopology();
	const int firstWire = dtTopo.firstChannel();
	const int lastWire = dtTopo.lastChannel();
	  
	if (theHistoEvtPerWireMap.find(layerId) != theHistoEvtPerWireMap.end() &&
	    skippedPlot[layerId] == counter) {
	  
	  for (int wire=firstWire; wire<=lastWire; wire++) {
	    DigiPerWirePerEvent[wire]= 0;
	  }	
	  // loop over all the digis of the event
	  DTDigiCollection::Range layerDigi= dtdigis->get(layerId);
	  for (DTDigiCollection::const_iterator digi = layerDigi.first;
	       digi!=layerDigi.second;
	       ++digi){
	    if((cosmicRun && (*digi).countsTDC()<upperLimit) || (!cosmicRun))
	      DigiPerWirePerEvent[(*digi).wire()]+=1;
	  }
	  // fill the digi event histo
	  for (int wire=firstWire; wire<=lastWire; wire++) {
	    theFile->cd();
	    int histoEvents = nevents - (counter*1000);
	    theHistoEvtPerWireMap[layerId]->Fill(histoEvents,wire,DigiPerWirePerEvent[wire]);
	  }
	}
      } //Loop Ls
    } //Loop SLs
  } //Loop chambers
  
  
  if(nevents % 1000 == 0) {
    counter++;
    // save the digis event plot on file
    for(map<DTLayerId,  TH2F* >::const_iterator lHisto = theHistoEvtPerWireMap.begin();
	lHisto != theHistoEvtPerWireMap.end();
	lHisto++) {
      theFile->cd();
      if((*lHisto).second)
	(*lHisto).second->Write();
    }
    theHistoEvtPerWireMap.clear();
  }
  
}


void DTNoiseCalibration::endJob(){

  cout << "[DTNoiseCalibration] endjob called!" <<endl;

  // save the TDC digi plot
  theFile->cd();
  hTDCTriggerWidth->Write();

  // save on file the occupancy histo and write the list of noisy cells
  double TriggerWidth_s=0;
  DTStatusFlag *statusMap = new DTStatusFlag();
  for(map<DTLayerId, TH1F*>::const_iterator lHisto = theHistoOccupancyMap.begin();
      lHisto != theHistoOccupancyMap.end();
      lHisto++) {
    if(cosmicRun){
      if ( parameters.getUntrackedParameter<bool>("readDB", true) ) 
	tTrigMap->get( ((*lHisto).first).superlayerId(), tTrig, tTrigRMS, kFactor,
		     DTTimeUnits::counts );
     else tTrig = parameters.getUntrackedParameter<int>("defaultTtrig", 4000);
      double TriggerWidth_ns = ((tTrig-theOffset)*25)/32;
      TriggerWidth_s = TriggerWidth_ns/1e9;
    }
    if(!cosmicRun)
      TriggerWidth_s = double(TriggerWidth/1e9);
    if(debug)
      cout<<"TriggerWidth (s): "<<TriggerWidth_s<<"  TotEvents: "<<nevents<<endl;
    double normalization = 1/double(nevents*TriggerWidth_s);
    if((*lHisto).second){
      (*lHisto).second->Scale(normalization);
      theFile->cd();
      (*lHisto).second->Write();
      const DTTopology& dtTopo = dtGeom->layer((*lHisto).first)->specificTopology();
      const int firstWire = dtTopo.firstChannel();
      const int lastWire = dtTopo.lastChannel();
      for(int bin=firstWire; bin<=lastWire; bin++){
	//from definition of "noisy cell"
	if((*lHisto).second->GetBinContent(bin)>500){
	  DTWireId wireID((*lHisto).first, bin);
	  statusMap->setCellNoise(wireID,1);
	}
      }
    }
  }
  cout << "Writing Noise Map object to DB!" << endl;
  string record = "DTStatusFlagRcd";
  DTCalibDBUtils::writeToDB<DTStatusFlag>(record, statusMap);

 
  /*
  //save the digi event plot per SuperLayer 
  bool histo=false;
  map<DTSuperLayerId, vector<int> > maxPerSuperLayer;
  int numPlot = (nevents/1000);
  int num=0;
  // loop over the numPlot 
  for(int i=0; i<numPlot; i++){  
    vector<DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_end = dtGeom->chambers().end();
    // Loop over the chambers
    for (; ch_it != ch_end; ++ch_it) {
      DTChamberId ch = (*ch_it)->id();
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
	double dummy = pow(10.,10.);
	maxPerSuperLayer[sl].push_back(0);
	// Loop over the Ls
	for(; l_it != l_end; ++l_it) {
	  DTLayerId layerId = (*l_it)->id();

	  if (theHistoEvtPerWireMap.find(layerId) != theHistoEvtPerWireMap.end() &&
	      theHistoEvtPerWireMap[layerId].size() > i){
	    if (theHistoEvtPerWireMap[layerId][i]->GetMaximum(dummy)>maxPerSuperLayer[sl][i])
	      maxPerSuperLayer[sl][i] = theHistoEvtPerWireMap[layerId][i]->GetMaximum(dummy);
	  }
	}
      } // loop over SLs
    } // loop over chambers
  } // loop over numPlot

  // loop over the numPlot 
  for(int i=0; i<numPlot; i++){
    vector<DTChamber*>::const_iterator chamber_it = dtGeom->chambers().begin();
    vector<DTChamber*>::const_iterator chamber_end = dtGeom->chambers().end();
    // Loop over the chambers
    for (; chamber_it != chamber_end; ++chamber_it) {
      DTChamberId ch = (*chamber_it)->id();
      vector<const DTSuperLayer*>::const_iterator sl_it = (*chamber_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*chamber_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();

	stringstream num; num << i;
	string canvasName = "c" + getSuperLayerName(sl) + "_" + num.str();
	TCanvas c1(canvasName.c_str(),canvasName.c_str(),600,780);
	TLegend *leg=new TLegend(0.5,0.6,0.7,0.8);
	for(; l_it != l_end; ++l_it) {
	  DTLayerId layerId = (*l_it)->id();

	  if (theHistoEvtPerWireMap.find(layerId) != theHistoEvtPerWireMap.end() &&
	    theHistoEvtPerWireMap[layerId].size() > i){

	    string TitleName = "DigiPerWirePerEvent_" + getSuperLayerName(sl) + "_" + num.str();
	    theHistoEvtPerWireMap[layerId][i]->SetTitle(TitleName.c_str());
	    stringstream layer; layer << layerId.layer();	
	    string legendHisto = "layer " + layer.str();
	    leg->AddEntry(theHistoEvtPerWireMap[layerId][i],legendHisto.c_str(),"L");
	    theHistoEvtPerWireMap[layerId][i]->SetMaximum(maxPerSuperLayer[sl][i]);
	    if(histo==false)
	      theHistoEvtPerWireMap[layerId][i]->Draw("lego");
	    else
	      theHistoEvtPerWireMap[layerId][i]->Draw("same , lego");
	    theHistoEvtPerWireMap[layerId][i]->SetLineColor(layerId.layer());
	    histo=true;
	  }
	} // loop over Ls
	if(histo){
	  leg->Draw("same");
	  theFile->cd();
	  c1.Write();
	}
	histo=false;
      } // loop over SLs
    } // loop over chambers 
  } // loop over numPlot
  */

}



DTNoiseCalibration::~DTNoiseCalibration(){

  cout << "DTNoiseCalibration: analyzed " << nevents << " events" <<endl;
  theFile->Close();

}



string DTNoiseCalibration::getLayerName(const DTLayerId& lId) const {

  const  DTSuperLayerId dtSLId = lId.superlayerId();
  const  DTChamberId dtChId = dtSLId.chamberId(); 
  stringstream Layer; Layer << lId.layer();
  stringstream superLayer; superLayer << dtSLId.superlayer();
  stringstream wheel; wheel << dtChId.wheel();	
  stringstream station; station << dtChId.station();	
  stringstream sector; sector << dtChId.sector();
  
  string LayerName = 
    "W" + wheel.str()
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str()
    + "_L" + Layer.str();
  
  return LayerName;

}


string DTNoiseCalibration::getSuperLayerName(const DTSuperLayerId& dtSLId) const {

  const  DTChamberId dtChId = dtSLId.chamberId(); 
  stringstream superLayer; superLayer << dtSLId.superlayer();
  stringstream wheel; wheel << dtChId.wheel();	
  stringstream station; station << dtChId.station();	
  stringstream sector; sector << dtChId.sector();
  
  string SuperLayerName = 
    "W" + wheel.str()
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str();
  
  return SuperLayerName;

}



  
