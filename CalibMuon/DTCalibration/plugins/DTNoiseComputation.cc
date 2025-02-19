/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/01/12 15:05:33 $
 *  $Revision: 1.7 $
 *  \author G. Mila - INFN Torino
 */


#include "CalibMuon/DTCalibration/plugins/DTNoiseComputation.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

// Framework
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/MakerMacros.h>

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

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TF1.h"
#include "TProfile.h"
#include "TPostScript.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace edm;
using namespace std;


DTNoiseComputation::DTNoiseComputation(const edm::ParameterSet& ps){

  cout << "[DTNoiseComputation]: Constructor" <<endl;

  // Get the debug parameter for verbose output
  debug = ps.getUntrackedParameter<bool>("debug");

  // The analysis type
  fastAnalysis = ps.getUntrackedParameter<bool>("fastAnalysis", true);

  // The root file which contain the histos
  string rootFileName = ps.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "READ");

  // The new root file which contain the histos
  string newRootFileName = ps.getUntrackedParameter<string>("newRootFileName");
  theNewFile = new TFile(newRootFileName.c_str(), "RECREATE");

  // The maximum number of events to analyze
  MaxEvents = ps.getUntrackedParameter<int>("MaxEvents");

}

void DTNoiseComputation::beginRun(const edm::Run&, const EventSetup& setup)
{
  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

  static int count = 0;

  if(count == 0){
    string CheckHistoName;
  
    TH1F *hOccHisto;
    TH1F *hAverageNoiseHisto;
    TH1F *hAverageNoiseIntegratedHisto;
    TH1F *hAverageNoiseHistoPerCh;
    TH1F *hAverageNoiseIntegratedHistoPerCh;
    TH2F *hEvtHisto;
    string HistoName;
    string Histo2Name;
    string AverageNoiseName;
    string AverageNoiseIntegratedName;
    string AverageNoiseNamePerCh;
    string AverageNoiseIntegratedNamePerCh;
    TH1F *hnoisyC;
    TH1F *hsomeHowNoisyC;
  
    // Loop over all the chambers 	 
    vector<DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin(); 	 
    vector<DTChamber*>::const_iterator ch_end = dtGeom->chambers().end(); 	 
    // Loop over the SLs 	 
    for (; ch_it != ch_end; ++ch_it) { 	 
      DTChamberId ch = (*ch_it)->id(); 	 
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 	 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end(); 	 
      // Loop over the SLs 	 
      for(; sl_it != sl_end; ++sl_it) { 	 
	//      DTSuperLayerId sl = (*sl_it)->id(); 	 
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 	 
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end(); 	 
	// Loop over the Ls 	 
	for(; l_it != l_end; ++l_it) { 	 
	  DTLayerId dtLId = (*l_it)->id();
	
	  //check if the layer has digi
	  theFile->cd();
	  CheckHistoName =  "DigiOccupancy_" + getLayerName(dtLId);
	  TH1F *hCheckHisto = (TH1F *) theFile->Get(CheckHistoName.c_str());
	  if(hCheckHisto){  
	    delete hCheckHisto;
	    stringstream wheel; wheel << ch.wheel();	
	    stringstream station; station << ch.station();
	  
	    if(someHowNoisyC.find(make_pair(ch.wheel(),ch.station())) == someHowNoisyC.end()) {
	      TString histoName_someHowNoisy = "somehowNoisyCell_W"+wheel.str()+"_St"+station.str();
	      hsomeHowNoisyC = new TH1F(histoName_someHowNoisy,histoName_someHowNoisy,getMaxNumBins(ch),1,getMaxNumBins(ch)+1);
	      someHowNoisyC[make_pair(ch.wheel(),ch.station())]=hsomeHowNoisyC;
	    }
	  
	    if(noisyC.find(make_pair(ch.wheel(),ch.station())) == noisyC.end()) {
	      TString histoName_noisy = "noisyCell_W"+wheel.str()+"_St"+station.str();
	      hnoisyC = new TH1F(histoName_noisy,histoName_noisy,getMaxNumBins(ch),1,getMaxNumBins(ch)+1);
	      noisyC[make_pair(ch.wheel(),ch.station())]=hnoisyC;
	    }
	  
	    //to fill a map with the average noise per wire and fill new noise histo	  
	    if(AvNoisePerSuperLayer.find(dtLId.superlayerId()) == AvNoisePerSuperLayer.end()) {
	      AverageNoiseName = "AverageNoise_" + getSuperLayerName(dtLId.superlayerId());
	      hAverageNoiseHisto = new TH1F(AverageNoiseName.c_str(), AverageNoiseName.c_str(), 200, 0, 10000);
	      AverageNoiseIntegratedName = "AverageNoiseIntegrated_" + getSuperLayerName(dtLId.superlayerId());
	      hAverageNoiseIntegratedHisto = new TH1F(AverageNoiseIntegratedName.c_str(), AverageNoiseIntegratedName.c_str(), 200, 0, 10000);
	      AvNoisePerSuperLayer[dtLId.superlayerId()] = hAverageNoiseHisto;
	      AvNoiseIntegratedPerSuperLayer[dtLId.superlayerId()] = hAverageNoiseIntegratedHisto;
	      if(debug){
		cout << "  New Average Noise Histo per SuperLayer : " << hAverageNoiseHisto->GetName() << endl;
		cout << "  New Average Noise Integrated Histo per SuperLayer : " << hAverageNoiseHisto->GetName() << endl;
	      }
	    }
	    if(AvNoisePerChamber.find(dtLId.superlayerId().chamberId()) == AvNoisePerChamber.end()) {
	      AverageNoiseNamePerCh = "AverageNoise_" + getChamberName(dtLId);
	      hAverageNoiseHistoPerCh = new TH1F(AverageNoiseNamePerCh.c_str(), AverageNoiseNamePerCh.c_str(), 200, 0, 10000);
	      AverageNoiseIntegratedNamePerCh = "AverageNoiseIntegrated_" + getChamberName(dtLId);
	      hAverageNoiseIntegratedHistoPerCh = new TH1F(AverageNoiseIntegratedNamePerCh.c_str(), AverageNoiseIntegratedNamePerCh.c_str(), 200, 0, 10000);
	      AvNoisePerChamber[dtLId.superlayerId().chamberId()] = hAverageNoiseHistoPerCh;
	      AvNoiseIntegratedPerChamber[dtLId.superlayerId().chamberId()] = hAverageNoiseIntegratedHistoPerCh;
	      if(debug)
		cout << "  New Average Noise Histo per chamber : " << hAverageNoiseHistoPerCh->GetName() << endl;
	    }
	  
	    HistoName = "DigiOccupancy_" + getLayerName(dtLId);
	    theFile->cd();
	    hOccHisto = (TH1F *) theFile->Get(HistoName.c_str());
	    int numBin = hOccHisto->GetXaxis()->GetNbins(); 
	    for (int bin=1; bin<=numBin; bin++) {
	      DTWireId wireID(dtLId, bin);
	      theAverageNoise[wireID]= hOccHisto->GetBinContent(bin);
	      if(theAverageNoise[wireID] != 0) {
		AvNoisePerSuperLayer[dtLId.superlayerId()]->Fill(theAverageNoise[wireID]);
		AvNoisePerChamber[dtLId.superlayerId().chamberId()]->Fill(theAverageNoise[wireID]);
	      }
	    }	      
	  
	    //to compute the average noise per layer (excluding the noisy cells)
	    double numCell=0;
	    double AvNoise=0;	     
	    HistoName = "DigiOccupancy_" + getLayerName(dtLId);
	    theFile->cd();
	    hOccHisto = (TH1F *) theFile->Get(HistoName.c_str());
	    numBin = hOccHisto->GetXaxis()->GetNbins(); 
	    for (int bin=1; bin<=numBin; bin++) {
	      DTWireId wireID(dtLId, bin);
	      theAverageNoise[wireID]= hOccHisto->GetBinContent(bin);
	      if(hOccHisto->GetBinContent(bin)<100){
		numCell++;
		AvNoise += hOccHisto->GetBinContent(bin);
	      }
	      if(hOccHisto->GetBinContent(bin)>100 && hOccHisto->GetBinContent(bin)<500){
		someHowNoisyC[make_pair(ch.wheel(),ch.station())]->Fill(bin);
		cout<<"filling somehow noisy cell"<<endl;
	      }
	      if(hOccHisto->GetBinContent(bin)>500){
		noisyC[make_pair(ch.wheel(),ch.station())]->Fill(bin);
		cout<<"filling noisy cell"<<endl;
	      }
	    }
	    AvNoise = AvNoise/numCell;
	    cout<<"theAverageNoise for layer "<<getLayerName(dtLId)<<" is : "<<AvNoise << endl;

	  
	    // book the digi event plots every 1000 events
	    int updates = MaxEvents/1000; 
	    for(int evt=0; evt<updates; evt++){
	      stringstream toAppend; toAppend << evt;
	      Histo2Name = "DigiPerWirePerEvent_" + getLayerName(dtLId) + "_" + toAppend.str();
	      theFile->cd();
	      hEvtHisto = (TH2F *) theFile->Get(Histo2Name.c_str());
	      if(hEvtHisto){
		if(debug)
		  cout << "  New Histo with the number of events per evt per wire: " << hEvtHisto->GetName() << endl;
		theEvtMap[dtLId].push_back(hEvtHisto);
	      }
	    }
	  
	  }// done if the layer has digi
	}// loop over layers
      }// loop over superlayers
    }// loop over chambers

    count++;
  }

}
    
void DTNoiseComputation::endJob(){

  cout << "[DTNoiseComputation] endjob called!" <<endl;
  TH1F *hEvtDistance=0;
  TF1 *ExpoFit = new TF1("ExpoFit","expo", 0.5, 1000.5);
  ExpoFit->SetMarkerColor();//just silence gcc complaining about unused vars
  TF1 *funct=0;
  TProfile *theNoiseHisto = new TProfile("theNoiseHisto","Time Constant versus Average Noise",100000,0,100000);
  

  //compute the time constant
  for(map<DTLayerId, vector<TH2F*> >::const_iterator lHisto = theEvtMap.begin();
      lHisto != theEvtMap.end();
      lHisto++) {
    for(int bin=1; bin<(*lHisto).second[0]->GetYaxis()->GetNbins(); bin++){
      int distanceEvt = 1;
      DTWireId wire((*lHisto).first, bin);
      for(int i=0; i<int((*lHisto).second.size()); i++){
	for(int evt=1; evt<=(*lHisto).second[i]->GetXaxis()->GetNbins(); evt++){
	  if((*lHisto).second[i]->GetBinContent(evt,bin) == 0) distanceEvt++;
	  else { 
	    if(toDel.find(wire) == toDel.end()) {
	      toDel[wire] = false;
	      stringstream toAppend; toAppend << bin;
	      string Histo = "EvtDistancePerWire_" + getLayerName((*lHisto).first) + "_" + toAppend.str();
	      hEvtDistance = new TH1F(Histo.c_str(),Histo.c_str(), 50000,0.5,50000.5);
	    }
	    hEvtDistance->Fill(distanceEvt); 
	    distanceEvt=1;
	  }
	}
      }
      if(toDel.find(wire) != toDel.end()){
	theHistoEvtDistancePerWire[wire] =  hEvtDistance;
	theNewFile->cd();
	theHistoEvtDistancePerWire[wire]->Fit("ExpoFit","R");
	funct = theHistoEvtDistancePerWire[wire]->GetFunction("ExpoFit");
	double par0 = funct->GetParameter(0);
	double par1 = funct->GetParameter(1);
	cout<<"par0: "<<par0<<"  par1: "<<par1<<endl;
	double chi2rid = funct->GetChisquare()/funct->GetNDF();
	if(chi2rid>10)
	  theTimeConstant[wire]=1;
	else
	  theTimeConstant[wire]=par1;
	toDel[wire] = true;
	theHistoEvtDistancePerWire[wire]->Write();
	delete hEvtDistance;
      }
    }
  }

  if(!fastAnalysis){
    //fill the histo with the time constant as a function of the average noise
    for(map<DTWireId, double>::const_iterator AvNoise = theAverageNoise.begin();
	AvNoise != theAverageNoise.end();
	AvNoise++) {
      DTWireId wire = (*AvNoise).first;
      theNoiseHisto->Fill((*AvNoise).second, theTimeConstant[wire]);
      cout<<"Layer: "<<getLayerName(wire.layerId())<<"  wire: "<<wire.wire()<<endl;
      cout<<"The Average noise: "<<(*AvNoise).second<<endl;
      cout<<"The time constant: "<<theTimeConstant[wire]<<endl;
    }
    theNewFile->cd();
    theNoiseHisto->Write();
  }  


  // histos with the integrated noise per layer
  int numBin;
  double integratedNoise, bin, halfBin, maxBin;
  for(map<DTSuperLayerId, TH1F*>::const_iterator AvNoiseHisto = AvNoisePerSuperLayer.begin();
      AvNoiseHisto != AvNoisePerSuperLayer.end();
      AvNoiseHisto++) {
    integratedNoise=0;
    numBin = (*AvNoiseHisto).second->GetXaxis()->GetNbins();
    maxBin = (*AvNoiseHisto).second->GetXaxis()->GetXmax();
    bin= double(maxBin/numBin);
    halfBin=double(bin/2);
    theNewFile->cd();
    (*AvNoiseHisto).second->Write();
    for(int i=1; i<numBin; i++){
      integratedNoise+=(*AvNoiseHisto).second->GetBinContent(i);
      AvNoiseIntegratedPerSuperLayer[(*AvNoiseHisto).first]->Fill(halfBin,integratedNoise);
      halfBin+=bin;
    }
    theNewFile->cd();
    AvNoiseIntegratedPerSuperLayer[(*AvNoiseHisto).first]->Write(); 
  }
  // histos with the integrated noise per chamber
  for(map<DTChamberId, TH1F*>::const_iterator AvNoiseHisto = AvNoisePerChamber.begin();
      AvNoiseHisto != AvNoisePerChamber.end();
      AvNoiseHisto++) {
    integratedNoise=0;
    numBin = (*AvNoiseHisto).second->GetXaxis()->GetNbins();
    maxBin = (*AvNoiseHisto).second->GetXaxis()->GetXmax();
    bin= maxBin/numBin;
    halfBin=bin/2;
    theNewFile->cd();
    (*AvNoiseHisto).second->Write();
    for(int i=1; i<numBin; i++){
      integratedNoise+=(*AvNoiseHisto).second->GetBinContent(i);
      AvNoiseIntegratedPerChamber[(*AvNoiseHisto).first]->Fill(halfBin,integratedNoise);
      halfBin+=bin;
    } 
    theNewFile->cd();
    AvNoiseIntegratedPerChamber[(*AvNoiseHisto).first]->Write(); 
  }

  
  //overimpose the average noise histo
  bool histo=false;
  vector<DTChamber*>::const_iterator chamber_it = dtGeom->chambers().begin();
  vector<DTChamber*>::const_iterator chamber_end = dtGeom->chambers().end();
  // Loop over the chambers
  for (; chamber_it != chamber_end; ++chamber_it) {
    vector<const DTSuperLayer*>::const_iterator sl_it = (*chamber_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*chamber_it)->superLayers().end();
    // Loop over the SLs
    for(; sl_it != sl_end; ++sl_it) {
      DTSuperLayerId sl = (*sl_it)->id();
      vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
      vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();

      string canvasName = "c" + getSuperLayerName(sl); 
      TCanvas c1(canvasName.c_str(),canvasName.c_str(),600,780);
      TLegend *leg=new TLegend(0.5,0.6,0.7,0.8);
      for(; l_it != l_end; ++l_it) {
	DTLayerId layerId = (*l_it)->id();
	string HistoName = "DigiOccupancy_" + getLayerName(layerId);
	theFile->cd();
	TH1F *hOccHisto = (TH1F *) theFile->Get(HistoName.c_str());
	if(hOccHisto){
	  string TitleHisto = "AverageNoise_" + getSuperLayerName(sl);
	  cout<<"TitleHisto : "<<TitleHisto<<endl;
	  hOccHisto->SetTitle(TitleHisto.c_str());
	  stringstream layer; layer << layerId.layer();	
	  string legendHisto = "layer " + layer.str();
	  leg->AddEntry(hOccHisto,legendHisto.c_str(),"L");
	  hOccHisto->SetMaximum(getYMaximum(sl));
	  histo=true;
	  if(layerId.layer() == 1)
	    hOccHisto->Draw();
	  else
	    hOccHisto->Draw("same");
	  hOccHisto->SetLineColor(layerId.layer());
	}
      }
      if(histo){
	leg->Draw("same");
	theNewFile->cd();
	c1.Write();
      }
    }
    histo=false;   
  }

  //write on file the noisy plots
  for(map<pair<int,int>, TH1F*>::const_iterator nCell = noisyC.begin();
      nCell != noisyC.end();
      nCell++) {
    theNewFile->cd();
    (*nCell).second->Write();
  }
  for(map<pair<int,int>, TH1F*>::const_iterator somehownCell = someHowNoisyC.begin();
      somehownCell != someHowNoisyC.end();
      somehownCell++) {
    theNewFile->cd();
    (*somehownCell).second->Write();
  }
    
}


DTNoiseComputation::~DTNoiseComputation(){

  theFile->Close();
  theNewFile->Close();

}


string DTNoiseComputation::getLayerName(const DTLayerId& lId) const {

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


string DTNoiseComputation::getSuperLayerName(const DTSuperLayerId& dtSLId) const {

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


string DTNoiseComputation::getChamberName(const DTLayerId& lId) const {

  const  DTSuperLayerId dtSLId = lId.superlayerId();
  const  DTChamberId dtChId = dtSLId.chamberId(); 
  stringstream wheel; wheel << dtChId.wheel();	
  stringstream station; station << dtChId.station();	
  stringstream sector; sector << dtChId.sector();
  
  string ChamberName = 
    "W" + wheel.str()
    + "_St" + station.str() 
    + "_Sec" + sector.str();
  
  return ChamberName;

}


int DTNoiseComputation::getMaxNumBins(const DTChamberId& chId) const {
  
  int maximum=0;
  
  for(int SL=1; SL<=3; SL++){
    if(!(chId.station()==4 && SL==2)){
      for (int L=1; L<=4; L++){	
	DTLayerId layerId = DTLayerId(chId, SL, L);
	string HistoName = "DigiOccupancy_" + getLayerName(layerId);
	theFile->cd();
	TH1F *hOccHisto = (TH1F *) theFile->Get(HistoName.c_str());
	if(hOccHisto){
	  if (hOccHisto->GetXaxis()->GetXmax()>maximum) 
	    maximum = hOccHisto->GetXaxis()->GetNbins();
	}
      }
    }
  }
  return maximum;
}


double DTNoiseComputation::getYMaximum(const DTSuperLayerId& slId) const {
  
  double maximum=0;
  double dummy = pow(10.,10.);

  for (int L=1; L<=4; L++){
    DTLayerId layerId = DTLayerId(slId, L);
    string HistoName = "DigiOccupancy_" + getLayerName(layerId);
    theFile->cd();
    TH1F *hOccHisto = (TH1F *) theFile->Get(HistoName.c_str());
    if(hOccHisto){
      if (hOccHisto->GetMaximum(dummy)>maximum)
        maximum = hOccHisto->GetMaximum(dummy);
    }
  }
  return maximum;
}



