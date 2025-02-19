
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/06/21 08:49:20 $
 *  $Revision: 1.6 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "DTVDriftAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include <iostream>
#include "TFile.h"
#include "TH1D.h"
#include "TString.h"

using namespace edm;
using namespace std;

DTVDriftAnalyzer::DTVDriftAnalyzer(const ParameterSet& pset) {
  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
}
 
DTVDriftAnalyzer::~DTVDriftAnalyzer(){  
  theFile->Close();
}

void DTVDriftAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  ESHandle<DTMtime> mTime;
  eventSetup.get<DTMtimeRcd>().get(mTime);
  mTimeMap = &*mTime;
  cout << "[DTVDriftAnalyzer] MTime version: " << mTime->version() << endl;
 
}

void DTVDriftAnalyzer::endJob() {
   // Loop over DB entries
   for(DTMtime::const_iterator mtime = mTimeMap->begin();
       mtime != mTimeMap->end(); mtime++) {
     DTWireId wireId((*mtime).first.wheelId,
		     (*mtime).first.stationId,
		     (*mtime).first.sectorId,
		     (*mtime).first.slId, 0, 0);
    float vdrift;
    float reso;
    DetId detId( wireId.rawId() );
    // vdrift is cm/ns , resolution is cm
    mTimeMap->get(detId, vdrift, reso, DTVelocityUnits::cm_per_ns);
    cout << "Wire: " <<  wireId <<endl
	 << " vdrift (cm/ns): " << vdrift<<endl
	 << " reso (cm): " << reso<<endl;

    //Define an histo for each wheel and each superlayer type
    TH1D *hVDriftHisto = theVDriftHistoMap[make_pair(wireId.wheel(),wireId.superlayer())];
    TH1D *hResoHisto = theResoHistoMap[make_pair(wireId.wheel(),wireId.superlayer())];
    if(hVDriftHisto == 0) {
      theFile->cd();
      TString name = getHistoName(wireId).c_str();
      if(wireId.superlayer() != 2){
	hVDriftHisto = new TH1D(name+"_VDrift",
			       "VDrift calibrated from MT per superlayer", 50, 0, 50);
  	hResoHisto = new TH1D(name+"_Reso",
			       "Reso calibrated from MT per superlayer", 50, 0, 50);
      }
      else{
 	hVDriftHisto = new TH1D(name+"_VDrift",
			       "VDrift calibrated from MT per superlayer", 36, 0, 36);
  	hResoHisto = new TH1D(name+"_Reso",
			       "Reso calibrated from MT per superlayer", 36, 0, 36);
      }
      theVDriftHistoMap[make_pair(wireId.wheel(),wireId.superlayer())] = hVDriftHisto;
      theResoHistoMap[make_pair(wireId.wheel(),wireId.superlayer())] = hResoHisto;
    }

    //Fill the histos and set the bin label
    int binNumber = wireId.sector() + 12 * (wireId.station() - 1);
    hVDriftHisto->SetBinContent(binNumber,vdrift);  
    hResoHisto->SetBinContent(binNumber,reso);  
    string labelName;
    stringstream theStream;
    if(wireId.sector() == 1)
      theStream << "MB" << wireId.station() << "_Sec" << wireId.sector(); 
    else
      theStream << "Sec" << wireId.sector(); 
    theStream >> labelName;
    hVDriftHisto->GetXaxis()->SetBinLabel(binNumber,labelName.c_str());  
    hResoHisto->GetXaxis()->SetBinLabel(binNumber,labelName.c_str());  

    //Define a distribution for each wheel,station and each superlayer type
    vector<int> Wh_St_SL;
    Wh_St_SL.push_back(wireId.wheel());
    Wh_St_SL.push_back(wireId.station());
    Wh_St_SL.push_back(wireId.superlayer());
    TH1D *hVDriftDistrib = theVDriftDistribMap[Wh_St_SL];
    TH1D *hResoDistrib = theResoDistribMap[Wh_St_SL];
    if(hVDriftDistrib == 0) {
      theFile->cd();
      TString name = getDistribName(wireId).c_str();
      hVDriftDistrib = new TH1D(name+"_VDrift",
			       "VDrift calibrated from MT per superlayer", 100, 0.00530, 0.00580);
      hResoDistrib = new TH1D(name+"_Reso",
			       "Reso calibrated from MT per superlayer", 300, 0.015, 0.045);
      theVDriftDistribMap[Wh_St_SL] = hVDriftDistrib;
      theResoDistribMap[Wh_St_SL] = hResoDistrib;
    }
    //Fill the distributions
    hVDriftDistrib->Fill(vdrift);
    hResoDistrib->Fill(reso);
  }

  //Write histos in a .root file
  theFile->cd();
  for(map<pair<int,int>, TH1D*>::const_iterator lHisto = theVDriftHistoMap.begin();
      lHisto != theVDriftHistoMap.end();
      lHisto++) {
    (*lHisto).second->GetXaxis()->LabelsOption("v");
    (*lHisto).second->Write(); 
  }    
  for(map<pair<int,int>, TH1D*>::const_iterator lHisto = theResoHistoMap.begin();
      lHisto != theResoHistoMap.end();
      lHisto++) {
    (*lHisto).second->GetXaxis()->LabelsOption("v");
    (*lHisto).second->Write(); 
  } 
   for(map<vector<int>, TH1D*>::const_iterator lDistrib = theVDriftDistribMap.begin();
      lDistrib != theVDriftDistribMap.end();
      lDistrib++) {
    (*lDistrib).second->Write(); 
  }    
  for(map<vector<int>, TH1D*>::const_iterator lDistrib = theResoDistribMap.begin();
      lDistrib != theResoDistribMap.end();
      lDistrib++) {
    (*lDistrib).second->Write(); 
  }  

}

string DTVDriftAnalyzer::getHistoName(const DTWireId& wId) const {
  string histoName;
  stringstream theStream;
  theStream << "Wheel" << wId.wheel() << "_SL" << wId.superlayer(); 
  theStream >> histoName;
  return histoName;
}

string DTVDriftAnalyzer::getDistribName(const DTWireId& wId) const {
  string histoName;
  stringstream theStream;
  theStream << "Wheel" << wId.wheel() <<"_Station"<< wId.station() << "_SL" << wId.superlayer(); 
  theStream >> histoName;
  return histoName;
}
