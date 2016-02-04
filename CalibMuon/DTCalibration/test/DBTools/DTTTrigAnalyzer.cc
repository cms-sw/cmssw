
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/11/19 19:08:19 $
 *  $Revision: 1.12 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "DTTTrigAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include <iostream>
#include "TFile.h"
#include "TH1D.h"
#include "TString.h"

using namespace edm;
using namespace std;

DTTTrigAnalyzer::DTTTrigAnalyzer(const ParameterSet& pset) {
  // The root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
  //The k factor to compute ttrig
  //kfactor = pset.getUntrackedParameter<double>("kfactor",0);
  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");

}
 
DTTTrigAnalyzer::~DTTTrigAnalyzer(){  
  theFile->Close();
}

void DTTTrigAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& eventSetup) {
  ESHandle<DTTtrig> tTrig;
  eventSetup.get<DTTtrigRcd>().get(dbLabel,tTrig);
  tTrigMap = &*tTrig;
  cout << "[DTTTrigAnalyzer] TTrig version: " << tTrig->version() << endl;
}

void DTTTrigAnalyzer::endJob() {
//   static const double convToNs = 25./32.;
   // Loop over DB entries
  for(DTTtrig::const_iterator ttrig = tTrigMap->begin();
	  ttrig != tTrigMap->end(); ttrig++) {
    DTWireId wireId((*ttrig).first.wheelId,
		    (*ttrig).first.stationId,
		    (*ttrig).first.sectorId,
		    (*ttrig).first.slId, 0, 0);
    float tmean;
    float sigma;
    float kFactor;

    DetId detId( wireId.rawId() );
    // ttrig and rms are ns
    tTrigMap->get(detId, tmean, sigma, kFactor, DTTimeUnits::ns);
    float ttcor = tmean + kFactor * sigma; 
    cout << "Wire: " <<  wireId <<endl
	 << " Ttrig (ns): " << ttcor << endl
	 << " tmean (ns): " << tmean << endl
	 << " sigma (ns): " << sigma << endl
         << " kfactor: " << kFactor << endl;


    //Define an histo for each wheel and each superlayer type
    TH1D *hTTrigHisto = theTTrigHistoMap[make_pair(wireId.wheel(),wireId.superlayer())];
    TH1D *hTMeanHisto = theTMeanHistoMap[make_pair(wireId.wheel(),wireId.superlayer())];
    TH1D *hSigmaHisto = theSigmaHistoMap[make_pair(wireId.wheel(),wireId.superlayer())];
    TH1D *hKFactorHisto = theKFactorHistoMap[make_pair(wireId.wheel(),wireId.superlayer())];
    if(hTTrigHisto == 0) {
      theFile->cd();
      TString name = getHistoName(wireId).c_str();
      if(wireId.superlayer() != 2){
	hTTrigHisto = new TH1D(name+"_TTrig",
			       "TTrig calibrated from TB per superlayer", 50, 0, 50);
 	hTMeanHisto = new TH1D(name+"_TMean",
			       "TMean calibrated from TB per superlayer", 50, 0, 50);
  	hSigmaHisto = new TH1D(name+"_Sigma",
			       "Sigma calibrated from TB per superlayer", 50, 0, 50);
        hKFactorHisto = new TH1D(name+"_KFactor",
                               "KFactor calibrated from TB per superlayer", 50, 0, 50); 
      }
      else{
 	hTTrigHisto = new TH1D(name+"_TTrig",
			       "TTrig calibrated from TB per superlayer", 36, 0, 36);
 	hTMeanHisto = new TH1D(name+"_TMean",
			       "TMean calibrated from TB per superlayer", 36, 0, 36);
  	hSigmaHisto = new TH1D(name+"_Sigma",
			       "Sigma calibrated from TB per superlayer", 36, 0, 36);
        hKFactorHisto = new TH1D(name+"_KFactor",
                               "KFactor calibrated from TB per superlayer", 36, 0, 36);
      }
      theTTrigHistoMap[make_pair(wireId.wheel(),wireId.superlayer())] = hTTrigHisto;
      theTMeanHistoMap[make_pair(wireId.wheel(),wireId.superlayer())] = hTMeanHisto;
      theSigmaHistoMap[make_pair(wireId.wheel(),wireId.superlayer())] = hSigmaHisto;
      theKFactorHistoMap[make_pair(wireId.wheel(),wireId.superlayer())] = hKFactorHisto;
    }

    //Fill the histos and set the bin label
    int binNumber = wireId.sector() + 12 * (wireId.station() - 1);
//    hTTrigHisto->SetBinContent(binNumber,ttrig);  
    hTTrigHisto->SetBinContent(binNumber,ttcor);  
    hTMeanHisto->SetBinContent(binNumber,tmean);  
    hSigmaHisto->SetBinContent(binNumber,sigma);  
    hKFactorHisto->SetBinContent(binNumber,kFactor);
    string labelName;
    stringstream theStream;
    if(wireId.sector() == 1)
      theStream << "MB" << wireId.station() << "_Sec" << wireId.sector(); 
    else
      theStream << "Sec" << wireId.sector(); 
    theStream >> labelName;
    hTTrigHisto->GetXaxis()->SetBinLabel(binNumber,labelName.c_str());  
    hTMeanHisto->GetXaxis()->SetBinLabel(binNumber,labelName.c_str());  
    hSigmaHisto->GetXaxis()->SetBinLabel(binNumber,labelName.c_str());
    hKFactorHisto->GetXaxis()->SetBinLabel(binNumber,labelName.c_str());  

    //Define a distribution for each wheel,station and each superlayer type
    vector<int> Wh_St_SL;
    Wh_St_SL.push_back(wireId.wheel());
    Wh_St_SL.push_back(wireId.station());
    Wh_St_SL.push_back(wireId.superlayer());
    TH1D *hTTrigDistrib = theTTrigDistribMap[Wh_St_SL];
    TH1D *hTMeanDistrib = theTMeanDistribMap[Wh_St_SL];
    TH1D *hSigmaDistrib = theSigmaDistribMap[Wh_St_SL];
    TH1D *hKFactorDistrib = theKFactorDistribMap[Wh_St_SL];
    if(hTTrigDistrib == 0) {
      theFile->cd();
      TString name = getDistribName(wireId).c_str();
      hTTrigDistrib = new TH1D(name+"_TTrig",
			       "TTrig calibrated from TB per superlayer",  500, 100, 600);
      hTMeanDistrib = new TH1D(name+"_TMean",
			       "TMean calibrated from TB per superlayer", 500, 100, 600);
      hSigmaDistrib = new TH1D(name+"_Sigma",
			       "Sigma calibrated from TB per superlayer", 50, 0, 50);
      hKFactorDistrib = new TH1D(name+"_KFactor",
                               "KFactor calibrated from TB per superlayer", 200, -10.0, 10.0);
      theTTrigDistribMap[Wh_St_SL] = hTTrigDistrib;
      theTMeanDistribMap[Wh_St_SL] = hTMeanDistrib;
      theSigmaDistribMap[Wh_St_SL] = hSigmaDistrib;
      theKFactorDistribMap[Wh_St_SL] = hKFactorDistrib;
    }
    //Fill the distributions
//    hTTrigDistrib->Fill(ttrig);
    hTTrigDistrib->Fill(ttcor);
    hTMeanDistrib->Fill(tmean);
    hSigmaDistrib->Fill(sigma);
    hKFactorDistrib->Fill(kFactor);
  }

  //Write histos in a .root file
  theFile->cd();
  for(map<pair<int,int>, TH1D*>::const_iterator lHisto = theTTrigHistoMap.begin();
      lHisto != theTTrigHistoMap.end();
      lHisto++) {
    (*lHisto).second->GetXaxis()->LabelsOption("v");
    (*lHisto).second->Write(); 
  }    
  for(map<pair<int,int>, TH1D*>::const_iterator lHisto = theTMeanHistoMap.begin();
      lHisto != theTMeanHistoMap.end();
      lHisto++) {
    (*lHisto).second->GetXaxis()->LabelsOption("v");
    (*lHisto).second->Write(); 
  }    
  for(map<pair<int,int>, TH1D*>::const_iterator lHisto = theSigmaHistoMap.begin();
      lHisto != theSigmaHistoMap.end();
      lHisto++) {
    (*lHisto).second->GetXaxis()->LabelsOption("v");
    (*lHisto).second->Write(); 
  }
  for(map<pair<int,int>, TH1D*>::const_iterator lHisto = theKFactorHistoMap.begin();
      lHisto != theKFactorHistoMap.end();
      lHisto++) {
    (*lHisto).second->GetXaxis()->LabelsOption("v");
    (*lHisto).second->Write();
  } 
  for(map<vector<int>, TH1D*>::const_iterator lDistrib = theTTrigDistribMap.begin();
      lDistrib != theTTrigDistribMap.end();
      lDistrib++) {
    (*lDistrib).second->Write(); 
  }    
  for(map<vector<int>, TH1D*>::const_iterator lDistrib = theTMeanDistribMap.begin();
      lDistrib != theTMeanDistribMap.end();
      lDistrib++) {
    (*lDistrib).second->Write(); 
  }    
  for(map<vector<int>, TH1D*>::const_iterator lDistrib = theSigmaDistribMap.begin();
      lDistrib != theSigmaDistribMap.end();
      lDistrib++) {
    (*lDistrib).second->Write(); 
  }
  for(map<vector<int>, TH1D*>::const_iterator lDistrib = theKFactorDistribMap.begin();
      lDistrib != theKFactorDistribMap.end();
      lDistrib++) {
    (*lDistrib).second->Write();
  }  

}

string DTTTrigAnalyzer::getHistoName(const DTWireId& wId) const {
  string histoName;
  stringstream theStream;
  theStream << "Wheel" << wId.wheel() << "_SL" << wId.superlayer(); 
  theStream >> histoName;
  return histoName;
}

string DTTTrigAnalyzer::getDistribName(const DTWireId& wId) const {
  string histoName;
  stringstream theStream;
  theStream << "Wheel" << wId.wheel() <<"_Station"<< wId.station() << "_SL" << wId.superlayer(); 
  theStream >> histoName;
  return histoName;
}
