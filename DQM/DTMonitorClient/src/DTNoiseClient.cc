/*
 * \file DTNoiseClient.cc
 * 
 * $Date: 2006/06/30 15:31:06 $
 * $Revision: 1.4 $
 * \author M. Zanetti - INFN Padova
 *
 */

#include "DQM/DTMonitorClient/interface/DTNoiseClient.h"

// DQM
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

// Geometry
#include "DataFormats/MuonDetId/interface/DTWireId.h"

// ROOT Staff
#include "TROOT.h"
#include "TCanvas.h"
#include "TText.h"
#include "TStyle.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>

using namespace edm;
using namespace std;

DTNoiseClient::DTNoiseClient() {

  // dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  summaryAverage_W2_Se10 = TH2F("summaryAverage_W2_Se10","Average Noise YB2_Sector10",4,1,5,15,1,16);
  summaryAverage_W2_Se11 = TH2F("summaryAverage_W2_Se11","Average Noise YB2_Sector11",4,1,5,15,1,16);
  summaryAverage_W1_Se10 = TH2F("summaryAverage_W1_Se10","Average Noise YB1_Sector10",4,1,5,15,1,16);

  summaryNoiseChs_W2_Se10 = TH2F("summaryNoiseChs_W2_Se10","Noisy Channels YB2_Sector10",4,1,5,15,1,16);
  summaryNoiseChs_W2_Se11 = TH2F("summaryNoiseChs_W2_Se11","Noisy Channels YB2_Sector11",4,1,5,15,1,16);
  summaryNoiseChs_W1_Se10 = TH2F("summaryNoiseChs_W1_Se10","Noisy Channels YB1_Sector10",4,1,5,15,1,16);

}


DTNoiseClient::~DTNoiseClient() {

  DaqMonitorBEInterface* dbe = mui->getBEInterface();
  dbe->save("NoiseSummary.root");

}


void DTNoiseClient::bookHistos(const DTLayerId& dtLayer) {

  stringstream wheel; wheel << dtLayer.wheel();	
  stringstream sector; sector << dtLayer.sector();	

  DaqMonitorBEInterface* dbe = mui->getBEInterface();

  dbe->setCurrentFolder("DT/DTDigiClient/Summary");

  int code;
  if ( dtLayer.sector() != 14 )
    code = dtLayer.sector()*10 + dtLayer.wheel();
  else code = 10 + dtLayer.wheel();

  string histoName =  "summaryAverage_W" + wheel.str() + "_Se" + sector.str();
  noiseAverageHistos[code] = dbe->book2D(histoName,histoName,4,1,5,15,1,16);
  histoName =  "summaryNoiseChs_W" + wheel.str() + "_Se" + sector.str();
  noiseChHistos[code] = dbe->book2D(histoName,histoName,4,1,5,15,1,16);


}


void DTNoiseClient::performCheck(MonitorUserInterface * mui_) {

  // Get the MonitorUserInterface
  mui = mui_;
  
  /// FIXME: ES missing, no way to get the geometry 
  /// fake loop 
  for (int w=-2; w<=2; w++) {
    stringstream wheel; wheel  << w;
    for (int st=1; st<=4; st++) {
      stringstream station; station  << st;
      for (int se=1; se<=14; se++) {
	stringstream sector; sector  << se;

	/// WARNING: Pay attantion to the FU number!!!
	string folder = "Collector/FU0/DT/DTDigiTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/Occupancies/Noise/";

	for (int sl=1; sl<=3; sl++) {
	  if (st==4 && sl==2) continue;
	  stringstream superLayer; superLayer  << sl;
	  for (int l=1; l<=4; l++) {
	    stringstream layer; layer  << sl;
	    
	    string histoName = folder + 
	      + "OccupancyNoise_W" + wheel.str() 
	      + "_St" + station.str() 
	      + "_Sec" + sector.str() 
	      + "_SL" + superLayer.str() 
	      + "_L" + layer.str();
	    
	    MonitorElement * noise = mui->get(histoName);

	    if (noise) {
	      
	      // get the NoisyChannels report
	      string criterionName = "NoisyChannels"; // this as to be the same in QualityTests.xml
	      const QReport * theQReport = noise->getQReport(criterionName);
	      if(theQReport) {
		
		vector<dqm::me_util::Channel> badChannels = theQReport->getBadChannels();
		for (vector<dqm::me_util::Channel>::iterator ch_it = badChannels.begin(); 
		     ch_it != badChannels.end(); ch_it++) {
		  
		  noisyChannelsStatistics[DTLayerId(w,st,se,sl,l)]++;
 		  		  
		}
	      }
	      
	      // noise average per layer
	      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>(noise);
	      if (ob) {
		
		TH1F * noiseT = dynamic_cast<TH1F*> (ob->operator->());
		if (noiseT) {

		  float average=0;
		  for (int i = 1; i <= noiseT->GetNbinsX(); i++){
		    average += noiseT->GetBinContent(i); 
		  }
		  noiseStatistics[DTLayerId(w,st,se,sl,l)] =  average/noiseT->GetNbinsX();
		}
	      }

	    }

	  }
	}
      }
    }
  }

  drawSummaryNoise();

}


void DTNoiseClient::drawSummaryNoise() {



  /// Filling
  for (map<DTLayerId,float>::iterator ns_it = noiseStatistics.begin();
       ns_it != noiseStatistics.end(); ns_it++) {

    int code;
    if ( ((*ns_it).first).sector() != 14 )
      code = ((*ns_it).first).sector()*10+((*ns_it).first).wheel();
    else code = 10 +((*ns_it).first).wheel();
    
    if (noiseAverageHistos.find(code) != noiseAverageHistos.end() ) {
      noiseAverageHistos.find(code)->second->Fill(((*ns_it).first).station() ,
						  5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
						  ((*ns_it).second));
    }
    else {
      bookHistos((*ns_it).first);
      noiseAverageHistos.find(code)->second->Fill(((*ns_it).first).station() ,
						  5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
						  ((*ns_it).second));
    }
      

    if ( ((*ns_it).first).superlayerId().chamberId().wheel() == 2 ) {
      if ( ((*ns_it).first).sector() == 10 ||((*ns_it).first).sector() == 14 ) 
	summaryAverage_W2_Se10.Fill( ((*ns_it).first).station() ,
				     5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
				     ((*ns_it).second));
      
      if ( ((*ns_it).first).sector() == 11 )
	summaryAverage_W2_Se11.Fill( ((*ns_it).first).station() ,
				     5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
				     ((*ns_it).second));
    }
    
    if ( ((*ns_it).first).superlayerId().chamberId().wheel() == 1 ) 
      summaryAverage_W1_Se10.Fill( ((*ns_it).first).station() ,
				   5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
				   ((*ns_it).second));
  }

  for (map<DTLayerId,int>::iterator ns_it = noisyChannelsStatistics.begin();
       ns_it != noisyChannelsStatistics.end(); ns_it++) {


    int code;
    if ( ((*ns_it).first).sector() != 14 )
      code = ((*ns_it).first).sector()*10+((*ns_it).first).wheel();
    else code = 10 +((*ns_it).first).wheel();
    
    if (noiseChHistos.find(code) != noiseChHistos.end() ) {
      noiseChHistos.find(code)->second->Fill(((*ns_it).first).station() ,
						  5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
						  ((*ns_it).second));
    }
    else {
      bookHistos((*ns_it).first);
      noiseChHistos.find(code)->second->Fill(((*ns_it).first).station() ,
						  5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
						  ((*ns_it).second));
    }
 


    if ( ((*ns_it).first).superlayerId().chamberId().wheel() == 2 ) {
      if ( ((*ns_it).first).sector() == 10 ||((*ns_it).first).sector() == 14 ) 
	summaryNoiseChs_W2_Se10.Fill( ((*ns_it).first).station() ,
				     5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
				      ((*ns_it).second));
      
      if ( ((*ns_it).first).sector() == 11 )
	summaryNoiseChs_W2_Se11.Fill( ((*ns_it).first).station() ,
				     5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
				      ((*ns_it).second));
    }
    
    if ( ((*ns_it).first).superlayerId().chamberId().wheel() == 1 ) 
      summaryNoiseChs_W1_Se10.Fill( ((*ns_it).first).station() ,
				    5*( ((*ns_it).first).superLayer()-1) + ((*ns_it).first).layer(),
				    ((*ns_it).second));
    
  }

  
  /// Drawing
  TCanvas noiseCanvas("noiseCanvas","noiseCanvas",50,0,1000,900);
  noiseCanvas.SetFillColor(10);
  noiseCanvas.Divide(3,2);

  gStyle->SetPalette(1);
  noiseCanvas.cd(1);
  summaryNoiseChs_W2_Se10.Draw("colz"); noiseCanvas.Update();
  noiseCanvas.cd(2);
  summaryNoiseChs_W2_Se11.Draw("colz"); noiseCanvas.Update();
  noiseCanvas.cd(3);
  summaryNoiseChs_W1_Se10.Draw("colz"); noiseCanvas.Update();

  noiseCanvas.cd(4);
  summaryAverage_W2_Se10.Draw("colz"); noiseCanvas.Update();
  noiseCanvas.cd(5);
  summaryAverage_W2_Se11.Draw("colz"); noiseCanvas.Update();
  noiseCanvas.cd(6);
  summaryAverage_W1_Se10.Draw("colz"); noiseCanvas.Update();

  noiseCanvas.Draw();
  noiseCanvas.SaveAs("NoiseSummary.jpg");

}

