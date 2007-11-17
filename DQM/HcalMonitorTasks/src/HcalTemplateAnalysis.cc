#include "DQM/HcalMonitorTasks/interface/HcalTemplateAnalysis.h"

HcalTemplateAnalysis::HcalTemplateAnalysis() {
}

HcalTemplateAnalysis::~HcalTemplateAnalysis() {
}

void HcalTemplateAnalysis::setup(const edm::ParameterSet& ps){
  
  outputFile_ = ps.getUntrackedParameter<string>("analysisFile", "");
  if ( outputFile_.size() != 0 ) 
    cout << "Hcal DQM analysis histograms will be saved to " << outputFile_.c_str() << endl;    
  else outputFile_ = "Hcal_DQM_Analysis.root";

  
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "RecHit eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "RecHit phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  rechitEnergy_HB = new TH1F("HB RecHit Energies","HB RecHit Energies",250,0,250);
  rechitTime_HB = new TH1F("HB RecHit Times","HB RecHit Times",250,0,250);
  rechitEnergy_HF = new TH1F("HF RecHit Energies","HF RecHit Energies",250,0,250);
  rechitTime_HF = new TH1F("HF RecHit Times","HF RecHit Times",250,0,250);
  digiShape = new TH1F("HB Digi Shape","HB Digi Shape",20,0,19);
  digiOccupancy = new TH2F("HB Digi Occupancy","HB Digi Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  return;
}

void HcalTemplateAnalysis::processEvent(const HBHEDigiCollection& hbhe,
					const HODigiCollection& ho,
					const HFDigiCollection& hf,
					const HBHERecHitCollection& hbHits, 
					const HORecHitCollection& hoHits,
					const HFRecHitCollection& hfHits,
					const LTCDigiCollection& ltc,
					const HcalDbService& cond){


  try{
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);	      
      for (int i=0; i<digi.size(); i++) {
	digiShape->Fill(i,digi.sample(i).adc());	
      }
      digiOccupancy->Fill(digi.id().ieta(),digi.id().iphi());
    }
  } catch (...) {
    printf("HcalTemplateAnalysis::processEvent  No HBHE Digis.\n");
  }




  HBHERecHitCollection::const_iterator _ib;
  //  HORecHitCollection::const_iterator _io;
  HFRecHitCollection::const_iterator _if;
  if(hbHits.size()>0){
    for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits
      if(_ib->energy()>0.0){
	if((HcalSubdetector)(_ib->id().subdet())==HcalBarrel){
	  rechitEnergy_HB->Fill(_ib->energy());
	  rechitTime_HB->Fill(_ib->time());
	  
	}
      }
    }
  }
  if(hfHits.size()>0){
    for (_if=hfHits.begin(); _if!=hfHits.end(); _if++) { // loop over all hits
      if(_if->energy()>0.0){
	rechitEnergy_HF->Fill(_if->energy());
	rechitTime_HF->Fill(_if->time());
      }
    }
  }

  return;
}

void HcalTemplateAnalysis::done(){

  TFile *writefile = new TFile(outputFile_.c_str(),"RECREATE");
  writefile->cd();
  
  rechitEnergy_HB->Write();
  rechitTime_HB->Write();
  rechitEnergy_HF->Write();
  rechitTime_HF->Write();
  digiShape->Write();
  digiOccupancy->Write();  
  
  writefile->Write();
  writefile->Close();

}
