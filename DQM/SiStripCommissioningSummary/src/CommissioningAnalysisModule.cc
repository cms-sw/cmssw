#include "DQM/SiStripCommissioningSummary/interface/CommissioningAnalysisModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//common
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
// edm
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
//data formats
#include "DataFormats/Common/interface/DetSetVector.h"
///analysis
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FedTimingAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/OptoScanAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/VpspScanAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvLatencyAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"

#include <iostream>
#include <iomanip>
#include <sstream>

CommissioningAnalysisModule::CommissioningAnalysisModule( const edm::ParameterSet& pset) :

  //initialise private data members
  c_summary_(0),
  c_summary2_(0),
  dirLevel_(pset.getUntrackedParameter<string>("summaryLevel","ControlView/")),
  controlView_(pset.getUntrackedParameter<bool>("controlView",true)),
  task_(SiStripHistoNamingScheme::task(pset.getUntrackedParameter<string>("commissioningTask","Pedestals"))),
  filename_(pset.getUntrackedParameter<string>("outputFilename","SUMMARY")),
  targetGain_(pset.getUntrackedParameter<double>("targetGain",0.8)),
  run_(0)
  
{

  //Check Commissioning Task
  if (task_ == sistrip::UNKNOWN_TASK) edm::LogWarning("Commissioning|AnalysisModule") << "Unknown commissioning task. Value used: " << pset.getUntrackedParameter<string>("CommissioningTask","Pedestals") << "; values accepted: Pedestals, ApvTiming, FedTiming, OptoScan, VpspScan, ApvLatency.";

  //construct summary objects as necessary
  if (task_ == sistrip::VPSP_SCAN) {
    c_summary_ = new CommissioningSummary(SiStripHistoNamingScheme::task(task_), sistrip::APV);}

  else if (task_ == sistrip::OPTO_SCAN) {
  c_summary_ = new CommissioningSummary((string)("Bias"), sistrip::LLD_CHAN);
  c_summary2_ = new CommissioningSummary((string)("Gain"), sistrip::LLD_CHAN);}

  else if (task_ == sistrip::PEDESTALS) {
  c_summary_ = new CommissioningSummary((string)("Pedestals"), sistrip::LLD_CHAN);
  c_summary2_ = new CommissioningSummary((string)("Noise"), sistrip::LLD_CHAN);
}

  else {c_summary_ = new CommissioningSummary(SiStripHistoNamingScheme::task(task_), sistrip::LLD_CHAN);}
}

//-----------------------------------------------------------------------------

CommissioningAnalysisModule::~CommissioningAnalysisModule() {

  //clean-up
  if (c_summary_) delete c_summary_;
  if (c_summary2_) delete c_summary2_;
}

//-----------------------------------------------------------------------------

void CommissioningAnalysisModule::beginJob() {;}

//-----------------------------------------------------------------------------

void CommissioningAnalysisModule::endJob() {

  //construct and name output file...
  string name = filename_.substr( 0, filename_.find(".root",0));
  stringstream ss; ss << name << "_" << SiStripHistoNamingScheme::task(task_) << "_" << setfill('0') << setw(7) << run_ << ".root";
  TFile* output = new TFile(ss.str().c_str(), "RECREATE");

  //write summary histogram(s) to file
  if (c_summary_) {
    TH1F* summ = (controlView_) ? c_summary_->controlSummary(dirLevel_) : c_summary_->summary(dirLevel_);
    summ->Write();
  }
  
  if (c_summary2_) {
    TH1F* summ2 = (controlView_) ? c_summary2_->controlSummary(dirLevel_) : c_summary2_->summary(dirLevel_);
    summ2->Write();
  }
  
  output->Close();

  //clean-up
  if (output) delete output;
}

//-----------------------------------------------------------------------------

void CommissioningAnalysisModule::analyze(const edm::Event& iEvent, const edm::EventSetup& setup) {
  
  // Extract run number
  if ( iEvent.id().run() != run_ ) { run_ = iEvent.id().run(); }
  
  //Get histograms from event
  edm::Handle< edm::DetSetVector< Profile > > profs;
  iEvent.getByType( profs );
  
  //storage tool for multi-histogram based analysis
  map< unsigned int, vector< vector< TProfile* > > > histo_organizer;

  //loop over histograms
  for (edm::DetSetVector<Profile>::const_iterator idetset = profs->begin(); idetset != profs->end(); idetset++) {
    for (edm::DetSet<Profile>::const_iterator prof = idetset->data.begin(); prof != idetset->data.end(); prof++) {
 
      //extract histogram details from encoded histogram name.
      std::string name(prof->get().GetName());
      SiStripHistoNamingScheme::HistoTitle h_title = SiStripHistoNamingScheme::histoTitle(name);
 
      //find control path from DetSetVector key
      SiStripControlKey::ControlPath path = SiStripControlKey::path(idetset->id);
      
      //get module information for the summary
      CommissioningSummary::ReadoutId readout(idetset->id, h_title.channel_);
      
      //commissioning analysis
      
      if (task_ == sistrip::APV_TIMING) {
	
	ApvTimingAnalysis anal;
	
	vector<const TProfile*> c_histos;
	c_histos.push_back(&prof->get());
	vector<unsigned short> c_monitorables;
	anal.analysis(c_histos, c_monitorables);
	unsigned int val = c_monitorables[0] * 24 + c_monitorables[1];
	c_summary_->update(readout, val); 
      }
      
      else if (task_ == sistrip::PEDESTALS) {

	//fill map with module histograms using key
	if (histo_organizer.find(h_title.channel_) == histo_organizer.end()) {
	  histo_organizer[h_title.channel_] = vector< vector<TProfile*> >(1, vector<TProfile*>(2,(TProfile*)(0)));}

	if (h_title.extraInfo_.find(sistrip::pedsAndRawNoise_) != string::npos) {histo_organizer[h_title.channel_][0][0] = const_cast<TProfile*>(&prof->get());}
	else if (h_title.extraInfo_.find(sistrip::residualsAndNoise_)  != string::npos) {histo_organizer[h_title.channel_][0][1] = const_cast<TProfile*>(&prof->get());}

	//if last histo in DetSet (i.e. for module) perform analysis and add to summary....
	if (prof == (idetset->data.end() - 1)) {

	  //define analysis object
	PedestalsAnalysis anal;

	//loop over lld channels
	for (map< unsigned int, vector< vector< TProfile* > > >::iterator it = histo_organizer.begin(); it != histo_organizer.end(); it++) {
	vector<const TProfile*> c_histos;
	c_histos.push_back(it->second[0][0]); c_histos.push_back(it->second[0][1]);
	vector< vector<float> > c_monitorables;
	anal.analysis(c_histos, c_monitorables);
       
	//ped == average pedestals, noise == average noise
	float ped = 0, noise = 0;
	
	if (c_monitorables[0].size() == c_monitorables[1].size() != 0) {
	  for (unsigned short istrip = 0; istrip < c_monitorables[0].size(); istrip++) {
	    ped += c_monitorables[0][istrip];
	    noise += c_monitorables[1][istrip];
	  }
	  ped = ped/c_monitorables[0].size();
	  noise = noise/c_monitorables[0].size();
	}
	
	//update summary
	CommissioningSummary::ReadoutId readout(idetset->id, it->first);
	c_summary_->update(readout, ped); 
	c_summary2_->update(readout, noise);
	}
	histo_organizer.clear();//refresh the container
	}
      }
      
      else if (task_ == sistrip::VPSP_SCAN) {
	
	VpspScanAnalysis anal;
	
	vector<const TProfile*> c_histos;
	c_histos.push_back(&prof->get());
	vector<unsigned short> c_monitorables;
	anal.analysis(c_histos, c_monitorables);
	unsigned int val = c_monitorables[0];
	c_summary_->update(readout, val); 
	
      }
      
      else if (task_ == sistrip::FED_TIMING) {
	
	FedTimingAnalysis anal;
	
	vector<const TProfile*> c_histos;
	c_histos.push_back(&prof->get());
	vector<unsigned short> c_monitorables;
	anal.analysis(c_histos, c_monitorables);
	unsigned int val = c_monitorables[0] * 25 + c_monitorables[1];
	c_summary_->update(readout, val); 
	
      }
      
      else if (task_ == sistrip::OPTO_SCAN) {

	//find gain value + digital level.
	string::size_type index = h_title.extraInfo_.find(sistrip::gain_);
	unsigned short gain = atoi(h_title.extraInfo_.substr((index + 4),1).c_str());

	index = h_title.extraInfo_.find(sistrip::digital_);
	unsigned short digital = atoi(h_title.extraInfo_.substr((index + 7),1).c_str());

	//fill map with module histograms using key
	if (histo_organizer.find(h_title.channel_) == histo_organizer.end()) {
	  histo_organizer[h_title.channel_] = vector< vector<TProfile*> >(4, vector<TProfile*>(2,(TProfile*)(0)));}
	
	if (digital == 0) {
	  histo_organizer[h_title.channel_][gain][0] = const_cast<TProfile*>(&prof->get());}
	
	if (digital == 1) {
	  histo_organizer[h_title.channel_][gain][1] = const_cast<TProfile*>(&prof->get());}
	
	//if last histo in DetSet (i.e. for module) perform analysis....
	if (prof == (idetset->data.end() - 1)) {
	  
	  OptoScanAnalysis anal;
	  vector<float> c_monitorables; c_monitorables.resize(2,0.);
	  
	  //loop over lld channels
	  for (map< unsigned int, vector< vector< TProfile* > > >::iterator it = histo_organizer.begin(); it != histo_organizer.end(); it++) {
	    
	    //loop over histos for of a single lld channel (loop over gain)
	    for (unsigned short igain = 0; igain < it->second.size(); igain++) {
	      
	      if (it->second[igain][0] && it->second[igain][1]) {
		vector<const TProfile*> c_histos; 
		c_histos.push_back(it->second[igain][0]);
		c_histos.push_back(it->second[igain][1]);
		vector<float> temp_monitorables;
		anal.analysis(c_histos, temp_monitorables);
		
		//store monitorables with gain nearest target.
		if ((fabs(temp_monitorables[0] - targetGain_) < fabs(c_monitorables[0] - targetGain_)) || ((it == histo_organizer.begin()) && igain == 0)) {c_monitorables = temp_monitorables;}
	      }
	    }
	    
	    CommissioningSummary::ReadoutId readout(idetset->id, it->first);
	    c_summary_->update(readout, c_monitorables[1]);
	    c_summary2_->update(readout, c_monitorables[0]); 
	  }
	  histo_organizer.clear();
	}
      }
      
      else if (task_ == sistrip::APV_LATENCY) {
	
	ApvLatencyAnalysis anal;
	
	vector<const TProfile*> c_histos;
	c_histos.push_back(&prof->get());
	vector<unsigned short> c_monitorables;
	anal.analysis(c_histos, c_monitorables);
	unsigned int val = c_monitorables[0];
	c_summary_->update(readout, val); 
      }
      
      else {edm::LogWarning("Commissioning|AnalysisModule") << "[CommissioningAnalysisModule::analyze]: Task \"" << task_ << "\" not recognized."; return;}
      
    }
  }
}



