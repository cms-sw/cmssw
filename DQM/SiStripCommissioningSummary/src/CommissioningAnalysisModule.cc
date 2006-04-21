#include "DQM/SiStripCommissioningSummary/interface/CommissioningAnalysisModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//common
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"
// edm
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
//data formats
#include "DataFormats/Common/interface/DetSetVector.h"
//conditions
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
///analysis
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FedTimingAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/BiasGainAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/VPSPAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvLatencyAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"

#include <iostream>
#include <iomanip>
#include <sstream>

CommissioningAnalysisModule::CommissioningAnalysisModule( const edm::ParameterSet& pset) :

  //initialise private data members
  c_summary_(0),
  c_summary2_(0),
  fec_cabling_(0),
  task_(SiStripHistoNamingScheme::task(pset.getUntrackedParameter<string>("CommissioningTask","Pedestals"))),
  filename_(pset.getUntrackedParameter<string>("outputFilename","SUMMARY")),
  targetGain_(pset.getUntrackedParameter<double>("targetGain",0.8)),
  run_(0)
  
{
  //Check Commissioning Task
  if (task_ == SiStripHistoNamingScheme::UNKNOWN_TASK) edm::LogWarning("Commissioning|AnalysisModule") << "Unknown commissioning task. Value used: " << pset.getUntrackedParameter<string>("CommissioningTask","Pedestals") << "; values accepted: Pedestals, ApvTiming, FedTiming, OptoScan, VpspScan, ApvLatency.";

  //construct summary objects as necessary
  if (task_ == SiStripHistoNamingScheme::VPSP_SCAN) {
    c_summary_ = new CommissioningSummary(SiStripHistoNamingScheme::task(task_), SiStripHistoNamingScheme::APV);}

  else if (task_ == SiStripHistoNamingScheme::OPTO_SCAN) {
  c_summary_ = new CommissioningSummary((string)("BIAS"), SiStripHistoNamingScheme::LLD_CHAN);
  c_summary2_ = new CommissioningSummary((string)("GAIN"), SiStripHistoNamingScheme::LLD_CHAN);}

  else if (task_ == SiStripHistoNamingScheme::PEDESTALS) {
  c_summary_ = new CommissioningSummary((string)("PEDESTALS"), SiStripHistoNamingScheme::LLD_CHAN);
  c_summary2_ = new CommissioningSummary((string)("NOISE"), SiStripHistoNamingScheme::LLD_CHAN);}

  else {c_summary_ = new CommissioningSummary(SiStripHistoNamingScheme::task(task_), SiStripHistoNamingScheme::LLD_CHAN);}
}

//-----------------------------------------------------------------------------

CommissioningAnalysisModule::~CommissioningAnalysisModule() {

  //clean-up
if (c_summary_) delete c_summary_;
if (c_summary2_) delete c_summary2_;
}

//-----------------------------------------------------------------------------

void CommissioningAnalysisModule::beginJob(const edm::EventSetup& setup) {

//Control view cabling
  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling );
  fec_cabling_ = new SiStripFecCabling( *fed_cabling );
}

//-----------------------------------------------------------------------------

void CommissioningAnalysisModule::endJob() {

  //construct and name output file...
  string name = filename_.substr( 0, filename_.find(".root",0));
  stringstream ss; ss << name << "_" << SiStripHistoNamingScheme::task(task_) << "_" << setfill('0') << setw(7) << run_ << ".root";
  TFile* output = new TFile(ss.str().c_str(), "RECREATE");
  
  //write summary histogram(s) to file
  if (c_summary_) {
    TH1F* summ = c_summary_->controlSummary("ControlView/",fec_cabling_);
    summ->Write();
  }

  if (c_summary2_) {
    TH1F* summ2 = c_summary2_->controlSummary("ControlView/",fec_cabling_);
    summ2->Write();
  }
  
  output->Close();

  //clean-up
  if (output) delete output;
  if(fec_cabling_) delete fec_cabling_;
}

//-----------------------------------------------------------------------------

void CommissioningAnalysisModule::analyze(const edm::Event& iEvent, const edm::EventSetup& setup) {
  
  // Extract run number
  if ( iEvent.id().run() != run_ ) { run_ = iEvent.id().run(); }
  
  //Get histograms from event
  edm::Handle< edm::DetSetVector< Histo > > th1fs;
  iEvent.getByType( th1fs );
  
  //storage tool for opto bias-gain analysis
  map< unsigned int, vector< pair< TH1F*, TH1F* > > > bias_gain;

  //loop over histograms
  for (edm::DetSetVector<Histo>::const_iterator idetset = th1fs->begin(); idetset != th1fs->end(); idetset++) {
    
    for (edm::DetSet<Histo>::const_iterator th1f = idetset->data.begin(); th1f != idetset->data.end(); th1f++) {
  
      //extract histogram details from encoded histogram name.
      std::string name(th1f->get().GetName());
      SiStripHistoNamingScheme::HistoTitle h_title = SiStripHistoNamingScheme::histoTitle(name);

      //find control path from DetSetVector key
      SiStripGenerateKey::ControlPath path = SiStripGenerateKey::controlPath(idetset->id);
      
      //get module information for the summary
      const FedChannelConnection conn(path.fecCrate_, path.fecSlot_, path.fecRing_, path.ccuAddr_, path.ccuChan_);
      unsigned int dcu_id = fec_cabling_->module(conn).dcuId();
      CommissioningSummary::ReadoutId readout(dcu_id, h_title.channel_);
      
      //commissioning analysis
      
      if (task_ == SiStripHistoNamingScheme::APV_TIMING) {
	
	ApvTimingAnalysis anal;
	
	vector<const TH1F*> c_histos;
	c_histos.reserve(1);
	c_histos.push_back(&th1f->get());
	vector<unsigned short> c_monitorables;
	anal.analysis(c_histos, c_monitorables);
	unsigned int val = c_monitorables[0] * 24 + c_monitorables[1];
	c_summary_->update(readout, val); 
      }
      
      else if (task_ == SiStripHistoNamingScheme::PEDESTALS) {
	
	PedestalsAnalysis anal;
	
	vector<const TH1F*> c_histos;
	c_histos.reserve(1);
	c_histos.push_back(&th1f->get());
	vector< vector<float> > c_monitorables;
	anal.analysis(c_histos, c_monitorables);
       
	//ped == average pedestals, noise == average noise
	float ped = 0, noise = 0;
	
	if (!c_monitorables[0].empty()) {
	  for (unsigned short istrip = 0; istrip < c_monitorables[0].size(); istrip++) {
	    ped += c_monitorables[0][istrip];
	    noise += c_monitorables[1][istrip];
	  }
	  ped = ped/c_monitorables[0].size();
	  noise = noise/c_monitorables[0].size();
	}
	c_summary_->update(readout, ped); 
	c_summary2_->update(readout, noise);
      }
      
      else if (task_ == SiStripHistoNamingScheme::VPSP_SCAN) {
	
	VPSPAnalysis anal;
	
	vector<const TH1F*> c_histos;
	c_histos.reserve(1);
	c_histos.push_back(&th1f->get());
	vector<unsigned short> c_monitorables;
	anal.analysis(c_histos, c_monitorables);
	unsigned int val = c_monitorables[0];
	c_summary_->update(readout, val); 
	
      }
      
      else if (task_ == SiStripHistoNamingScheme::FED_TIMING) {
	
	FedTimingAnalysis anal;
	
	vector<const TH1F*> c_histos;
	c_histos.reserve(1);
	c_histos.push_back(&th1f->get());
	vector<unsigned short> c_monitorables;
	anal.analysis(c_histos, c_monitorables);
	unsigned int val = c_monitorables[0] * 25 + c_monitorables[1];
	c_summary_->update(readout, val); 
	
      }
      
      else if (task_ == SiStripHistoNamingScheme::OPTO_SCAN) {

	//find gain value + digital level.
	string::size_type index = h_title.extraInfo_.find(SiStripHistoNamingScheme::gain());
	unsigned short gain = atoi(h_title.extraInfo_.substr((index + 4),1).c_str());

	index = h_title.extraInfo_.find(SiStripHistoNamingScheme::digital());
	unsigned short digital = atoi(h_title.extraInfo_.substr((index + 7),1).c_str());

	//fill map with module histograms using key
	if (bias_gain.find(h_title.channel_) == bias_gain.end()) {
	  bias_gain[h_title.channel_] = vector< pair<TH1F*, TH1F*> >(4, pair<TH1F*,TH1F*>(0,0));}
	
	if (digital == 0) {
	  bias_gain[h_title.channel_][gain].first = const_cast<TH1F*>(&th1f->get());}
	
	if (digital == 1) {
	  bias_gain[h_title.channel_][gain].second = const_cast<TH1F*>(&th1f->get());}
	
	//if last histo in DetSet (i.e. for module) perform analysis....
	if (th1f == (idetset->data.end() - 1)) {
	  
	  BiasGainAnalysis anal;
	  vector<float> c_monitorables; c_monitorables.resize(2,0.);
	  
	  //loop over apvs
	  for (map< unsigned int, vector< pair< TH1F*, TH1F* > > >::iterator it = bias_gain.begin(); it != bias_gain.end(); it++) {
	    
	    //loop over histos for of a single apv (loop over gain)
	    for (unsigned short igain = 0; igain < it->second.size(); igain++) {
	      
	      if (it->second[igain].first && it->second[igain].second) {
		vector<const TH1F*> c_histos; 
		c_histos.reserve(2);
		c_histos.push_back(it->second[igain].first);
		c_histos.push_back(it->second[igain].second);
		vector<float> temp_monitorables;
		anal.analysis(c_histos, temp_monitorables);
		
		//store monitorables with gain nearest target.
		if ((fabs(temp_monitorables[0] - targetGain_) < fabs(c_monitorables[0] - targetGain_)) || ((it == bias_gain.begin()) && igain == 0)) {c_monitorables = temp_monitorables;}
	      }
	    }
	    
	    CommissioningSummary::ReadoutId readout(dcu_id, it->first);
	    c_summary_->update(readout, c_monitorables[1]);
	    c_summary2_->update(readout, c_monitorables[0]); 
	  }
	  bias_gain.clear();
	}
      }
      
      else if (task_ == SiStripHistoNamingScheme::APV_LATENCY) {
	
	ApvLatencyAnalysis anal;
	
	vector<const TH1F*> c_histos;
	c_histos.reserve(1);
	c_histos.push_back(&th1f->get());
	vector<unsigned short> c_monitorables;
	anal.analysis(c_histos, c_monitorables);
	unsigned int val = c_monitorables[0];
	c_summary_->update(readout, val); 
      }
      
      else {edm::LogWarning("Commissioning|AnalysisModule") << "[CommissioningAnalysisModule::analyze]: Task \"" << task_ << "\" not recognized."; return;}
      
    }
  }
}



