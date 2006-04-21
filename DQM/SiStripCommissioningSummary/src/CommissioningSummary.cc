#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummary.h"

#include <cmath>

CommissioningSummary::CommissioningSummary(string summary_title, SiStripHistoNamingScheme::Granularity gran) :


  //initialise private data members
  title_(summary_title),
  granularity_(gran),

  map_(),
  summary_(0),
  controlSummary_(0),
  max_val_(0),
  max_val_err_(0),
  min_val_(0),
  min_val_err_(0)

{

  //construct histograms
  summary_ = new TH1F();
  controlSummary_ = new TH1F();
}

//------------------------------------------------------------------------------

CommissioningSummary::~CommissioningSummary() 
{

  //clean up
  if(summary_) delete summary_;
  if(controlSummary_) delete controlSummary_;
}

//------------------------------------------------------------------------------

void CommissioningSummary::update(ReadoutId& readout, float comm_val, float comm_val_error) {

  //find range for histograms
  if ((comm_val > max_val_)  || map_.empty()) max_val_ = comm_val;
  if ((comm_val_error > max_val_err_)  || map_.empty()) max_val_err_ = comm_val_error;
  if ((comm_val < min_val_) || map_.empty()) {min_val_ = comm_val;}
  if ((comm_val_error < min_val_err_) || map_.empty()) {min_val_err_ = comm_val_error;}

  //fill map
  map_[readout.dcu_id][readout.channel].first = comm_val;
  map_[readout.dcu_id][readout.channel].second = comm_val_error;
}

//------------------------------------------------------------------------------

TH1F* CommissioningSummary::controlSummary(const string& dir, const SiStripFecCabling* fec_cabling) {

 //interpret top level directory structure in terms of devices to be histogrammed
SiStripHistoNamingScheme::ControlPath path = SiStripHistoNamingScheme::controlPath(dir);

 //To get number of bins, loop through all devices in the cabling, only accepting devices that are within the requested path.
 unsigned int numOfBins = 0;

 const vector<SiStripFec>& fecs = fec_cabling->fecs();
 for (vector<SiStripFec>::const_iterator ifec = fecs.begin(); ifec != fecs.end(); ifec++) {
   if ((ifec->fecSlot() == path.fecSlot_) || (path.fecSlot_ == SiStripHistoNamingScheme::all())) {
    const vector<SiStripRing>& rings = (*ifec).rings();
    for ( vector<SiStripRing>::const_iterator iring = rings.begin(); iring != rings.end(); iring++ ) {
      if ((iring->fecRing() == path.fecRing_) || (path.fecRing_ == SiStripHistoNamingScheme::all())) {
      const vector<SiStripCcu>& ccus = (*iring).ccus();
      for ( vector<SiStripCcu>::const_iterator iccu = ccus.begin(); iccu != ccus.end(); iccu++ ) {
	if ((iccu->ccuAddr() == path.ccuAddr_) || (path.ccuAddr_ == SiStripHistoNamingScheme::all())) {
	const vector<SiStripModule>& modules = (*iccu).modules();
	for ( vector<SiStripModule>::const_iterator imodule = modules.begin(); imodule != modules.end(); imodule++ ) {
	  if ((imodule->ccuChan() == path.ccuChan_) || (path.ccuChan_ == SiStripHistoNamingScheme::all())) {

	    //increment bin number
	    numOfBins += (map_[imodule->dcuId()].size());	
	  }
	}
	}
      }
      }
    }
   }
 }

 //Format histogram
 controlSummary_->SetTitle(title_.c_str());
 controlSummary_->SetName(title_.c_str());
 controlSummary_->SetBins(numOfBins, 0.,(Double_t)numOfBins);
 
 //bin number and label containers
 unsigned int bin_num = 0;
 stringstream bin;
 
 for (vector<SiStripFec>::const_iterator ifec = fecs.begin(); ifec != fecs.end(); ifec++) {
   if ((ifec->fecSlot() == path.fecSlot_) || (path.fecSlot_ == SiStripHistoNamingScheme::all())) {
     const vector<SiStripRing>& rings = (*ifec).rings();
     for ( vector<SiStripRing>::const_iterator iring = rings.begin(); iring != rings.end(); iring++ ) {
       if ((iring->fecRing() == path.fecRing_) || (path.fecRing_ == SiStripHistoNamingScheme::all())) {
	 const vector<SiStripCcu>& ccus = (*iring).ccus();
	 for ( vector<SiStripCcu>::const_iterator iccu = ccus.begin(); iccu != ccus.end(); iccu++ ) {
	   if ((iccu->ccuAddr() == path.ccuAddr_) || (path.ccuAddr_ == SiStripHistoNamingScheme::all())) {
	     const vector<SiStripModule>& modules = (*iccu).modules();
	     for ( vector<SiStripModule>::const_iterator imodule = modules.begin(); imodule != modules.end(); imodule++ ) {
	       if ((imodule->ccuChan() == path.ccuChan_) || ( path.ccuChan_ == SiStripHistoNamingScheme::all())) {
		   for ( map< unsigned int, pair< float,float > >::const_iterator ichan = map_[imodule->dcuId()].begin(); ichan != map_[imodule->dcuId()].end(); ichan++) {

		     //update the bbin label with the control path
		   bin.str("");
		   bin << (*ifec).fecSlot() << "|" << (*iring).fecRing() << "|" << (*iccu).ccuAddr() << "|" << (*imodule).ccuChan() ;

		   if (granularity_ != SiStripHistoNamingScheme::MODULE) bin << "|" << ichan->first;
		   bin_num++;	
		   controlSummary_->GetXaxis()->SetBinLabel((Int_t)bin_num, bin.str().c_str());
		   //For each channel in the map add comm_val and error to histogram
		   controlSummary_->SetBinContent((Int_t)bin_num, ichan->second.first);
		   controlSummary_->SetBinError((Int_t)bin_num, ichan->second.second);
		 }
	       }
	     }
	   }
	   
	 }
       }
     }
   }
 }

 //return the histogram
 return controlSummary_;

}

//------------------------------------------------------------------------------

TH1F* CommissioningSummary::summary(const string& dir,  const SiStripFecCabling* fec_cabling, const string& option) {
 
  //check
  if ((option != "errors") && (option != "values")) {cout << "[CommissioningSummary::summary]: Unknown option. Option entered: " << option << "Expected either \"errors\" or \"values\". Returning null." << endl;
  return 0;}

 //interpret top level directory structure in terms of devices to be histogrammed
SiStripHistoNamingScheme::ControlPath path = SiStripHistoNamingScheme::controlPath(dir);

//Format histogram
 summary_->SetTitle(title_.c_str());
 summary_->SetName(title_.c_str());

 if (option == "errors") {
 summary_->SetTitle((title_+"Errors").c_str());
 summary_->SetName((title_+"Errors").c_str());
}

 //Calculate bin range based on the range of commissioning values (errors).
 int top_range = (option == "errors") ? (int)ceil(fabs(max_val_err_)) : (int)ceil(fabs(max_val_));

 if (((option == "errors") && (max_val_err_ < 0.)) || ((option == "values") && (max_val_ < 0.))) { top_range *= -1;}

 int bottom_range = (option == "errors") ? (int)ceil(fabs(min_val_err_)) : (int)ceil(fabs(min_val_));

 if (((option == "errors") && (min_val_err_ < 0.)) || ((option == "values") && (min_val_ < 0.))) { bottom_range *= -1;}

 unsigned int range = top_range - bottom_range;
 if (range) {
   summary_->SetBins((range + 2*(range/10)), (bottom_range - range/10), (top_range + range/10));}
 else { summary_->SetBins(2, (bottom_range - 1), (bottom_range + 1));}

 //Fill Histogram by looping over devices within the requested path.
 const vector<SiStripFec>& fecs = fec_cabling->fecs();
 for (vector<SiStripFec>::const_iterator ifec = fecs.begin(); ifec != fecs.end(); ifec++) {
   if ((ifec->fecSlot() == path.fecSlot_) || (path.fecSlot_ == SiStripHistoNamingScheme::all())) {
     const vector<SiStripRing>& rings = (*ifec).rings();
     for ( vector<SiStripRing>::const_iterator iring = rings.begin(); iring != rings.end(); iring++ ) {
       if ((iring->fecRing() == path.fecRing_) || (path.fecRing_ == SiStripHistoNamingScheme::all())) {
	 const vector<SiStripCcu>& ccus = (*iring).ccus();
	 for ( vector<SiStripCcu>::const_iterator iccu = ccus.begin(); iccu != ccus.end(); iccu++ ) {
	   if ((iccu->ccuAddr() == path.ccuAddr_) || (path.ccuAddr_ == SiStripHistoNamingScheme::all())) {
	     const vector<SiStripModule>& modules = (*iccu).modules();
	     for ( vector<SiStripModule>::const_iterator imodule = modules.begin(); imodule != modules.end(); imodule++ ) {
	       if ((imodule->ccuChan() == path.ccuChan_) || ( path.ccuChan_ == SiStripHistoNamingScheme::all())) {
		 for (unsigned short ichan = 0; ichan < (map_[imodule->dcuId()]).size(); ichan++) {
		   //For each channel in the map add comm_val to histogram

		   (option == "errors") ? summary_->Fill((Int_t)(map_[imodule->dcuId()][ichan].second)) : summary_->Fill((Int_t)(map_[imodule->dcuId()][ichan].first));

		 }
	       }
	     }
	   }
	   
	 }
       }
     }
   }
 }

 //return the histogram.
 return summary_;
}
