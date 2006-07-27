#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorControlView.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"

using namespace std;

//------------------------------------------------------------------------------
SummaryGeneratorControlView::SummaryGeneratorControlView() {;}

//------------------------------------------------------------------------------
SummaryGeneratorControlView::~SummaryGeneratorControlView() {;}

//------------------------------------------------------------------------------

void SummaryGeneratorControlView::summary(TH1F* controlSumm, TH1F* summ, const string& dir, const string& option) {

 //interpret top level directory structure in terms of devices to be histogrammed
  stringstream directory;
  if (dir.substr(0,11) != sistrip::controlView_) {
  directory << sistrip::controlView_
	    << sistrip::dir_
	    << dir;}
  else {directory << dir;}
  SiStripHistoNamingScheme::ControlPath path = SiStripHistoNamingScheme::controlPath(directory.str());

 //To get number of bins, loop through all devices in the cabling, only accepting devices that are within the requested path.
 unsigned int numOfBins = 0;

 for ( map< unsigned int, pair< float,float > >::const_iterator idevice = map_.begin(); idevice != map_.end(); idevice++) {

   //unpack fec-key
   SiStripControlKey::ControlPath device_path = SiStripControlKey::path(idevice->first);
 
   if (((device_path.fecCrate_ == path.fecCrate_) || (path.fecCrate_ == sistrip::all_)) &&
       ((device_path.fecSlot_ == path.fecSlot_) || (path.fecSlot_ == sistrip::all_)) &&
       ((device_path.fecRing_ == path.fecRing_) || (path.fecRing_ == sistrip::all_)) && 
       ((device_path.ccuAddr_ == path.ccuAddr_) || (path.ccuAddr_ == sistrip::all_)) &&
       ((device_path.ccuChan_ == path.ccuChan_) || (path.ccuChan_ == sistrip::all_))) {
     //increment bin number
     numOfBins ++;
   }
 }

 //Format histogram
 controlSumm->SetBins(numOfBins, 0.,(Double_t)numOfBins);
 
 //bin number and label containers
 unsigned int bin_num = 0;
 stringstream bin;
 
 //Fill Histogram by looping over devices within the requested path.
 for (map< unsigned int, pair< float,float > >::const_iterator idevice = map_.begin(); idevice != map_.end(); idevice++) {
   
   //unpack fec-key
   SiStripControlKey::ControlPath device_path = SiStripControlKey::path(idevice->first);
   if (((device_path.fecCrate_ == path.fecCrate_) || (path.fecCrate_ == sistrip::all_)) &&
       ((device_path.fecSlot_ == path.fecSlot_) || (path.fecSlot_ == sistrip::all_)) &&
       ((device_path.fecRing_ == path.fecRing_) || (path.fecRing_ == sistrip::all_)) && 
       ((device_path.ccuAddr_ == path.ccuAddr_) || (path.ccuAddr_ == sistrip::all_)) &&
       ((device_path.ccuChan_ == path.ccuChan_) || (path.ccuChan_ == sistrip::all_))) {
     
     //update the bin label with the control path
     bin.str("");
     bin << device_path.fecCrate_ << "|" << device_path.fecSlot_ << "|" << device_path.fecRing_ << "|" <<  device_path.ccuAddr_ << "|" << device_path.ccuChan_ << "|" << device_path.channel_;
     
     bin_num++;	
     controlSumm->GetXaxis()->SetBinLabel((Int_t)bin_num, bin.str().c_str());
     //For each channel in the map add comm_val and error to histogram
     controlSumm->SetBinContent((Int_t)bin_num, idevice->second.first);
     controlSumm->SetBinError((Int_t)bin_num, idevice->second.second);
   }
 }
 histogram(summ,directory.str(),option);
}

//------------------------------------------------------------------------------

void SummaryGeneratorControlView::histogram(TH1F* summ, const string& dir, const string& option) {
 
  //check
  if ((option != "errors") && (option != "values")) {cout << "[SummaryGeneratorControlView::summary]: Unknown option. Option entered: " << option << "Expected either \"errors\" or \"values\". Returning null." << endl;
  return;}

 //interpret top level directory structure in terms of devices to be histogrammed
  stringstream directory;
  if (dir.substr(0,11) != sistrip::controlView_) {
  directory << sistrip::controlView_
	    << sistrip::dir_
	    << dir;}
  else {directory << dir;}
SiStripHistoNamingScheme::ControlPath path = SiStripHistoNamingScheme::controlPath(directory.str());

 //Calculate bin range based on the range of commissioning values (errors).
 int top_range = (option == "errors") ? (int)ceil(fabs(max_val_err_)) : (int)ceil(fabs(max_val_));

 if (((option == "errors") && (max_val_err_ < 0.)) || ((option == "values") && (max_val_ < 0.))) { top_range *= -1;}

 int bottom_range = (option == "errors") ? (int)ceil(fabs(min_val_err_)) : (int)ceil(fabs(min_val_));

 if (((option == "errors") && (min_val_err_ < 0.)) || ((option == "values") && (min_val_ < 0.))) { bottom_range *= -1;}

 unsigned int range = top_range - bottom_range;
 if (range) {
   summ->SetBins((range + 2*(range/10)), (bottom_range - range/10), (top_range + range/10));}
 else { summ->SetBins(2, (bottom_range - 1), (bottom_range + 1));}

 //Fill Histogram by looping over devices within the requested path.
for ( map< unsigned int, pair< float,float > >::const_iterator idevice = map_.begin(); idevice != map_.end(); idevice++) {

   //unpack fec-key
   SiStripControlKey::ControlPath device_path = SiStripControlKey::path(idevice->first);
   if (((device_path.fecCrate_ == path.fecCrate_) || (path.fecCrate_ == sistrip::all_)) &&
       ((device_path.fecSlot_ == path.fecSlot_) || (path.fecSlot_ == sistrip::all_)) &&
       ((device_path.fecRing_ == path.fecRing_) || (path.fecRing_ == sistrip::all_)) && 
       ((device_path.ccuAddr_ == path.ccuAddr_) || (path.ccuAddr_ == sistrip::all_)) &&
       ((device_path.ccuChan_ == path.ccuChan_) || (path.ccuChan_ == sistrip::all_))) {

  //For each channel in the map add comm_val to histogram
 (option == "errors") ? summ->Fill((Int_t)(idevice->second.second)) : summ->Fill((Int_t)(idevice->second.first));
   }
}
}
