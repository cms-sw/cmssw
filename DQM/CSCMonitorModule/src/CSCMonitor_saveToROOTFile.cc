#include "DQM/CSCMonitorModule/interface/CSCMonitor.h"

void CSCMonitor::saveToROOTFile(std::string filename)
{
  LOG4CPLUS_INFO(logger_, "Saving MEs to ROOT File... " << filename)
    if (filename == "") {
      LOG4CPLUS_ERROR(logger_, "Empty ROOT file name ")
	return;
    }
  TFile f(filename.c_str(), "recreate");
  if (!f.IsZombie()) {
    fBusy = true;
    gStyle->SetPalette(1,0);
    f.cd();
    std::map<std::string, ME_List>::iterator itr;
    ME_List_const_iterator h_itr;

    TDirectory * hdir = f.mkdir("histos");
    hdir->cd();
    for (itr = MEs.begin(); itr != MEs.end(); ++itr) {
      TDirectory * rdir = hdir->mkdir((itr->first).c_str());
      rdir->cd();
      for (h_itr = itr->second.begin(); h_itr != itr->second.end(); ++h_itr) {
	h_itr->second->Write();
      }
      hdir->cd();
    } 

    f.Close();
    LOG4CPLUS_INFO(logger_, "Done.");

    fBusy=false;
  } else {
    LOG4CPLUS_ERROR(logger_, "Unable to open output ROOT file " << filename);
  }
}


