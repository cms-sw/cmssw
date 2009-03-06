#include <Riostream.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TKey.h>
#include <TH1.h>
#include <TXMLEngine.h>

#include <string>
#include <vector>
#include <map>

bool goToDir(TDirectory* top_dir, string dname);
void findHistoParameters(TDirectory* top_dir, vector<string>& names, string& tag, 
                               TXMLEngine* xml_engine,XMLNodePointer_t& mainnode);

void sistrip_extract_dataquality(string fname) {

  TFile* file = new TFile(fname.c_str());
  if (!file || !file->IsOpen()) return;

  TDirectory* topDir = dynamic_cast<TDirectory*>( file->Get("DQMData"));
  goToDir(topDir, "SiStrip");
  TDirectory* stripDir = gDirectory;

  TIter next(stripDir->GetListOfKeys());
  TKey *key;
  while  ( (key = dynamic_cast<TKey*>(next())) ) {
    string clName(key->GetClassName());
    if (clName == "TDirectoryFile") {
      TDirectory *curr_dir = dynamic_cast<TDirectory*>(key->ReadObj());
      string name = curr_dir->GetName();
      if (name == "Run summary"){
	curr_dir->cd();
	break;
      }
    }
  }
  TDirectory* rsDir = gDirectory;

  // First create engine
  TXMLEngine* xml = new TXMLEngine;
  // Create main node of document tree
   XMLNodePointer_t mainnode = xml->NewChild(0, 0, "main");

  vector<string> hnames;  
  string tag;

  tag = "Tracks";
  hnames.push_back("Tracks/NumberOfTracks_CKFTk");
  hnames.push_back("Tracks/NumberOfRecHitsPerTrack_CKFTk");
  findHistoParameters(rsDir, hnames, tag, xml, mainnode);

  hnames.clear();
  tag = "TEC";
  hnames.push_back("MechanicalView/TEC/Summary_TotalNumberOfClusters_OnTrack_in_TEC");
  hnames.push_back("MechanicalView/TEC/Summary_TotalNumberOfClusters_OffTrack_in_TEC");
  hnames.push_back("MechanicalView/TEC/Summary_ClusterStoNCorr_OnTrack_in_TEC");
  hnames.push_back("MechanicalView/TEC/Summary_ClusterCharge_OffTrack_in_TEC");
  findHistoParameters(rsDir, hnames, tag, xml, mainnode);

  hnames.clear();
  tag = "TIB";
  hnames.push_back("MechanicalView/TIB/Summary_TotalNumberOfClusters_OnTrack_in_TIB");
  hnames.push_back("MechanicalView/TIB/Summary_TotalNumberOfClusters_OffTrack_in_TIB");
  hnames.push_back("MechanicalView/TIB/Summary_ClusterCharge_OffTrack_in_TIB");
  hnames.push_back("MechanicalView/TIB/Summary_ClusterStoNCorr_OnTrack_in_TIB");
  findHistoParameters(rsDir, hnames, tag, xml, mainnode);

  hnames.clear();
  tag = "TOB";
  hnames.push_back("MechanicalView/TOB/Summary_TotalNumberOfClusters_OnTrack_in_TOB");
  hnames.push_back("MechanicalView/TOB/Summary_TotalNumberOfClusters_OffTrack_in_TOB");
  hnames.push_back("MechanicalView/TOB/Summary_ClusterStoNCorr_OnTrack_in_TOB");
  hnames.push_back("MechanicalView/TOB/Summary_ClusterCharge_OffTrack_in_TOB");
  findHistoParameters(rsDir, hnames, tag, xml, mainnode);

  hnames.clear();
  tag = "TID";
  hnames.push_back("MechanicalView/TID/Summary_TotalNumberOfClusters_OffTrack_in_TID");
  hnames.push_back("MechanicalView/TID/Summary_TotalNumberOfClusters_OnTrack_in_TID");
  hnames.push_back("MechanicalView/TID/Summary_ClusterStoNCorr_OnTrack_in_TID");
  hnames.push_back("MechanicalView/TID/Summary_ClusterCharge_OffTrack_in_TID");
  findHistoParameters(rsDir, hnames, tag, xml, mainnode);


  file->Close();

   // now create doccumnt and assign main node of document
   XMLDocPointer_t xmldoc = xml->NewDoc();
   xml->DocSetRootElement(xmldoc, mainnode);
   
   // Save document to file
   xml->SaveDoc(xmldoc, "sistrip_report.xml");
      
   // Release memory before exit
   xml->FreeDoc(xmldoc);
   delete xml;
}

void findHistoParameters(TDirectory* top_dir,vector<string>& hnames, string& tag, 
                                  TXMLEngine* xml_engine, XMLNodePointer_t& mainnode){
  char mval[10];
  char rval[10];
  string dir_name = top_dir->GetTitle();
  cout << " ======================= " << tag <<  " ======================= " << endl;

  for (vector<string>::iterator it = hnames.begin(); it != hnames.end(); it++) { 
    TH1* th1 = dynamic_cast<TH1*>(gDirectory->Get((*it).c_str()));
    if (th1) {
      sprintf(mval,"%.2f",th1->GetMean());
      sprintf(rval,"%.2f",th1->GetRMS());
      XMLNodePointer_t child = xml_engine->NewChild(mainnode, 0, tag.c_str());
      xml_engine->NewAttr(child, 0, "Name", th1->GetName()); 
      xml_engine->NewAttr(child, 0, "Mean", mval); 
      xml_engine->NewAttr(child, 0, "RMS", rval); 
      cout << th1->GetName() << " Mean  " << mval << " RMS " << rval << endl;
    }
  }
  cout << endl;
}  
bool goToDir(TDirectory* top_dir, string dname){
  string dir_name = top_dir->GetTitle();
  if (dir_name == dname) {
    return true;
  } else {
    TIter next(top_dir->GetListOfKeys());
    TKey *key;
    while  ( (key = dynamic_cast<TKey*>(next())) ) {
      string clName(key->GetClassName());
      if (clName == "TDirectoryFile") {
        TDirectory *curr_dir = dynamic_cast<TDirectory*>(key->ReadObj());
        string name = curr_dir->GetName();
        if (name == "Reference") continue;
	curr_dir->cd();
	if (goToDir(curr_dir, dname)) return true;
	curr_dir->cd("..");
      }
    }
  }
  return false;
}
