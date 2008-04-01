#include <TFile.h>
#include <TKey.h>
#include <TClass.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TGraph.h>
#include <TObjString.h>
#include <TCanvas.h>
#include <string>
#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <cmath>
#include <cstring>

typedef std::pair<int, int> Parameters;
typedef std::map<Parameters,TFile*> FileList;
typedef std::map<Parameters,std::vector<TH1*> > SummaryV;

#define DATAPATH "/DQMData/Collate/SiStrip/"
#define HISTOPATH "/DQMData/Collate/SiStrip/ControlView/"

#define DEBUG_ON

class CalibrationScanAnalysis
{

  public:
    CalibrationScanAnalysis(bool tuneISHA = true, bool tuneVFS = true);
    virtual ~CalibrationScanAnalysis();
    void tuneISHA(bool tune) { tuneISHA_ = tune; }
    void tuneVFS(bool tune)  { tuneVFS_  = tune; }
    void addFile(const std::string&);
    void analyze();
    void sanitizeResult(unsigned int cut = 2);
    void print(Option_t* option = "") const;
    void draw(Option_t* option = "") const;
    void save();

  protected:
    void addFile(TFile* );
    void getSummaries(FileList::const_iterator);
    void sortByGeometry();
    void loadPresentValues();
    float getX(const TGraph*, const float&) const;
    bool checkInput() const;
    
  private:
    bool tuneISHA_, tuneVFS_;
    FileList files_;
    SummaryV summaries_;
    std::map<std::string, Parameters> result_;
    std::map<std::string, int> geometries_;
    std::map<std::string, Parameters> presentValues_;
};

CalibrationScanAnalysis::CalibrationScanAnalysis(bool tuneISHA, bool tuneVFS):
    tuneISHA_(tuneISHA),tuneVFS_(tuneVFS) {
}

CalibrationScanAnalysis::~CalibrationScanAnalysis() {
  // close and delete all files
  for(FileList::iterator file = files_.begin();file!=files_.end();++file) {
    // this will automatically delete histograms in summaries_
    file->second->Close();
    delete file->second;
  }
}

void CalibrationScanAnalysis::addFile(const std::string& filename) {
  TFile* newFile = new TFile(filename.c_str(),"UPDATE");
  addFile(newFile);
}

void CalibrationScanAnalysis::addFile(TFile* newFile) {
  int isha,vfs;
  TList* keyList = newFile->GetDirectory(DATAPATH)->GetListOfKeys();
  TIter next(keyList);
  TNamed* ishaObj = NULL;
  TNamed* vfsObj  = NULL;
  TNamed* obj = NULL;
  while ((obj = (TNamed*)(next()))) {
    if(strncmp(obj->GetName(),"<isha>",6)==0) ishaObj = (TNamed*)obj;
    if(strncmp(obj->GetName(),"<vfs>",5)==0)  vfsObj  = (TNamed*)obj;
  }
  if(!ishaObj || !vfsObj) {
     std::cerr << "Error: Unexpected file structure. ISHA/VFS values not found." << std::endl;
     newFile->Close();
     delete newFile;
     return;
  }
  isha = atoi(ishaObj->GetName()+8);
  vfs  = atoi(vfsObj->GetName()+7 );
  std::cout << "Loaded File for ISHA/VFS = " << isha << "/" << vfs << std::endl;
  files_[std::make_pair(isha,vfs)] = newFile;
}

void CalibrationScanAnalysis::getSummaries(FileList::const_iterator file) {
  std::vector<TH1*> result;  
  TFile* input = file->second;
  TDirectory* directory = input->GetDirectory(HISTOPATH);
  TList* histograms = directory->GetListOfKeys();
  TIter next(histograms);
  TKey* key = NULL;
  while ((key = (TKey*)next())) {
    if(TClass(key->GetClassName()).InheritsFrom("TH1")) {
      TH1* h = (TH1*)key->ReadObj();
      result.push_back(h);
    }
  }
  summaries_[file->first] = result;
}

float CalibrationScanAnalysis::getX(const TGraph* graph, const float& y) const {
   Double_t* arrayX = graph->GetX();
   Double_t* arrayY = graph->GetY();
   //first, look for an intersection
   for(int i=0;i<graph->GetN()-1;++i) {
     if((arrayY[i]-y)*(arrayY[i+1]-y)<0)
       return (arrayX[i]+((arrayX[i+1]-arrayX[i])/(arrayY[i+1]-arrayY[i])*(y-arrayY[i])));
   }
   // if none, look for a plateau
   float finalDelta = fabs(arrayY[graph->GetN()-1]-y);
   // allow for a 50% increase of the difference
   float delta = finalDelta*0.5;
   int lastpoint = graph->GetN()-1;
   for(int i=lastpoint-1;i>=0;--i) {
     if(fabs(arrayY[lastpoint]-arrayY[i])>delta)
       return arrayX[i+1];
   }
   // in last ressort, return the central value.
   return arrayX[lastpoint]-arrayX[0];
}

void CalibrationScanAnalysis::analyze() {

#ifdef DEBUG_ON
  TFile* debugFile = new TFile("debug.root","RECREATE");
#endif
  
  // load data from files
  for(FileList::const_iterator it=files_.begin();it!=files_.end();++it) {
    getSummaries(it);
  }
  sortByGeometry();
  loadPresentValues();

  // sanity check
  if(!checkInput()) return;

  // check if both ISHA and VFS have to be tuned
  int minISHA = 1000;
  int maxISHA = 0;
  int minVFS  = 1000;
  int maxVFS  = 0;
  for(FileList::const_iterator file=files_.begin();file!=files_.end();++file){
    int isha = file->first.first;
    int vfs  = file->first.second;
    minISHA = minISHA<isha ? minISHA : isha;
    maxISHA = maxISHA>isha ? maxISHA : isha;
    minVFS  = minVFS <vfs  ? minVFS  : vfs ;
    maxVFS  = maxVFS >vfs  ? maxVFS  : vfs ;
  }
  tuneISHA_ &= (minISHA!=maxISHA);
  tuneVFS_  &= (minVFS !=maxVFS );
  if(!tuneISHA_) std::cout << "ISHA tune disabled" << std::endl;
  if(!tuneVFS_ ) std::cout << "VFS  tune disabled" << std::endl;

  // number of APVs
  unsigned int nAPVs = (*(summaries_.begin()->second.begin()))->GetNbinsX();

  // loop over apvs (bins)
  for(unsigned int apv=1;apv<=nAPVs;++apv) {
     TGraph* g1 = new TGraph();
     TGraph* g2 = new TGraph();
     int ii=0;
     for(SummaryV::const_iterator summary = summaries_.begin(); summary!=summaries_.end(); ++summary,++ii) {
       // two cases are possible:
       // ISHA tune: look at the rise time
       // VFS  tune: look at the tail
       
       // determine which histogram are the rise time and the tail
       const std::vector<TH1*>& observables = summary->second;
       int tail_index = 0;
       int rise_index = 0;
       for( std::vector<TH1*>::const_iterator histo = observables.begin();histo<observables.end();++histo) {
          std::string name = (*histo)->GetName();
          if(name.find("CalibrationTail")!=std::string::npos) tail_index = histo-observables.begin();
          if(name.find("CalibrationRiseTime")!=std::string::npos) rise_index = histo-observables.begin();
       }
        
       // fill the graphs
       g1->SetPoint(ii,summary->first.first,observables[rise_index]->GetBinContent(apv));
       g2->SetPoint(ii,summary->first.second, observables[tail_index]->GetBinContent(apv));
     }
#ifdef DEBUG_ON
     g1->Write(Form("%s%s",summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv),"CalibrationRiseTime"));
     g2->Write(Form("%s%s",summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv),"CalibrationTail"));
#endif
     // analyse the graphs
     float best_isha = tuneISHA_ ? getX(g1,66. ) : 
                                   presentValues_[summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv)].first;
     float best_vfs  = tuneVFS_  ? getX(g2,0.36) : 
                                   presentValues_[summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv)].second;
     // save the result
     result_[summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv)] = 
                         std::make_pair((int)round(best_isha),(int)round(best_vfs));
     // cleaning
     delete g1;
     delete g2;
  }

#ifdef DEBUG_ON
  debugFile->Write();
  debugFile->Close();
  delete debugFile;
#endif

}

bool CalibrationScanAnalysis::checkInput() const {

  // check that we have data
  if(!summaries_.size()) {
    std::cerr << "Error: No summary histogram found." << std::endl
              << " Did you load any file ? " << std::endl;
    return 0;
  }

  if(summaries_.size()<2) {
    std::cerr << "Error: Only one  summary histogram found." << std::endl
              << " Analysis does not make sense with only one measurement" << std::endl;
    return 0;
  }

  // check that we have the same entries in each record,
  // check that the binning is the same in all histograms
  int nbinsAll = -1;
  std::vector<std::string> namesAll;
  for(SummaryV::const_iterator summary = summaries_.begin(); summary!=summaries_.end(); ++summary) {
    const std::vector<TH1*>& observables = summary->second;
    for( std::vector<TH1*>::const_iterator histo = observables.begin();histo<observables.end();++histo) {
       std::string name = (*histo)->GetName();
       if(summary == summaries_.begin()) {
          namesAll.push_back(name);
       } else {
          if(find(namesAll.begin(),namesAll.end(),name)==namesAll.end()) {
            std::cerr << "Error: Found an histogram that is not common to all inputs: "
                      << name << std::endl;
            return 0;
          }
       }
       int nbins = (*histo)->GetNbinsX();
       if(nbinsAll<0) nbinsAll = nbins;
       if(nbins != nbinsAll) {
         std::cerr << "Error: The number of bins is not the same in all inputs." << std::endl;
         return 0;
       }
    }
  }

  // check that we have at least 2 histograms with measurements
  if(namesAll.size()<2) {
    std::cerr << "Error: The number of available measurements is smaller than 2." << std::endl;
    return 0;
  }

  // check that the bin labels are all the same
  std::vector<std::string> labelsAll;
  for(SummaryV::const_iterator summary = summaries_.begin(); summary!=summaries_.end(); ++summary) {
    const std::vector<TH1*>& observables = summary->second;
    for( std::vector<TH1*>::const_iterator histo = observables.begin();histo<observables.end();++histo) {
       for(int i = 1;i <= (*histo)->GetNbinsX(); ++i) {
         std::string label = (*histo)->GetXaxis()->GetBinLabel(i);
         if(summary == summaries_.begin() && histo == observables.begin()) {
           labelsAll.push_back(label);
         } else {
           if(labelsAll[i-1] != label) {
             std::cerr << "Error: Incoherency in bin labels. Bin " << i 
                       << " of " << (*histo)->GetName() << " is " << label
                       << " and not " << labelsAll[i] << "." << std::endl;
             return 0;
           }
         }
       }
    }
  }

  // check that all APVs have an associated geometry
  for(std::vector<std::string>::const_iterator apvLabel = labelsAll.begin();
      apvLabel != labelsAll.end(); ++apvLabel) {
    if(geometries_.find(*apvLabel)==geometries_.end()) {
      std::cerr << "Error: Geometry unknown for APV " << *apvLabel << std::endl;
      return 0;
    }
  }

  return 1;

}

void CalibrationScanAnalysis::sortByGeometry() {

  //categorize APVs per module geometry
  std::cout << "Reading cabling from debug.log" << std::endl;
  ifstream debuglog("debug.log");
  char buffer[1024];
  while(debuglog.getline(buffer,1024)) {
    if(strncmp(buffer," FED:cr/sl/id/fe/ch/chan",23)==0) {

      // Decode input
      int fecCrate,fecSlot,fecRing,ccuAddr,ccuChan,channel1,channel2,detid,tmp;
      sscanf(strstr(buffer,"FEC:cr/sl/ring/ccu/mod"), "FEC:cr/sl/ring/ccu/mod=%d/%d/%d,%d/%d", &fecCrate,&fecSlot,&fecRing,&ccuAddr, &ccuChan);
      sscanf(strstr(buffer,"apvs"), "apvs=%d/%d", &channel1,&channel2);
      sscanf(strstr(buffer,"dcu/detid"), "dcu/detid=%x/%x", &tmp,&detid);

      // Construct bin label
      std::stringstream bin1;
      bin1        << std::setw(1) << std::setfill('0') << fecCrate;
      bin1 << "." << std::setw(2) << std::setfill('0') << fecSlot;
      bin1 << "." << std::setw(1) << std::setfill('0') << fecRing;
      bin1 << "." << std::setw(3) << std::setfill('0') << ccuAddr;
      bin1 << "." << std::setw(2) << std::setfill('0') << ccuChan;
      bin1  << "." << channel1;
      std::stringstream bin2;
      bin2        << std::setw(1) << std::setfill('0') << fecCrate;
      bin2 << "." << std::setw(2) << std::setfill('0') << fecSlot;
      bin2 << "." << std::setw(1) << std::setfill('0') << fecRing;
      bin2 << "." << std::setw(3) << std::setfill('0') << ccuAddr;
      bin2 << "." << std::setw(2) << std::setfill('0') << ccuChan;
      bin2 << "." << channel2;

      // Decode the detid -> sensor geometry
      int subdet = (detid>>25)&0x7;
      int ring = 0;
      if(subdet == 6) ring = (detid>>5)&0x7;
      if(subdet == 4) ring = (detid>>9)&0x3;
      int geom = ring + ((subdet==6) ? 3 : 0);
      if(subdet==3) geom +=10 ;
      if(subdet==5) geom +=15 ;

      // Save
      geometries_[bin1.str()] = geom;
      geometries_[bin2.str()] = geom;    
    }
  }
}

void CalibrationScanAnalysis::loadPresentValues() {

  //categorize APVs per module geometry
  std::cout << "Reading present ISHA/VFS values from debug.log" << std::endl;
  ifstream debuglog("debug.log");
  char buffer[1024];
  while(debuglog.getline(buffer,1024)) {
    if(strncmp(buffer,"Present values for ISHA/VFS",27)==0) {
      // Decode input
      int isha, vfs;
      char apv_addr[256];
      sscanf(strstr(buffer,"APV"),"APV %s : %d %d", apv_addr,&isha,&vfs);
      // Save
      std::string apv_address = apv_addr;
      presentValues_[apv_address] = std::make_pair(isha,vfs);
    }
  }

}

void CalibrationScanAnalysis::sanitizeResult(unsigned int cut) {

  // create and fill the utility histograms (similar to the draw method)
  std::map<int,TH2F*> histos;
  for(std::map<std::string, int>::iterator it = geometries_.begin(); it!= geometries_.end(); ++it) {
    if(histos.find(it->second)==histos.end()) {
      TH2F* histo = new TH2F(Form("modulesGeometry%d",it->second),
                             Form("Module Geometry %d",it->second),255,0,255,255,0,255);
      histos[it->second] = histo;
    }
  }

  // first loop to compute mean and rms
  for(std::map<std::string, Parameters>::const_iterator apvValue = result_.begin();
      apvValue != result_.end(); ++apvValue) {
    histos[geometries_[apvValue->first]]->Fill(apvValue->second.first,apvValue->second.second);
  }

  // second loop to cut at x RMS
  int lowVFS,highVFS,lowISHA,highISHA;
  for(std::map<std::string, Parameters>::iterator apvValue = result_.begin();
      apvValue != result_.end(); ++apvValue) {
    lowISHA  = (int)round(histos[geometries_[apvValue->first]]->GetMean(1) - 
                          cut*histos[geometries_[apvValue->first]]->GetRMS(1));
    highISHA = (int)round(histos[geometries_[apvValue->first]]->GetMean(1) + 
                          cut*histos[geometries_[apvValue->first]]->GetRMS(1));
    lowVFS   = (int)round(histos[geometries_[apvValue->first]]->GetMean(2) - 
                          cut*histos[geometries_[apvValue->first]]->GetRMS(2));
    highVFS  = (int)round(histos[geometries_[apvValue->first]]->GetMean(2) + 
                          cut*histos[geometries_[apvValue->first]]->GetRMS(2));
    if(apvValue->second.first<lowISHA || apvValue->second.first>highISHA) { 
      apvValue->second.first = (int)round((lowISHA+highISHA)/2.);
    }
    if(apvValue->second.second<lowVFS || apvValue->second.second>highVFS) {
      apvValue->second.second = (int)round((lowVFS+highVFS)/2.);
    }
  }

  // finaly delete the temporary histograms
  for(std::map<int,TH2F*>::iterator it = histos.begin(); it!=histos.end(); ++it) {
    delete it->second;
  }

}

void CalibrationScanAnalysis::print(Option_t*) const {
  
  // input
  std::cout << "Analysis of ISHA/VFS performed using the following inputs: " << std::endl;
  std::cout << "ISHA \t VFS \t File" << std::endl;
  for(FileList::const_iterator file=files_.begin();file!=files_.end();++file){
    int isha = file->first.first;
    int vfs  = file->first.second;
    std::string filename = file->second->GetName();
    std::cout << isha << "\t " << vfs << "\t " << filename << std::endl;
  }
  std::cout << std::endl << std::endl;

  // output
  std::cout << "Resulting values: " << std::endl;
  std::cout << "APV \t \t ISHA \t VFS" << std::endl;
  for(std::map<std::string, Parameters>::const_iterator result = result_.begin();
      result != result_.end(); ++result) {
    std::cout << result->first << "\t " 
              << result->second.first << "\t " 
              << result->second.second << std::endl;
  }

}

void CalibrationScanAnalysis::draw(Option_t*) const {

  new TCanvas;

  // first create the histograms
  std::map<int,TH2F*> histos;
  for(std::map<std::string, int>::const_iterator it = geometries_.begin(); it!= geometries_.end(); ++it) {
    if(histos.find(it->second)==histos.end()) {
      TH2F* histo = new TH2F(Form("modulesGeometry%d",it->second),
                             Form("Module Geometry %d",it->second),255,-1.25,0.6625,255,0,255);
      histo->GetXaxis()->SetTitle("VFS");
      histo->GetYaxis()->SetTitle("ISHA");
      histo->SetMarkerStyle(7);
      histo->SetMarkerColor(2+it->second);
      histos[it->second] = histo;
    }
  }

  // loop over apvs
  for(std::map<std::string, Parameters>::const_iterator apvValue = result_.begin();
      apvValue != result_.end(); ++apvValue) {
    histos[geometries_.find(apvValue->first)->second]->Fill(-1.25+apvValue->second.second*0.0075,apvValue->second.first);
  }

  // draw the histograms
  for(std::map<int,TH2F*>::iterator h = histos.begin(); h != histos.end(); ++h) {
    h->second->Draw(h == histos.begin() ? "" : "same");
  }

  // draw the histogram with the mean per geometry
  TH2F* histo = new TH2F("Geometries","Geometries",255,-1.25,0.6625,255,0,255);
  histo->GetXaxis()->SetTitle("VFS");
  histo->GetYaxis()->SetTitle("ISHA");
  histo->SetMarkerStyle(20);
  histo->SetMarkerColor(2);
  for(std::map<int,TH2F*>::iterator h = histos.begin(); h != histos.end(); ++h) {
    histo->Fill(h->second->GetMean(1),h->second->GetMean(2));
  }
  histo->Draw("same");
  
}

void CalibrationScanAnalysis::save() {
  for(FileList::const_iterator it=files_.begin();it!=files_.end();++it) {
    TFile* input = it->second;
    TDirectory* directory = input->GetDirectory(HISTOPATH);
    directory->cd();
    TList* histograms = directory->GetListOfKeys();
    TIter next(histograms);
    TKey* key = NULL;
    while ((key = (TKey*)next())) {
      if(TClass(key->GetClassName()).InheritsFrom("TH1")) {
        TH1* h = (TH1*)key->ReadObj();
        std::string name = h->GetName();
        if(name.find("CalibrationTail")!=std::string::npos) {
          TH1F* ishaOutput = new TH1F("isha","isha",h->GetNbinsX(),0,h->GetNbinsX());
          TH1F* vfsOutput = new TH1F("vfs","vfs",h->GetNbinsX(),0,h->GetNbinsX());
          for(int i=1;i<=h->GetNbinsX();++i) {
            ishaOutput->SetBinContent(i,result_[h->GetXaxis()->GetBinLabel(i)].first);
            vfsOutput ->SetBinContent(i,result_[h->GetXaxis()->GetBinLabel(i)].second);
            ishaOutput->GetXaxis()->SetBinLabel(i,h->GetXaxis()->GetBinLabel(i));
            vfsOutput ->GetXaxis()->SetBinLabel(i,h->GetXaxis()->GetBinLabel(i));
          }
          break;
        }
      }
    }
    input->Write();
  }
}

