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
#include <list>
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

//#define DEBUG_ON

class CalibrationScanAnalysis
{

  public:
    CalibrationScanAnalysis(bool tuneISHA = true, bool tuneVFS = true);
    virtual ~CalibrationScanAnalysis();
    void tuneISHA(bool tune) { tuneISHA_ = tune; }
    void tuneVFS(bool tune)  { tuneVFS_  = tune; }
    void addFile(const std::string&);
    void analyze();
    void sanitizeResult(unsigned int cut = 2, bool doItForISHA = true, bool doItForVFS = true);
    void print(Option_t* option = "") const;
    void draw(Option_t* option = "") const;
    void save(const char* fileName="-");

  protected:
    void addFile(TFile* );
    void getSummaries(FileList::const_iterator);
    void sortByGeometry();
    void loadPresentValues();
    float getX(const TGraph*, const float&) const;
    bool checkInput() const;
    TH1F* fixHisto(std::vector<std::string>&,TH1*) const;
    
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
  TFile* test = new TFile(filename.c_str());
  bool noFile = test->IsZombie();
  test->Close();
  delete test;
  if(!noFile) {
    TFile* newFile = new TFile(filename.c_str(),"UPDATE");
    addFile(newFile);
  }
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
  std::cout << "." << std::flush;
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
     if((arrayY[i]-y)*(arrayY[i+1]-y)<0) { 
       return (arrayX[i]+((arrayX[i+1]-arrayX[i])/(arrayY[i+1]-arrayY[i])*(y-arrayY[i])));
     }
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
  std::cout << "Loading data from files..." << std::endl;
  for(FileList::const_iterator it=files_.begin();it!=files_.end();++it) {
    getSummaries(it);
  }
  std::cout << endl;
  sortByGeometry();
  loadPresentValues();

  // sanity check
  if(!checkInput()) return;

  // check if both ISHA and VFS have to be tuned
  std::cout << "Preparing analysis..." << std::endl;
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
  // two cases are possible:
  // ISHA tune: look at the rise time
  // VFS  tune: look at the tail

  // number of APVs
  unsigned int nAPVs = (*(summaries_.begin()->second.begin()))->GetNbinsX();

  // loop over the inputs to find individual values of ISHA ans VFS
  std::list<unsigned int> ishaValues;
  std::list<unsigned int> vfsValues;
  for(SummaryV::const_iterator summary = summaries_.begin(); summary!=summaries_.end(); ++summary) {
     ishaValues.push_back(summary->first.first);
     vfsValues.push_back(summary->first.second);
  }
  ishaValues.sort();
  vfsValues.sort();
  ishaValues.unique();
  vfsValues.unique();

  // loop over apvs (bins)
  std::cout << "Running analysis..." << std::endl;
  for(unsigned int apv=1;apv<=nAPVs;++apv) {
     TGraph* g1 = new TGraph();
     TGraph* g2 = new TGraph();
     int ii=0;
     cout << "\r" << setw(5) << setfill('0') << apv << flush; 

     // loop over the VFS values
     for(std::list<unsigned int>::const_iterator vfs = vfsValues.begin(); vfs!=vfsValues.end(); ++vfs,++ii) {
       float tail = 0.;
       unsigned int npts = 0;
       for(SummaryV::const_iterator summary = summaries_.begin(); summary!=summaries_.end(); ++summary){
         if((unsigned int)summary->first.second==(*vfs)) {
           // determine which histogram are the rise time and the tail
           const std::vector<TH1*>& observables = summary->second;
           int tail_index = 0;
           int rise_index = 0;
           for( std::vector<TH1*>::const_iterator histo = observables.begin();histo<observables.end();++histo) {
              std::string name = (*histo)->GetName();
              if(name.find("CalibrationTail")!=std::string::npos) tail_index = histo-observables.begin();
              if(name.find("CalibrationRiseTime")!=std::string::npos) rise_index = histo-observables.begin();
           }
	   //for vfs, we take the mean tail over the ISHA values at that point
	   tail += observables[tail_index]->GetBinContent(apv);
	   ++npts;
	 }
       }
       // fill the graph
       g2->SetPoint(ii,(*vfs), tail/npts);
     }
#ifdef DEBUG_ON
     std::string name2 = Form("graph%s%s",summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv),"CalibrationTail");
     std::replace( name2.begin(), name2.end(), '.', '_' );
     g2->Write(name2.c_str());
#endif
     // analyse the graphs
     float best_vfs  = tuneVFS_  ? getX(g2,50) : 
                                   presentValues_[summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv)].second;
     // now that VFS is optimized, take the ISHA values for the closest VFS point
     // for ISHA, we consider the rise time for VFS values close to the optimal

     // find the closest point in the VFS scan
     float dist = 1000.;
     std::list<unsigned int>::const_iterator vfsPoint = vfsValues.begin();
     for(std::list<unsigned int>::const_iterator vfs = vfsValues.begin(); vfs!=vfsValues.end(); ++vfs) {
       if(dist>fabs((*vfs)-best_vfs)) {
         dist = fabs((*vfs)-best_vfs);
	 vfsPoint = vfs;
       }
     }
     // loop over the ISHA values
     ii=0;
     for(std::list<unsigned int>::const_iterator isha = ishaValues.begin(); isha!=ishaValues.end(); ++isha,++ii) {
       for(SummaryV::const_iterator summary = summaries_.begin(); summary!=summaries_.end(); ++summary){
         if(((unsigned int)summary->first.second==(*vfsPoint))&&((unsigned int)summary->first.first==(*isha))) {
           // determine which histogram are the rise time and the tail
           const std::vector<TH1*>& observables = summary->second;
           int tail_index = 0;
           int rise_index = 0;
           for( std::vector<TH1*>::const_iterator histo = observables.begin();histo<observables.end();++histo) {
              std::string name = (*histo)->GetName();
              if(name.find("CalibrationTail")!=std::string::npos) tail_index = histo-observables.begin();
              if(name.find("CalibrationRiseTime")!=std::string::npos) rise_index = histo-observables.begin();
           }
	   // fill the graph
	   g1->SetPoint(ii,summary->first.first,observables[rise_index]->GetBinContent(apv));
#ifdef DEBUG_ON
           std::string name1 = Form("graph%s%s",summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv),"CalibrationRiseTime");
           std::replace( name1.begin(), name1.end(), '.', '_' );
           g1->Write(name1.c_str());
#endif
	 }
       }
     }
     // analyse the graphs
     float best_isha = tuneISHA_ ? getX(g1,53.5 ) :
                                   presentValues_[summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv)].first;

     // save the result
     result_[summaries_.begin()->second[0]->GetXaxis()->GetBinLabel(apv)] = std::make_pair((int)round(best_isha),(int)round(best_vfs));

     // cleaning
     delete g1;
     delete g2;
  }
  std::cout << std::endl;

#ifdef DEBUG_ON
  debugFile->Write();
  debugFile->Close();
  delete debugFile;
#endif

}

bool CalibrationScanAnalysis::checkInput() const {

  // check that we have data
  std::cout << "Checking data integrity." << std::endl;
  std::cout << "Step 1/5" << std::endl;
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
  std::cout << "Step 2/5" << std::endl;
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
// non fatal
//         return 0;
       }
    }
  }

  // check that we have at least 2 histograms with measurements
  std::cout << "Step 3/5" << std::endl;
  if(namesAll.size()<2) {
    std::cerr << "Error: The number of available measurements is smaller than 2." << std::endl;
    return 0;
  }

  // check that the bin labels are all the same
  std::cout << "Step 4/5" << std::endl;
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
*((TH1F*)(*histo)) = TH1F(*(fixHisto(labelsAll,*histo)));
/*
             std::cerr << "Error: Incoherency in bin labels. Bin " << i 
                       << " of " << (*histo)->GetName() << " is " << label
                       << " and not " << labelsAll[i] << "." << std::endl;
             return 0;
*/
           }
         }
       }
    }
  }

  // check that all APVs have an associated geometry
  std::cout << "Step 5/5" << std::endl;
   for(std::vector<std::string>::const_iterator apvLabel = labelsAll.begin();
       apvLabel != labelsAll.end(); ++apvLabel) {
     if(geometries_.find(*apvLabel)==geometries_.end()) {
       std::cerr << "Error: Geometry unknown for APV " << *apvLabel << std::endl;
       // made this a non-fatal error
 //      return 0;
       std::string label = *apvLabel;
       ((CalibrationScanAnalysis*)this)->geometries_[label] = 0; 
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
      sscanf(strstr(buffer,"FEC:cr/sl/ring/ccu/mod"), "FEC:cr/sl/ring/ccu/mod=%d/%d/%d/%d/%d", &fecCrate,&fecSlot,&fecRing,&ccuAddr, &ccuChan);
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

void CalibrationScanAnalysis::sanitizeResult(unsigned int cut, bool doItForISHA, bool doItForVFS) {

  // create and fill the utility histograms (similar to the draw method)
  std::cout << "Applying sanity constraints on the results." << std::endl;
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
  int lowVFS,highVFS,lowISHA,highISHA,geom;
  for(std::map<std::string, Parameters>::iterator apvValue = result_.begin();
      apvValue != result_.end(); ++apvValue) {
    geom = geometries_[apvValue->first];
    lowISHA  = (int)round(histos[geom]->GetMean(1) - 
                          cut*histos[geom]->GetRMS(1));
    highISHA = (int)round(histos[geom]->GetMean(1) + 
                          cut*histos[geom]->GetRMS(1));
    lowVFS   = (int)round(histos[geom]->GetMean(2) - 
                          cut*histos[geom]->GetRMS(2));
    highVFS  = (int)round(histos[geom]->GetMean(2) + 
                          cut*histos[geom]->GetRMS(2));
    if((apvValue->second.first<lowISHA || apvValue->second.first>highISHA) && doItForISHA) { 
      apvValue->second.first = (int)round((lowISHA+highISHA)/2.);
    }
    if((apvValue->second.second<lowVFS || apvValue->second.second>highVFS) && doItForVFS) {
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

  std::cout << "Drawing results..." << std::endl;

  // first create the histograms
  std::cout << "   - first create the 2D histograms" << std::endl;
  new TCanvas;
  std::map<int,TH2F*> histos;
  for(std::map<std::string, int>::const_iterator it = geometries_.begin(); it!= geometries_.end(); ++it) {
    if(histos.find(it->second)==histos.end()) {
      TH2F* histo = new TH2F(Form("modulesGeometry%d",it->second),
                             Form("Module Geometry %d",it->second),255,-1.25,0.6625,255,0,255);
      histo->SetDirectory(0);
      histo->GetXaxis()->SetTitle("VFS");
      histo->GetYaxis()->SetTitle("ISHA");
      histo->SetMarkerStyle(7);
      histo->SetMarkerColor(2+it->second);
      histos[it->second] = histo;
    }
  }

  // loop over apvs
  std::cout << "   - loop over apvs" << std::endl;
  for(std::map<std::string, Parameters>::const_iterator apvValue = result_.begin();
      apvValue != result_.end(); ++apvValue) {
    histos[geometries_.find(apvValue->first)->second]->Fill(-1.25+apvValue->second.second*0.0075,apvValue->second.first);
  }

  // draw the histograms
  std::cout << "   - draw the histograms" << std::endl;
  for(std::map<int,TH2F*>::iterator h = histos.begin(); h != histos.end(); ++h) {
    h->second->Draw(h == histos.begin() ? "" : "same");
  }

  // draw the histogram with the mean per geometry
  std::cout << "   - draw the histogram with the mean per geometry" << std::endl;
  TH2F* histo = new TH2F("Geometries","Geometries",255,-1.25,0.6625,255,0,255);
  histo->SetDirectory(0);
  histo->GetXaxis()->SetTitle("VFS");
  histo->GetYaxis()->SetTitle("ISHA");
  histo->SetMarkerStyle(20);
  histo->SetMarkerColor(2);
  for(std::map<int,TH2F*>::iterator h = histos.begin(); h != histos.end(); ++h) {
    histo->Fill(h->second->GetMean(1),h->second->GetMean(2));
  }
  histo->Draw("same");
  
//////////////////////////////////////////////////////////////


  // first create the histograms
  std::cout << "   - create the 1D histograms" << std::endl;
  new TCanvas;
  std::map<int,TH1F*> histosVFS;
  std::map<int,TH1F*> histosISHA;
  for(std::map<std::string, int>::const_iterator it = geometries_.begin(); it!= geometries_.end(); ++it) {
    if(histosVFS.find(it->second)==histosVFS.end()) {
      TH1F* histoVFS = new TH1F(Form("VFSmodulesGeometry%d",it->second),
                                Form("VFS for Module Geometry %d",it->second),255,0,255);
      histoVFS->SetDirectory(0);
      histosVFS[it->second] = histoVFS;
      TH1F* histoISHA = new TH1F(Form("ISHAmodulesGeometry%d",it->second),
                                 Form("ISHA for Module Geometry %d",it->second),255,0,255);
      histoISHA->SetDirectory(0);
      histosISHA[it->second] = histoISHA;
    }
  }

  // loop over apvs
  std::cout << "   - loop over apvs" << std::endl;
  for(std::map<std::string, Parameters>::const_iterator apvValue = result_.begin();
      apvValue != result_.end(); ++apvValue) {
    histosISHA[geometries_.find(apvValue->first)->second]->Fill(apvValue->second.first);
    histosVFS[geometries_.find(apvValue->first)->second]->Fill(apvValue->second.second);
  }

  // draw the histograms
  std::cout << "   - draw the histograms" << std::endl;
  for(std::map<int,TH1F*>::iterator h = histosISHA.begin(); h != histosISHA.end(); ++h) {
    h->second->Draw(h == histosISHA.begin() ? "" : "same");
  }

  new TCanvas;
  
  for(std::map<int,TH1F*>::iterator h = histosVFS.begin(); h != histosVFS.end(); ++h) {
    h->second->Draw(h == histosVFS.begin() ? "" : "same");
  }

}

void CalibrationScanAnalysis::save(const char* fileName) {

  std::cout << "Saving results..." << std::endl;
/*
  // save in the input files
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
*/
  // save in a file for TkConfigurationDb
  std::ostream * output;
  TString filen = fileName;
  if (filen == "")
    return;
  if (filen == "-")
    output = &std::cout;
  else
    output = new std::ofstream(fileName);
  FileList::const_iterator it=files_.begin();
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
        for(int i=1;i<=h->GetNbinsX();++i) {
	  std::string address = h->GetXaxis()->GetBinLabel(i);
	  address = address.substr(address.find('.')+1);
	  std::replace(address.begin(),address.end(),'.',' ');
	  *output << address << " " 
	          << result_[h->GetXaxis()->GetBinLabel(i)].first << " " 
                  << result_[h->GetXaxis()->GetBinLabel(i)].second << std::endl;
        }
        break;
      }
    }
  }
}

TH1F* CalibrationScanAnalysis::fixHisto(std::vector<std::string>& names,TH1* histo) const
{
   // prepare an histogram to replace input
   TH1F* newHisto = new TH1F(histo->GetName(),histo->GetTitle(),names.size(),histo->GetXaxis()->GetXmin(),histo->GetXaxis()->GetXmax());
   std::cout << "fixing histo " << histo->GetName() << " at " << histo << " by " << newHisto << std::endl;
   for(std::vector<std::string>::iterator name=names.begin();name!=names.end();++name) {
     newHisto->GetXaxis()->SetBinLabel(name-names.begin()+1,name->c_str());
     int pos = histo->GetXaxis()->FindBin(name->c_str());
     if(pos!=-1) {
       newHisto->SetBinContent(pos,histo->GetBinContent(pos));
     }
   }
   return newHisto;
}
