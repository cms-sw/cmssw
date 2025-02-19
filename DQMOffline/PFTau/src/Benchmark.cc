#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <TDirectory.h>


using namespace std;

DQMStore* Benchmark::DQM_ = 0;

Benchmark::~Benchmark() {

}

void Benchmark::setDirectory(TDirectory* dir) {
  dir_ = dir;
  DQM_ = 0; 
} 


TH1F* Benchmark::book1D(const char* histname, const char* title, 
			int nbins, float xmin, float xmax) {
  // DQM_ has to be initialized in a child analyzer.
  if(DQM_) {
    edm::LogInfo("Benchmark") << " Benchmark::book1D " << "booking "<<histname;
    return DQM_->book1D(histname,title,nbins,xmin, xmax)->getTH1F();
  }
  else if(dir_){
    TDirectory *oldpwd = gDirectory; 
    dir_->cd();
    TH1F *hist =  new TH1F(histname, title, nbins, xmin, xmax);
    cout<<"booking (ROOT mode) "<<histname<<" in "<<dir_->GetName()<<endl;
    oldpwd->cd();
    return hist;
  }
  else assert(0);
}

TH2F* Benchmark::book2D(const char* histname, const char* title, 
			int nbinsx, float xmin, float xmax,
			int nbinsy, float ymin, float ymax ) {
  // DQM_ has to be initialized in a child analyzer.
  if(DQM_) {
    edm::LogInfo("Benchmark") << " Benchmark::book2D "<<"booked "<<histname;
    return DQM_->book2D(histname,title,nbinsx,xmin, xmax, nbinsy, ymin, ymax)->getTH2F();
  }
  else if(dir_) {
    TDirectory *oldpwd = gDirectory; 
    dir_->cd();
    TH2F *hist = new TH2F(histname, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax);
    cout<<"booked (ROOT mode) "<<histname<<" in "<<dir_->GetName()<<endl;
    oldpwd->cd();
    return hist;
  }
  else assert(0);
}

TH2F* Benchmark::book2D(const char* histname, const char* title, 
			int nbinsx, float* xbins,
			int nbinsy, float ymin, float ymax ) {
  if(DQM_) {
    edm::LogInfo("Benchmark") << " Benchmark::book2D " << " booked "<<histname;
    
    // need to build the y bin array manually, due to a missing 
    // function in DQMStore
    vector<float> ybins( nbinsy+1 );
    double binsize = (ymax - ymin) / nbinsy;
    for(int i=0; i<=nbinsy; ++i) {
      ybins[i] = ymin + i*binsize;
    } 
    
    return DQM_->book2D(histname,title,nbinsx, xbins, nbinsy, &ybins[0])->getTH2F();
  }
  else if(dir_) {
    TDirectory *oldpwd = gDirectory; 
    dir_->cd();

    // need to convert the float bin array into a double bin array,
    // because the DQMStore functions take floats, while the ROOT functions
    // take double. 
    vector<double> xbinsd(nbinsx+1); 
    for(int i=0; i<=nbinsx; ++i) {
      xbinsd[i] = xbins[i];
    }

    TH2F *hist = new TH2F(histname, title, nbinsx, &xbinsd[0], nbinsy, ymin, ymax);
    cout<<"booked (ROOT mode) "<<histname<<" in "<<dir_->GetName()<<endl;
    oldpwd->cd();
    return hist;
  }
  else assert(0);
}


void Benchmark::write() {
  //COLIN not sure about the root mode 
  if( dir_ )
    dir_->Write();

  //COLIN remove old bullshit:
//   if ( ame.size() != 0 && file_)
//     cout<<"saving histograms in "<<fileName<<endl;
//     file_->Write(fileName.c_str());
}
