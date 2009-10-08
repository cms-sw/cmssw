#include "RecoParticleFlow/Benchmark/interface/Benchmark.h"


#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <TFile.h>


using namespace std;



Benchmark::~Benchmark() {}


TH1F* Benchmark::book1D(const char* histname, const char* title, 
			int nbins, float xmin, float xmax) {
  // DQM_ has to be initialized in a child analyzer.
  if(DQM_) {
    cout<<"booking "<<histname<<endl;
    return DQM_->book1D(histname,title,nbins,xmin, xmax)->getTH1F();
  }
  else {
    return new TH1F(histname, title, nbins, xmin, xmax);
  }
}

TH2F* Benchmark::book2D(const char* histname, const char* title, 
			int nbinsx, float xmin, float xmax,
			int nbinsy, float ymin, float ymax ) {
  // DQM_ has to be initialized in a child analyzer.
  if(DQM_) {
    cout<<"booking "<<histname<<endl;
    return DQM_->book2D(histname,title,nbinsx,xmin, xmax, nbinsy, ymin, ymax)->getTH2F();
  }
  else {
    return new TH2F(histname, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax);
  }
}


void Benchmark::write(std::string fileName) {
  //COLIN not sure about the root mode 
  if( file_ )
    file_->Write();

  //COLIN remove old bullshit:
//   if ( ame.size() != 0 && file_)
//     cout<<"saving histograms in "<<fileName<<endl;
//     file_->Write(fileName.c_str());
}
