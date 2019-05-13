#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <TDirectory.h>

using namespace std;

Benchmark::~Benchmark() noexcept(false) {}

void Benchmark::setDirectory(TDirectory *dir) { dir_ = dir; }

TH1F *Benchmark::book1D(
    DQMStore::IBooker &b, const char *histname, const char *title, int nbins, float xmin, float xmax) {
  edm::LogInfo("Benchmark") << " Benchmark::book1D "
                            << "booking " << histname;
  return b.book1D(histname, title, nbins, xmin, xmax)->getTH1F();
}

TH2F *Benchmark::book2D(DQMStore::IBooker &b,
                        const char *histname,
                        const char *title,
                        int nbinsx,
                        float xmin,
                        float xmax,
                        int nbinsy,
                        float ymin,
                        float ymax) {
  edm::LogInfo("Benchmark") << " Benchmark::book2D "
                            << "booked " << histname;
  return b.book2D(histname, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax)->getTH2F();
}

TH2F *Benchmark::book2D(DQMStore::IBooker &b,
                        const char *histname,
                        const char *title,
                        int nbinsx,
                        float *xbins,
                        int nbinsy,
                        float ymin,
                        float ymax) {
  edm::LogInfo("Benchmark") << " Benchmark::book2D "
                            << " booked " << histname;

  // need to build the y bin array manually, due to a missing function in
  // DQMStore
  vector<float> ybins(nbinsy + 1);
  double binsize = (ymax - ymin) / nbinsy;
  for (int i = 0; i <= nbinsy; ++i) {
    ybins[i] = ymin + i * binsize;
  }

  return b.book2D(histname, title, nbinsx, xbins, nbinsy, &ybins[0])->getTH2F();
}

TProfile *Benchmark::bookProfile(DQMStore::IBooker &b,
                                 const char *histname,
                                 const char *title,
                                 int nbinsx,
                                 float xmin,
                                 float xmax,
                                 float ymin,
                                 float ymax,
                                 const char *option) {
  edm::LogInfo("Benchmark") << " Benchmark::bookProfile "
                            << "booked " << histname;
  return b.bookProfile(histname, title, nbinsx, xmin, xmax, 0.0, 0.0, option)->getTProfile();
}

TProfile *Benchmark::bookProfile(DQMStore::IBooker &b,
                                 const char *histname,
                                 const char *title,
                                 int nbinsx,
                                 float *xbins,
                                 float ymin,
                                 float ymax,
                                 const char *option) {
  // need to convert the float bin array into a double bin array, because the
  // DQMStore TProfile functions take floats, while the  DQMStore TH2 functions
  // take double.
  vector<double> xbinsd(nbinsx + 1);
  for (int i = 0; i <= nbinsx; ++i) {
    xbinsd[i] = xbins[i];
  }

  edm::LogInfo("Benchmark") << " Benchmark::bookProfile "
                            << "booked " << histname;
  return b.bookProfile(histname, title, nbinsx, &xbinsd[0], ymin, ymax, option)->getTProfile();
}

void Benchmark::write() {
  // COLIN not sure about the root mode
  if (dir_)
    dir_->Write();

  // COLIN remove old bullshit:
  //   if ( ame.size() != 0 && file_)
  //     cout<<"saving histograms in "<<fileName<<endl;
  //     file_->Write(fileName.c_str());
}
