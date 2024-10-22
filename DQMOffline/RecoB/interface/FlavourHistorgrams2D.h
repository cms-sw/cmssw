
#ifndef FlavourHistograms2D_H
#define FlavourHistograms2D_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// #include "BTagPlotPrintC.h"

#include "TH2F.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"

#include "DQMOffline/RecoB/interface/Tools.h"
#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"

#include <iostream>
#include <vector>
#include <string>

//
// class to describe Histo
//
template <class T, class G>
class FlavourHistograms2D {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  FlavourHistograms2D(TString baseNameTitle_,
                      TString baseNameDescription_,
                      int nBinsX_,
                      double lowerBoundX_,
                      double upperBoundX_,
                      int nBinsY_,
                      double lowerBoundY_,
                      double upperBoundY_,
                      std::string folder,
                      unsigned int mc,
                      bool createProfile,
                      DQMStore::IGetter &iget);

  FlavourHistograms2D(TString baseNameTitle_,
                      TString baseNameDescription_,
                      int nBinsX_,
                      double lowerBoundX_,
                      double upperBoundX_,
                      int nBinsY_,
                      double lowerBoundY_,
                      double upperBoundY_,
                      bool statistics_,
                      std::string folder,
                      unsigned int mc,
                      bool createProfile,
                      DQMStore::IBooker &ibook);

  virtual ~FlavourHistograms2D();

  // define arrays (if needed)
  //   void defineArray ( int * dimension , int max , int indexToPlot ) ;

  // fill entry
  // For single variables and arrays (for arrays only a single index can be filled)
  void fill(const int &flavour, const T &variableX, const G &variableY) const;
  void fill(const int &flavour, const T &variableX, const G &variableY, const float &w) const;

  // For single variables and arrays
  void fill(const int &flavour, const T *variableX, const G *variableY) const;
  void fill(const int &flavour, const T *variableX, const G *variableY, const float &w) const;

  void settitle(const char *titleX, const char *titleY);

  void plot(TPad *theCanvas = nullptr);
  void epsPlot(const std::string &name);

  // needed for efficiency computations -> this / b
  // (void : alternative would be not to overwrite the histos but to return a cloned HistoDescription)
  void divide(const FlavourHistograms2D<T, G> &bHD) const;
  void setEfficiencyFlag();

  inline void SetMaximum(const double &max) { theMax = max; }
  inline void SetMinimum(const double &min) { theMin = min; }

  // trivial access functions
  inline std::string baseNameTitle() const { return theBaseNameTitle; }
  inline std::string baseNameDescription() const { return theBaseNameDescription; }
  inline int nBinsX() const { return theNBinsX; }
  inline int nBinsY() const { return theNBinsY; }
  inline double lowerBoundX() const { return theLowerBoundX; }
  inline double upperBoundX() const { return theUpperBoundX; }
  inline double lowerBoundY() const { return theLowerBoundY; }
  inline double upperBoundY() const { return theUpperBoundY; }
  inline bool statistics() const { return theStatistics; }

  // access to the histos
  inline TH2F *histo_all() const { return theHisto_all->getTH2F(); }
  inline TH2F *histo_d() const { return theHisto_d->getTH2F(); }
  inline TH2F *histo_u() const { return theHisto_u->getTH2F(); }
  inline TH2F *histo_s() const { return theHisto_s->getTH2F(); }
  inline TH2F *histo_c() const { return theHisto_c->getTH2F(); }
  inline TH2F *histo_b() const { return theHisto_b->getTH2F(); }
  inline TH2F *histo_g() const { return theHisto_g->getTH2F(); }
  inline TH2F *histo_ni() const { return theHisto_ni->getTH2F(); }
  inline TH2F *histo_dus() const { return theHisto_dus->getTH2F(); }
  inline TH2F *histo_dusg() const { return theHisto_dusg->getTH2F(); }
  inline TH2F *histo_pu() const { return theHisto_pu->getTH2F(); }

  TProfile *profile_all() const { return theProfile_all->getTProfile(); }
  TProfile *profile_d() const { return theProfile_d->getTProfile(); }
  TProfile *profile_u() const { return theProfile_u->getTProfile(); }
  TProfile *profile_s() const { return theProfile_s->getTProfile(); }
  TProfile *profile_c() const { return theProfile_c->getTProfile(); }
  TProfile *profile_b() const { return theProfile_b->getTProfile(); }
  TProfile *profile_g() const { return theProfile_g->getTProfile(); }
  TProfile *profile_ni() const { return theProfile_ni->getTProfile(); }
  TProfile *profile_dus() const { return theProfile_dus->getTProfile(); }
  TProfile *profile_dusg() const { return theProfile_dusg->getTProfile(); }
  TProfile *profile_pu() const { return theProfile_pu->getTProfile(); }

  std::vector<TH2F *> getHistoVector() const;

  std::vector<TProfile *> getProfileVector() const;

protected:
  void fillVariable(const int &flavour, const T &varX, const G &varY, const float &w) const;

  //
  // the data members
  //

  //   T *   theVariable   ;

  // for arrays
  int *theArrayDimension;
  int theMaxDimension;
  int theIndexToPlot;  // in case that not the complete array has to be plotted

  std::string theBaseNameTitle;
  std::string theBaseNameDescription;
  int theNBinsX;
  int theNBinsY;
  double theLowerBoundX;
  double theUpperBoundX;
  double theLowerBoundY;
  double theUpperBoundY;
  bool theStatistics;
  double theMin, theMax;

  // the histos
  MonitorElement *theHisto_all;
  MonitorElement *theHisto_d;
  MonitorElement *theHisto_u;
  MonitorElement *theHisto_s;
  MonitorElement *theHisto_c;
  MonitorElement *theHisto_b;
  MonitorElement *theHisto_g;
  MonitorElement *theHisto_ni;
  MonitorElement *theHisto_dus;
  MonitorElement *theHisto_dusg;
  MonitorElement *theHisto_pu;

  // the profiles
  MonitorElement *theProfile_all;
  MonitorElement *theProfile_d;
  MonitorElement *theProfile_u;
  MonitorElement *theProfile_s;
  MonitorElement *theProfile_c;
  MonitorElement *theProfile_b;
  MonitorElement *theProfile_g;
  MonitorElement *theProfile_ni;
  MonitorElement *theProfile_dus;
  MonitorElement *theProfile_dusg;
  MonitorElement *theProfile_pu;

  //  DQMStore * dqmStore_;

  // the canvas to plot
private:
  FlavourHistograms2D() {}

  unsigned int mcPlots_;
  bool createProfile_;
};

template <class T, class G>
FlavourHistograms2D<T, G>::FlavourHistograms2D(TString baseNameTitle_,
                                               TString baseNameDescription_,
                                               int nBinsX_,
                                               double lowerBoundX_,
                                               double upperBoundX_,
                                               int nBinsY_,
                                               double lowerBoundY_,
                                               double upperBoundY_,
                                               std::string folder,
                                               unsigned int mc,
                                               bool createProfile,
                                               DQMStore::IGetter &iget)
    :  // BaseFlavourHistograms2D () ,
      // theVariable ( variable_ ) ,
      theMaxDimension(-1),
      theIndexToPlot(-1),
      theBaseNameTitle(baseNameTitle_.Data()),
      theBaseNameDescription(baseNameDescription_.Data()),
      theNBinsX(nBinsX_),
      theNBinsY(nBinsY_),
      theLowerBoundX(lowerBoundX_),
      theUpperBoundX(upperBoundX_),
      theLowerBoundY(lowerBoundY_),
      theUpperBoundY(upperBoundY_),
      theMin(-1.),
      theMax(-1.),
      mcPlots_(mc),
      createProfile_(createProfile) {
  // defaults for array dimensions
  theArrayDimension = nullptr;

  if (mcPlots_ % 2 == 0)
    theHisto_all = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "ALL");
  else
    theHisto_all = nullptr;
  if (mcPlots_) {
    if (mcPlots_ > 2) {
      theHisto_d = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "D");
      theHisto_u = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "U");
      theHisto_s = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "S");
      theHisto_g = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "G");
      theHisto_dus = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "DUS");
    } else {
      theHisto_d = nullptr;
      theHisto_u = nullptr;
      theHisto_s = nullptr;
      theHisto_g = nullptr;
      theHisto_dus = nullptr;
    }
    theHisto_c = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "C");
    theHisto_b = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "B");
    theHisto_ni = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "NI");
    theHisto_dusg = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "DUSG");
    theHisto_pu = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "PU");
  } else {
    theHisto_d = nullptr;
    theHisto_u = nullptr;
    theHisto_s = nullptr;
    theHisto_c = nullptr;
    theHisto_b = nullptr;
    theHisto_g = nullptr;
    theHisto_ni = nullptr;
    theHisto_dus = nullptr;
    theHisto_dusg = nullptr;
    theHisto_pu = nullptr;
  }

  if (createProfile_) {
    if (mcPlots_ % 2 == 0)
      theProfile_all = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_ALL");
    else
      theProfile_all = nullptr;
    if (mcPlots_) {
      if (mcPlots_ > 2) {
        theProfile_d = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_D");
        theProfile_u = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_U");
        theProfile_s = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_S");
        theProfile_g = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_G");
        theProfile_dus = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_DUS");
      } else {
        theProfile_d = nullptr;
        theProfile_u = nullptr;
        theProfile_s = nullptr;
        theProfile_g = nullptr;
        theProfile_dus = nullptr;
      }
      theProfile_c = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_C");
      theProfile_b = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_B");
      theProfile_ni = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_NI");
      theProfile_dusg = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_DUSG");
      theProfile_pu = iget.get("Btag/" + folder + "/" + theBaseNameTitle + "_Profile_PU");
    } else {
      theProfile_d = nullptr;
      theProfile_u = nullptr;
      theProfile_s = nullptr;
      theProfile_c = nullptr;
      theProfile_b = nullptr;
      theProfile_g = nullptr;
      theProfile_ni = nullptr;
      theProfile_dus = nullptr;
      theProfile_dusg = nullptr;
      theProfile_pu = nullptr;
    }
  } else {
    theProfile_all = nullptr;
    theProfile_d = nullptr;
    theProfile_u = nullptr;
    theProfile_s = nullptr;
    theProfile_c = nullptr;
    theProfile_b = nullptr;
    theProfile_g = nullptr;
    theProfile_ni = nullptr;
    theProfile_dus = nullptr;
    theProfile_dusg = nullptr;
    theProfile_pu = nullptr;
  }
}

template <class T, class G>
FlavourHistograms2D<T, G>::FlavourHistograms2D(TString baseNameTitle_,
                                               TString baseNameDescription_,
                                               int nBinsX_,
                                               double lowerBoundX_,
                                               double upperBoundX_,
                                               int nBinsY_,
                                               double lowerBoundY_,
                                               double upperBoundY_,
                                               bool statistics_,
                                               std::string folder,
                                               unsigned int mc,
                                               bool createProfile,
                                               DQMStore::IBooker &ibook)
    : theMaxDimension(-1),
      theIndexToPlot(-1),
      theBaseNameTitle(baseNameTitle_.Data()),
      theBaseNameDescription(baseNameDescription_.Data()),
      theNBinsX(nBinsX_),
      theNBinsY(nBinsY_),
      theLowerBoundX(lowerBoundX_),
      theUpperBoundX(upperBoundX_),
      theLowerBoundY(lowerBoundY_),
      theUpperBoundY(upperBoundY_),
      theStatistics(statistics_),
      theMin(-1.),
      theMax(-1.),
      mcPlots_(mc),
      createProfile_(createProfile) {
  // defaults for array dimensions
  theArrayDimension = nullptr;

  // book histos
  HistoProviderDQM prov("Btag", folder, ibook);
  if (mcPlots_ % 2 == 0)
    theHisto_all = (prov.book2D(theBaseNameTitle + "ALL",
                                theBaseNameDescription + " all jets",
                                theNBinsX,
                                theLowerBoundX,
                                theUpperBoundX,
                                theNBinsY,
                                theLowerBoundY,
                                theUpperBoundY));
  else
    theHisto_all = nullptr;
  if (mcPlots_) {
    if (mcPlots_ > 2) {
      theHisto_d = (prov.book2D(theBaseNameTitle + "D",
                                theBaseNameDescription + " d-jets",
                                theNBinsX,
                                theLowerBoundX,
                                theUpperBoundX,
                                theNBinsY,
                                theLowerBoundY,
                                theUpperBoundY));
      theHisto_u = (prov.book2D(theBaseNameTitle + "U",
                                theBaseNameDescription + " u-jets",
                                theNBinsX,
                                theLowerBoundX,
                                theUpperBoundX,
                                theNBinsY,
                                theLowerBoundY,
                                theUpperBoundY));
      theHisto_s = (prov.book2D(theBaseNameTitle + "S",
                                theBaseNameDescription + " s-jets",
                                theNBinsX,
                                theLowerBoundX,
                                theUpperBoundX,
                                theNBinsY,
                                theLowerBoundY,
                                theUpperBoundY));
      theHisto_g = (prov.book2D(theBaseNameTitle + "G",
                                theBaseNameDescription + " g-jets",
                                theNBinsX,
                                theLowerBoundX,
                                theUpperBoundX,
                                theNBinsY,
                                theLowerBoundY,
                                theUpperBoundY));
      theHisto_dus = (prov.book2D(theBaseNameTitle + "DUS",
                                  theBaseNameDescription + " dus-jets",
                                  theNBinsX,
                                  theLowerBoundX,
                                  theUpperBoundX,
                                  theNBinsY,
                                  theLowerBoundY,
                                  theUpperBoundY));
    } else {
      theHisto_d = nullptr;
      theHisto_u = nullptr;
      theHisto_s = nullptr;
      theHisto_g = nullptr;
      theHisto_dus = nullptr;
    }
    theHisto_c = (prov.book2D(theBaseNameTitle + "C",
                              theBaseNameDescription + " c-jets",
                              theNBinsX,
                              theLowerBoundX,
                              theUpperBoundX,
                              theNBinsY,
                              theLowerBoundY,
                              theUpperBoundY));
    theHisto_b = (prov.book2D(theBaseNameTitle + "B",
                              theBaseNameDescription + " b-jets",
                              theNBinsX,
                              theLowerBoundX,
                              theUpperBoundX,
                              theNBinsY,
                              theLowerBoundY,
                              theUpperBoundY));
    theHisto_ni = (prov.book2D(theBaseNameTitle + "NI",
                               theBaseNameDescription + " ni-jets",
                               theNBinsX,
                               theLowerBoundX,
                               theUpperBoundX,
                               theNBinsY,
                               theLowerBoundY,
                               theUpperBoundY));
    theHisto_dusg = (prov.book2D(theBaseNameTitle + "DUSG",
                                 theBaseNameDescription + " dusg-jets",
                                 theNBinsX,
                                 theLowerBoundX,
                                 theUpperBoundX,
                                 theNBinsY,
                                 theLowerBoundY,
                                 theUpperBoundY));
    theHisto_pu = (prov.book2D(theBaseNameTitle + "PU",
                               theBaseNameDescription + " pu-jets",
                               theNBinsX,
                               theLowerBoundX,
                               theUpperBoundX,
                               theNBinsY,
                               theLowerBoundY,
                               theUpperBoundY));
  } else {
    theHisto_d = nullptr;
    theHisto_u = nullptr;
    theHisto_s = nullptr;
    theHisto_c = nullptr;
    theHisto_b = nullptr;
    theHisto_g = nullptr;
    theHisto_ni = nullptr;
    theHisto_dus = nullptr;
    theHisto_dusg = nullptr;
    theHisto_pu = nullptr;
  }

  if (createProfile_) {
    if (mcPlots_ % 2 == 0)
      theProfile_all = (prov.bookProfile(theBaseNameTitle + "_Profile_ALL",
                                         theBaseNameDescription + " all jets",
                                         theNBinsX,
                                         theLowerBoundX,
                                         theUpperBoundX,
                                         theNBinsY,
                                         theLowerBoundY,
                                         theUpperBoundY));
    else
      theProfile_all = nullptr;
    if (mcPlots_) {
      if (mcPlots_ > 2) {
        theProfile_d = (prov.bookProfile(theBaseNameTitle + "_Profile_D",
                                         theBaseNameDescription + " d-jets",
                                         theNBinsX,
                                         theLowerBoundX,
                                         theUpperBoundX,
                                         theNBinsY,
                                         theLowerBoundY,
                                         theUpperBoundY));
        theProfile_u = (prov.bookProfile(theBaseNameTitle + "_Profile_U",
                                         theBaseNameDescription + " u-jets",
                                         theNBinsX,
                                         theLowerBoundX,
                                         theUpperBoundX,
                                         theNBinsY,
                                         theLowerBoundY,
                                         theUpperBoundY));
        theProfile_s = (prov.bookProfile(theBaseNameTitle + "_Profile_S",
                                         theBaseNameDescription + " s-jets",
                                         theNBinsX,
                                         theLowerBoundX,
                                         theUpperBoundX,
                                         theNBinsY,
                                         theLowerBoundY,
                                         theUpperBoundY));
        theProfile_g = (prov.bookProfile(theBaseNameTitle + "_Profile_G",
                                         theBaseNameDescription + " g-jets",
                                         theNBinsX,
                                         theLowerBoundX,
                                         theUpperBoundX,
                                         theNBinsY,
                                         theLowerBoundY,
                                         theUpperBoundY));
        theProfile_dus = (prov.bookProfile(theBaseNameTitle + "_Profile_DUS",
                                           theBaseNameDescription + " dus-jets",
                                           theNBinsX,
                                           theLowerBoundX,
                                           theUpperBoundX,
                                           theNBinsY,
                                           theLowerBoundY,
                                           theUpperBoundY));
      } else {
        theProfile_d = nullptr;
        theProfile_u = nullptr;
        theProfile_s = nullptr;
        theProfile_g = nullptr;
        theProfile_dus = nullptr;
      }
      theProfile_c = (prov.bookProfile(theBaseNameTitle + "_Profile_C",
                                       theBaseNameDescription + " c-jets",
                                       theNBinsX,
                                       theLowerBoundX,
                                       theUpperBoundX,
                                       theNBinsY,
                                       theLowerBoundY,
                                       theUpperBoundY));
      theProfile_b = (prov.bookProfile(theBaseNameTitle + "_Profile_B",
                                       theBaseNameDescription + " b-jets",
                                       theNBinsX,
                                       theLowerBoundX,
                                       theUpperBoundX,
                                       theNBinsY,
                                       theLowerBoundY,
                                       theUpperBoundY));
      theProfile_ni = (prov.bookProfile(theBaseNameTitle + "_Profile_NI",
                                        theBaseNameDescription + " ni-jets",
                                        theNBinsX,
                                        theLowerBoundX,
                                        theUpperBoundX,
                                        theNBinsY,
                                        theLowerBoundY,
                                        theUpperBoundY));
      theProfile_dusg = (prov.bookProfile(theBaseNameTitle + "_Profile_DUSG",
                                          theBaseNameDescription + " dusg-jets",
                                          theNBinsX,
                                          theLowerBoundX,
                                          theUpperBoundX,
                                          theNBinsY,
                                          theLowerBoundY,
                                          theUpperBoundY));
      theProfile_pu = (prov.bookProfile(theBaseNameTitle + "_Profile_PU",
                                        theBaseNameDescription + " pu-jets",
                                        theNBinsX,
                                        theLowerBoundX,
                                        theUpperBoundX,
                                        theNBinsY,
                                        theLowerBoundY,
                                        theUpperBoundY));
    } else {
      theProfile_d = nullptr;
      theProfile_u = nullptr;
      theProfile_s = nullptr;
      theProfile_c = nullptr;
      theProfile_b = nullptr;
      theProfile_g = nullptr;
      theProfile_ni = nullptr;
      theProfile_dus = nullptr;
      theProfile_dusg = nullptr;
      theProfile_pu = nullptr;
    }
  } else {
    theProfile_all = nullptr;
    theProfile_d = nullptr;
    theProfile_u = nullptr;
    theProfile_s = nullptr;
    theProfile_c = nullptr;
    theProfile_b = nullptr;
    theProfile_g = nullptr;
    theProfile_ni = nullptr;
    theProfile_dus = nullptr;
    theProfile_dusg = nullptr;
    theProfile_pu = nullptr;
  }
  // statistics if requested
  if (theStatistics) {
    if (theHisto_all)
      theHisto_all->enableSumw2();
    if (createProfile)
      if (theProfile_all)
        theProfile_all->getTProfile()->Sumw2();
    if (mcPlots_) {
      if (mcPlots_ > 2) {
        theHisto_d->enableSumw2();
        theHisto_u->enableSumw2();
        theHisto_s->enableSumw2();
        theHisto_g->enableSumw2();
        theHisto_dus->enableSumw2();
      }
      theHisto_c->enableSumw2();
      theHisto_b->enableSumw2();
      theHisto_ni->enableSumw2();
      theHisto_dusg->enableSumw2();
      theHisto_pu->enableSumw2();

      if (createProfile) {
        if (mcPlots_ > 2) {
          theProfile_d->getTProfile()->Sumw2();
          theProfile_u->getTProfile()->Sumw2();
          theProfile_s->getTProfile()->Sumw2();
          theProfile_g->getTProfile()->Sumw2();
          theProfile_dus->getTProfile()->Sumw2();
        }
        theProfile_c->getTProfile()->Sumw2();
        theProfile_b->getTProfile()->Sumw2();
        theProfile_ni->getTProfile()->Sumw2();
        theProfile_dusg->getTProfile()->Sumw2();
        theProfile_pu->getTProfile()->Sumw2();
      }
    }
  }
}

template <class T, class G>
FlavourHistograms2D<T, G>::~FlavourHistograms2D() {}

// fill entry
template <class T, class G>
void FlavourHistograms2D<T, G>::fill(const int &flavour, const T &variableX, const G &variableY, const float &w) const {
  // For single variables and arrays (for arrays only a single index can be filled)
  fillVariable(flavour, variableX, variableY, w);
}

template <class T, class G>
void FlavourHistograms2D<T, G>::fill(const int &flavour, const T &variableX, const G &variableY) const {
  fill(flavour, variableX, variableY, 1.);
}

template <class T, class G>
void FlavourHistograms2D<T, G>::fill(const int &flavour, const T *variableX, const G *variableY, const float &w) const {
  if (theArrayDimension == nullptr) {
    // single variable
    fillVariable(flavour, *variableX, *variableY, w);
  } else {
    // array
    int iMax = *theArrayDimension;
    if (*theArrayDimension > theMaxDimension)
      iMax = theMaxDimension;
    //
    for (int i = 0; i != iMax; ++i) {
      // check if only one index to be plotted (<0: switched off -> plot all)
      if ((theIndexToPlot < 0) || (i == theIndexToPlot)) {
        fillVariable(flavour, *(variableX + i), *(variableY + i), w);
      }
    }

    // if single index to be filled but not enough entries: fill 0.0 (convention!)
    if (theIndexToPlot >= iMax) {
      // cout << "==>> The index to be filled is too big -> fill 0.0 : " << theBaseNameTitle << " : " << theIndexToPlot << " >= " << iMax << endl ;
      const T &theZeroT = static_cast<T>(0.0);
      const G &theZeroG = static_cast<T>(0.0);
      fillVariable(flavour, theZeroT, theZeroG, w);
    }
  }
}

template <class T, class G>
void FlavourHistograms2D<T, G>::fill(const int &flavour, const T *variableX, const G *variableY) const {
  fill(flavour, variableX, variableY, 1.);
}

template <class T, class G>
void FlavourHistograms2D<T, G>::settitle(const char *titleX, const char *titleY) {
  if (theHisto_all)
    theHisto_all->setAxisTitle(titleX);
  if (theHisto_all)
    theHisto_all->setAxisTitle(titleY, 2);
  if (mcPlots_) {
    if (theHisto_d)
      theHisto_d->setAxisTitle(titleX);
    if (theHisto_u)
      theHisto_u->setAxisTitle(titleX);
    if (theHisto_s)
      theHisto_s->setAxisTitle(titleX);
    if (theHisto_c)
      theHisto_c->setAxisTitle(titleX);
    if (theHisto_b)
      theHisto_b->setAxisTitle(titleX);
    if (theHisto_g)
      theHisto_g->setAxisTitle(titleX);
    if (theHisto_ni)
      theHisto_ni->setAxisTitle(titleX);
    if (theHisto_dus)
      theHisto_dus->setAxisTitle(titleX);
    if (theHisto_dusg)
      theHisto_dusg->setAxisTitle(titleX);
    if (theHisto_d)
      theHisto_d->setAxisTitle(titleY, 2);
    if (theHisto_u)
      theHisto_u->setAxisTitle(titleY, 2);
    if (theHisto_s)
      theHisto_s->setAxisTitle(titleY, 2);
    if (theHisto_c)
      theHisto_c->setAxisTitle(titleY, 2);
    if (theHisto_b)
      theHisto_b->setAxisTitle(titleY, 2);
    if (theHisto_g)
      theHisto_g->setAxisTitle(titleY, 2);
    if (theHisto_ni)
      theHisto_ni->setAxisTitle(titleY, 2);
    if (theHisto_dus)
      theHisto_dus->setAxisTitle(titleY, 2);
    if (theHisto_dusg)
      theHisto_dusg->setAxisTitle(titleY, 2);
    if (theHisto_pu)
      theHisto_pu->setAxisTitle(titleY, 2);
  }

  if (createProfile_) {
    if (theProfile_all)
      theProfile_all->setAxisTitle(titleX);
    if (theProfile_all)
      theProfile_all->setAxisTitle(titleY, 2);
    if (mcPlots_) {
      if (theProfile_d)
        theProfile_d->setAxisTitle(titleX);
      if (theProfile_u)
        theProfile_u->setAxisTitle(titleX);
      if (theProfile_s)
        theProfile_s->setAxisTitle(titleX);
      if (theProfile_c)
        theProfile_c->setAxisTitle(titleX);
      if (theProfile_b)
        theProfile_b->setAxisTitle(titleX);
      if (theProfile_g)
        theProfile_g->setAxisTitle(titleX);
      if (theProfile_ni)
        theProfile_ni->setAxisTitle(titleX);
      if (theProfile_dus)
        theProfile_dus->setAxisTitle(titleX);
      if (theProfile_dusg)
        theProfile_dusg->setAxisTitle(titleX);
      if (theProfile_d)
        theProfile_d->setAxisTitle(titleY, 2);
      if (theProfile_u)
        theProfile_u->setAxisTitle(titleY, 2);
      if (theProfile_s)
        theProfile_s->setAxisTitle(titleY, 2);
      if (theProfile_c)
        theProfile_c->setAxisTitle(titleY, 2);
      if (theProfile_b)
        theProfile_b->setAxisTitle(titleY, 2);
      if (theProfile_g)
        theProfile_g->setAxisTitle(titleY, 2);
      if (theProfile_ni)
        theProfile_ni->setAxisTitle(titleY, 2);
      if (theProfile_dus)
        theProfile_dus->setAxisTitle(titleY, 2);
      if (theProfile_dusg)
        theProfile_dusg->setAxisTitle(titleY, 2);
      if (theProfile_pu)
        theProfile_pu->setAxisTitle(titleY, 2);
    }
  }
}

template <class T, class G>
void FlavourHistograms2D<T, G>::plot(TPad *theCanvas /* = 0 */) {
  //fixme:
  bool btppNI = false;
  bool btppColour = true;

  if (theCanvas)
    theCanvas->cd();

  RecoBTag::setTDRStyle()->cd();
  gPad->UseCurrentStyle();
  gPad->SetLogy(0);
  gPad->SetGridx(0);
  gPad->SetGridy(0);
  gPad->SetTitle(nullptr);

  MonitorElement *histo[4];
  int col[4], lineStyle[4], markerStyle[4];
  int lineWidth = 1;

  const double markerSize = gPad->GetWh() * gPad->GetHNDC() / 500.;
  histo[0] = theHisto_dusg;
  histo[1] = theHisto_b;
  histo[2] = theHisto_c;
  histo[3] = nullptr;

  double max = theMax;
  if (theMax <= 0.) {
    max = theHisto_dusg->getTH2F()->GetMaximum();
    if (theHisto_b->getTH2F()->GetMaximum() > max)
      max = theHisto_b->getTH2F()->GetMaximum();
    if (theHisto_c->getTH2F()->GetMaximum() > max)
      max = theHisto_c->getTH2F()->GetMaximum();
  }

  if (btppNI) {
    histo[3] = theHisto_ni;
    if (theHisto_ni->getTH2F()->GetMaximum() > max)
      max = theHisto_ni->getTH2F()->GetMaximum();
  }

  if (btppColour) {  // print colours
    col[0] = 4;
    col[1] = 2;
    col[2] = 6;
    col[3] = 3;
    lineStyle[0] = 1;
    lineStyle[1] = 1;
    lineStyle[2] = 1;
    lineStyle[3] = 1;
    markerStyle[0] = 20;
    markerStyle[1] = 21;
    markerStyle[2] = 22;
    markerStyle[3] = 23;
    lineWidth = 1;
  } else {  // different marker/line styles
    col[1] = 1;
    col[2] = 1;
    col[3] = 1;
    col[0] = 1;
    lineStyle[0] = 2;
    lineStyle[1] = 1;
    lineStyle[2] = 3;
    lineStyle[3] = 4;
    markerStyle[0] = 20;
    markerStyle[1] = 21;
    markerStyle[2] = 22;
    markerStyle[3] = 23;
  }

  histo[0]->setAxisTitle(theBaseNameDescription);
  histo[0]->getTH2F()->GetYaxis()->SetTitle("Arbitrary Units");
  histo[0]->getTH2F()->GetYaxis()->SetTitleOffset(1.25);

  for (int i = 0; i != 4; ++i) {
    if (histo[i] == nullptr)
      continue;
    histo[i]->getTH2F()->SetStats(false);
    histo[i]->getTH2F()->SetLineStyle(lineStyle[i]);
    histo[i]->getTH2F()->SetLineWidth(lineWidth);
    histo[i]->getTH2F()->SetLineColor(col[i]);
    histo[i]->getTH2F()->SetMarkerStyle(markerStyle[i]);
    histo[i]->getTH2F()->SetMarkerColor(col[i]);
    histo[i]->getTH2F()->SetMarkerSize(markerSize);
  }

  histo[0]->getTH2F()->SetMaximum(max * 1.05);
  if (theMin != -1.)
    histo[0]->getTH2F()->SetMinimum(theMin);
  histo[0]->getTH2F()->Draw();
  histo[1]->getTH2F()->Draw("Same");
  histo[2]->getTH2F()->Draw("Same");
  if (histo[3] != nullptr)
    histo[3]->getTH2F()->Draw("Same");
}

template <class T, class G>
void FlavourHistograms2D<T, G>::epsPlot(const std::string &name) {
  TCanvas tc(theBaseNameTitle.c_str(), theBaseNameDescription.c_str());

  plot(&tc);
  tc.Print((name + theBaseNameTitle + ".eps").c_str());
}

// needed for efficiency computations -> this / b
// (void : alternative would be not to overwrite the histos but to return a cloned HistoDescription)
template <class T, class G>
void FlavourHistograms2D<T, G>::divide(const FlavourHistograms2D<T, G> &bHD) const {
  // divide histos using binomial errors
  //
  // ATTENTION: It's the responsability of the user to make sure that the HistoDescriptions
  //            involved in this operation have been constructed with the statistics option switched on!!
  //
  if (theHisto_all)
    theHisto_all->getTH2F()->Divide(theHisto_all->getTH2F(), bHD.histo_all(), 1.0, 1.0, "b");
  if (mcPlots_) {
    if (mcPlots_ > 2) {
      theHisto_d->getTH2F()->Divide(theHisto_d->getTH2F(), bHD.histo_d(), 1.0, 1.0, "b");
      theHisto_u->getTH2F()->Divide(theHisto_u->getTH2F(), bHD.histo_u(), 1.0, 1.0, "b");
      theHisto_s->getTH2F()->Divide(theHisto_s->getTH2F(), bHD.histo_s(), 1.0, 1.0, "b");
      theHisto_g->getTH2F()->Divide(theHisto_g->getTH2F(), bHD.histo_g(), 1.0, 1.0, "b");
      theHisto_dus->getTH2F()->Divide(theHisto_dus->getTH2F(), bHD.histo_dus(), 1.0, 1.0, "b");
    }
    theHisto_c->getTH2F()->Divide(theHisto_c->getTH2F(), bHD.histo_c(), 1.0, 1.0, "b");
    theHisto_b->getTH2F()->Divide(theHisto_b->getTH2F(), bHD.histo_b(), 1.0, 1.0, "b");
    theHisto_ni->getTH2F()->Divide(theHisto_ni->getTH2F(), bHD.histo_ni(), 1.0, 1.0, "b");
    theHisto_dusg->getTH2F()->Divide(theHisto_dusg->getTH2F(), bHD.histo_dusg(), 1.0, 1.0, "b");
    theHisto_pu->getTH2F()->Divide(theHisto_pu->getTH2F(), bHD.histo_pu(), 1.0, 1.0, "b");
  }
}

template <class T, class G>
void FlavourHistograms2D<T, G>::setEfficiencyFlag() {
  if (theHisto_all)
    theHisto_all->setEfficiencyFlag();
  if (mcPlots_) {
    if (mcPlots_ > 2) {
      theHisto_d->setEfficiencyFlag();
      theHisto_u->setEfficiencyFlag();
      theHisto_s->setEfficiencyFlag();
      theHisto_g->setEfficiencyFlag();
      theHisto_dus->setEfficiencyFlag();
    }
    theHisto_c->setEfficiencyFlag();
    theHisto_b->setEfficiencyFlag();
    theHisto_ni->setEfficiencyFlag();
    theHisto_dusg->setEfficiencyFlag();
    theHisto_pu->setEfficiencyFlag();
  }
}

template <class T, class G>
void FlavourHistograms2D<T, G>::fillVariable(const int &flavour, const T &varX, const G &varY, const float &w) const {
  // all
  if (theHisto_all)
    theHisto_all->Fill(varX, varY, w);
  if (createProfile_)
    //if(theProfile_all) theProfile_all->Fill( varX, varY, w );
    if (theProfile_all)
      theProfile_all->Fill(varX, varY);

  //exit(-1);
  // flavour specific
  if (!mcPlots_)
    return;

  switch (flavour) {
    case 1:
      if (mcPlots_ > 2) {
        theHisto_d->Fill(varX, varY, w);
        theHisto_dus->Fill(varX, varY, w);
      }
      theHisto_dusg->Fill(varX, varY, w);
      if (createProfile_) {
        //theProfile_d->Fill(varX, varY,w);
        //theProfile_dus->Fill(varX, varY,w);
        //theProfile_dusg->Fill(varX, varY,w);
        if (mcPlots_ > 2) {
          theProfile_d->Fill(varX, varY);
          theProfile_dus->Fill(varX, varY);
        }
        theProfile_dusg->Fill(varX, varY);
      }
      return;
    case 2:
      if (mcPlots_ > 2) {
        theHisto_u->Fill(varX, varY, w);
        theHisto_dus->Fill(varX, varY, w);
      }
      theHisto_dusg->Fill(varX, varY, w);
      if (createProfile_) {
        //theProfile_u->Fill(varX, varY,w);
        //theProfile_dus->Fill(varX, varY,w);
        //theProfile_dusg->Fill(varX, varY,w);
        if (mcPlots_ > 2) {
          theProfile_u->Fill(varX, varY);
          theProfile_dus->Fill(varX, varY);
        }
        theProfile_dusg->Fill(varX, varY);
      }
      return;
    case 3:
      if (mcPlots_ > 2) {
        theHisto_s->Fill(varX, varY, w);
        theHisto_dus->Fill(varX, varY, w);
      }
      theHisto_dusg->Fill(varX, varY, w);
      if (createProfile_) {
        //theProfile_s->Fill(varX, varY,w);
        //theProfile_dus->Fill(varX, varY,w);
        //theProfile_dusg->Fill(varX, varY,w);
        if (mcPlots_ > 2) {
          theProfile_s->Fill(varX, varY);
          theProfile_dus->Fill(varX, varY);
        }
        theProfile_dusg->Fill(varX, varY);
      }
      return;
    case 4:
      theHisto_c->Fill(varX, varY, w);
      //if(createProfile_) theProfile_c->Fill(varX, varY,w);
      if (createProfile_)
        theProfile_c->Fill(varX, varY);
      return;
    case 5:
      theHisto_b->Fill(varX, varY, w);
      //if(createProfile_) theProfile_b->Fill(varX, varY,w);
      if (createProfile_)
        theProfile_b->Fill(varX, varY);
      return;
    case 21:
      if (mcPlots_ > 2)
        theHisto_g->Fill(varX, varY, w);
      theHisto_dusg->Fill(varX, varY, w);
      if (createProfile_) {
        //theProfile_g->Fill(varX, varY,w);
        //theProfile_dusg->Fill(varX, varY,w);
        if (mcPlots_ > 2)
          theProfile_g->Fill(varX, varY);
        theProfile_dusg->Fill(varX, varY);
      }
      return;
    case 20:
      theHisto_pu->Fill(varX, varY, w);
      //if(createProfile_) theProfile_pu->Fill(varX, varY,w);
      if (createProfile_)
        theProfile_pu->Fill(varX, varY);
      return;
    default:
      theHisto_ni->Fill(varX, varY, w);
      //if(createProfile_) theProfile_ni->Fill(varX, varY,w);
      if (createProfile_)
        theProfile_ni->Fill(varX, varY);
      return;
  }
}

template <class T, class G>
std::vector<TH2F *> FlavourHistograms2D<T, G>::getHistoVector() const {
  std::vector<TH2F *> histoVector;
  if (theHisto_all)
    histoVector.push_back(theHisto_all->getTH2F());
  if (mcPlots_) {
    if (mcPlots_ > 2) {
      histoVector.push_back(theHisto_d->getTH2F());
      histoVector.push_back(theHisto_u->getTH2F());
      histoVector.push_back(theHisto_s->getTH2F());
      histoVector.push_back(theHisto_g->getTH2F());
      histoVector.push_back(theHisto_dus->getTH2F());
    }
    histoVector.push_back(theHisto_c->getTH2F());
    histoVector.push_back(theHisto_b->getTH2F());
    histoVector.push_back(theHisto_ni->getTH2F());
    histoVector.push_back(theHisto_dusg->getTH2F());
    histoVector.push_back(theHisto_pu->getTH2F());
  }
  return histoVector;
}

template <class T, class G>
std::vector<TProfile *> FlavourHistograms2D<T, G>::getProfileVector() const {
  std::vector<TProfile *> profileVector;
  if (createProfile_) {
    if (theProfile_all)
      profileVector.push_back(theProfile_all->getTProfile());
    if (mcPlots_) {
      if (mcPlots_ > 2) {
        profileVector.push_back(theProfile_d->getTProfile());
        profileVector.push_back(theProfile_u->getTProfile());
        profileVector.push_back(theProfile_s->getTProfile());
        profileVector.push_back(theProfile_g->getTProfile());
        profileVector.push_back(theProfile_dus->getTProfile());
      }
      profileVector.push_back(theProfile_c->getTProfile());
      profileVector.push_back(theProfile_b->getTProfile());
      profileVector.push_back(theProfile_ni->getTProfile());
      profileVector.push_back(theProfile_dusg->getTProfile());
      profileVector.push_back(theProfile_pu->getTProfile());
    }
  }
  return profileVector;
}

#endif
