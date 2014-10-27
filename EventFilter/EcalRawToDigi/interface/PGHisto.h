//Emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/* Utility for histogramming using ROOT.
 * The particularity of this utility is that booking and filling histograms
 * can be done in a single line of code.
 *
 * Author: Ph. Gras CEA/Saclay
 */

//TODO: the histos map might be removed since ROOT already maintain an object
//registry=> use of FindObject(const char*)

#ifndef PGHISTO_H
#define PGHISTO_H

#include "EventFilter/EcalRawToDigi/interface/printMacros.h"
#include "TH1.h"
#include "TH1C.h"
#include "TH1S.h"
#include "TH1I.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2.h"
#include "TH2C.h"
#include "TH2S.h"
#include "TH2I.h"
#include "TH2F.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TProfile2D.h"

#include <TFile.h>
#include <map>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>

//TODO: use this class to simplify PGHisto method argument lists
class PGAxis{
public:
  PGAxis(int nbins, double min, double max){
    this->nbins = nbins;
    this->min = min;
    this->max = max;
  }
  int nbins;
  double min;
  double max;
};
typedef PGAxis PGXAxis;
typedef PGAxis PGYAxis;

/** A convenient class for histogramming.
 */
class PGHisto {
  //type definitions
  typedef std::map<std::string, TH1*> histos_t;

  //attribute(s)
protected:
private:
  histos_t histos;
  /** Used when file opened by ctor. Elsewhere rootFile which will refer
   * to the ROOT file where to store the histogram (either rootFile_ or another
   * file depending of the called ctor).
   */
  TFile rootFile_;
  TFile& rootFile;

  //constructor(s) and destructor(s)
public:
  /** Constructs a PGHisto
   * @param filename name of the ROOT file to store the histograms
   * @param openmode ROOT file openmode: "CREATE" or "UPDATE"
   */
  PGHisto(const char* filename, const char* openmode = "RECREATE"):
    rootFile_(filename, openmode), rootFile(rootFile_){};

  /** Constructs a PGHisto
   * @param file ROOT file to store the histograms
   */
  PGHisto(TFile& file): rootFile(file){};

  ~PGHisto(){
    std::cout << "Writing hitos on file." << std::endl;
    rootFile.cd();
    rootFile.Write();
  }

  /** Shrink axis of an histogram. The 'nbins' first bins
   * are kept.
   * @param h histogram to shrink
   * @param nbins number of bins to keep
   * @param axis axis to shrink: 'X', 'Y' or 'Z'
   */
  static void axisDeflate(TH1* h, int nbins, char axis = 'X');

private:
  PGHisto(): rootFile(rootFile_){};

  //method(s)
private:
  template<class T, class U>
  Int_t fill(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, U x, Stat_t w = 1., TH1** h = NULL);
  template<class T>
  Int_t fill(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup,Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL);
  Int_t fillProfile(int ctorId, const char* name, const char* title,
                    Int_t nbinsx, Axis_t xlow, Axis_t xup,
                    const Double_t* xbins,
                    Axis_t ylow, Axis_t yup,
                    Option_t* option, Double_t x, Double_t y,
                    Stat_t w, TProfile** pph);
  Int_t fillProfile2D(int ctorId, const char* name, const char* title,
                      Int_t nbinsx, Axis_t xlow, Axis_t xup,
                      const Double_t* xbins,
                      Int_t nbinsy, Axis_t ylow, Axis_t yup,
                      const Double_t* ybins,
                      Axis_t zlow, Axis_t zup,
                      Option_t* option, Double_t x, Double_t y, Double_t z,
                      Stat_t w, TProfile2D** pph);

public:
  /** Dump list of histograms
   * @param out stream, where to dump the list (cout by default)
   */
  void dumpList(std::ostream& out);

  //@{ Histogram getter. Creates histogram at the first call.
  template<class T>
  TH1* get(const char* name, const char* title, PGXAxis xAxis, TH1** pph = 0);
  //@}

  //@{
  /** Filling methods for 1-D histos:
   * with full histo definition
   */
  template<class T>
  Int_t fillC(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, T x, Stat_t w = 1., TH1** pph = NULL){
    return fill<Char_t>(name, title, nbinsx, xlow, xup, x, w);
  }
  template<class T>
  Int_t fillC(const char* name, const char* title, const PGAxis& xAxis, T x, Stat_t w = 1., TH1** pph = NULL){
    return fillC(name, title,  xAxis.nbins, xAxis.min, xAxis.max, x, w, pph);
  }
  template<class T>
  Int_t fillS(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, T x, Stat_t w = 1., TH1** pph = NULL){
    return fill<Short_t>(name, title, nbinsx, xlow, xup, x, w);
  }
  template<class T>
  Int_t fillS(const char* name, const char* title, const PGAxis& xAxis, T x, Stat_t w = 1., TH1** pph = NULL){
    return fillS(name, title, xAxis.nbins, xAxis.min, xAxis.max, x, w, pph);
  }
  template<class T>
  Int_t fillI(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, T x, Stat_t w = 1., TH1** pph = NULL){
    return fill<Int_t>(name, title, nbinsx, xlow, xup, x, w);
  }
  template<class T>
  Int_t fillI(const char* name, const char* title, PGAxis xAxis, T x, Stat_t w = 1., TH1** pph = NULL){
    return fillI(name, title, xAxis.nbins, xAxis.min, xAxis.max, x, w, pph);
  }
  template<class T>
  Int_t fillF(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, T x, Stat_t w = 1., TH1** pph = NULL){
    return fill<Float_t>(name, title, nbinsx, xlow, xup, x, w);
  }
  template<class T>
  Int_t fillF(const char* name, const char* title, const PGAxis& xAxis, T x, Stat_t w = 1., TH1** pph = NULL){
    return fillF(name, title, xAxis.nbins, xAxis.min, xAxis.max, x, w, pph);
  }
  template<class T>
  Int_t fillD(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, T x, Stat_t w = 1., TH1** pph = NULL){
    return fill<Double_t>(name, title, nbinsx, xlow, xup, x, w);
  }
  template<class T>
  Int_t fillD(const char* name, const char* title, const PGAxis& xAxis, T x, Stat_t w = 1., TH1** pph = NULL){
    return fillD(name, title, xAxis.nbins, xAxis.min, xAxis.max, x, w, pph);
  }
  // with histo name only
  template<class T>
  Int_t fill(const char* name, T x, Stat_t w = 1., TH1** pph = NULL);
  //@}

  //@{
  /** Filling methods for 2-D hitos
   * with full histo definition
   */
  Int_t fillC(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fill<Char_t>(name, title, nbinsx, xlow, xup,
                        nbinsy, ylow, yup, x, y, w, pph);
  }
  Int_t fillC(const char* name, const char* title, const PGAxis& xAxis, const PGAxis& yAxis, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fillC(name, title, xAxis.nbins, xAxis.min, xAxis.max, yAxis.nbins, yAxis.min, yAxis.max, x, y, w, pph);
  }
  Int_t fillS(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fill<Short_t>(name, title, nbinsx, xlow, xup,
                         nbinsy, ylow, yup, x, y, w, pph);
  }
  Int_t fillS(const char* name, const char* title, const PGAxis& xAxis, PGAxis yAxis, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fillS(name, title, xAxis.nbins, xAxis.min, xAxis.max, yAxis.nbins, yAxis.min, yAxis.max, x, y, w, pph);
  }
  Int_t fillI(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fill<Int_t>(name, title, nbinsx, xlow, xup,
                       nbinsy, ylow, yup, x, y, w, pph);
  }
  Int_t fillI(const char* name, const char* title, const PGAxis& xAxis, const PGAxis& yAxis, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fillI(name, title, xAxis.nbins, xAxis.min, xAxis.max, yAxis.nbins, yAxis.min, yAxis.max, x, y, w, pph);
  }
  Int_t fillF(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fill<Float_t>(name, title, nbinsx, xlow, xup,
                         nbinsy, ylow, yup, x, y, w, pph);
  }
  Int_t fillF(const char* name, const char* title, const PGAxis& xAxis, const PGAxis& yAxis, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fillF(name, title, xAxis.nbins, xAxis.min, xAxis.max, yAxis.nbins, yAxis.min, yAxis.max, x, y, w, pph);
  }
  Int_t fillD(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL){
    return fill<Double_t>(name, title, nbinsx, xlow, xup,
                          nbinsy, ylow, yup, x, y, w, pph);
  }
  Int_t fillD(const char* name, const char* title, const PGAxis& xAxis,
              const PGAxis& yAxis, Axis_t x, Axis_t y,
              Stat_t w = 1., TH2** pph = NULL){
    return fillD(name, title, xAxis.nbins, xAxis.min, xAxis.max, yAxis.nbins, yAxis.min, yAxis.max, x, y, w, pph);
  }
  // with histo name only
  Int_t fill(const char* name, Axis_t x, Axis_t y, Stat_t w = 1., TH2** pph = NULL);
  Int_t fill(const char* name, Axis_t x[2], Stat_t w = 1., TH2** pph = NULL){
    return fill(name, x[0], x[1], w, pph);
  }
  Int_t fill(const char* name, std::vector<Axis_t> x, Stat_t w = 1.,
             TH2** pph = NULL){
    return fill(name, x[0], x[1], w, pph);
  }
  //@}

  //@{
  /** Filling methods for 1D profiles
   */
  Int_t fillProfile(const char* name, const char* title,
                    Int_t nbinsx, Axis_t xlow, Axis_t xup,
                    Option_t* option, Double_t x, Double_t y,
                    Stat_t w=1., TProfile** pph = 0){
    return  fillProfile(0, name, title, nbinsx, xlow, xup, (Double_t*) 0,
                        0., 0., option, x, y, w, pph);
  }
  Int_t fillProfile(const char* name, const char* title,
                    const PGAxis& xAxis,
                    Option_t* option, Double_t x, Double_t y,
                    Stat_t w=1., TProfile** pph = 0){
    return  fillProfile(name, title,
                        xAxis.nbins, xAxis.min, xAxis.max,
                        option,x,y,
                        w, pph);
  }
  Int_t fillProfile(const char* name, const char* title,
                    Int_t nbinsx, Axis_t xlow, Axis_t xup,
                    Axis_t ylow, Axis_t yup, Option_t* option,
                    Double_t x, Double_t y, Stat_t w=1.,
                    TProfile** pph = 0){
    return fillProfile(1, name, title, nbinsx, xlow, xup, (Double_t*) 0,
                       ylow, yup, option, x, y, w, pph);
  }
  Int_t fillProfile(const char* name, const char* title,
                    const PGAxis& xAxis,
                    Axis_t ylow, Axis_t yup, Option_t* option,
                    Double_t x, Double_t y, Stat_t w=1., TProfile** pph = 0){
    return fillProfile(name, title,
                       xAxis.nbins, xAxis.min, xAxis.max,
                       ylow, yup, option,
                       x, y, 1., pph);
  }
  Int_t fillProfile(const char* name, const char* title,
                    Int_t nbinsx, const Double_t* xbins,
                    Option_t* option, Double_t x, Double_t y,
                    Stat_t w=1., TProfile** pph = 0){
    return fillProfile(2, name, title, nbinsx, 0., 0., xbins, 0., 0.,
                       option, x, y, w, pph);
  }
  Int_t fillProfile(const char* name, const char* title,
                    Int_t nbinsx, const Double_t* xbins,
                    Axis_t ylow, Axis_t yup, Option_t* option,
                    Double_t x, Double_t y,
                    Stat_t w=1., TProfile** pph = 0){
    return fillProfile(3, name, title, nbinsx, 0., 0., xbins, ylow, yup,
                       option, x, y, w, pph);
  }
  //@}

  //@{
  /** Filling methods for 2D profiles
   */
  Int_t fillProfile2D(const char* name, const char* title,
                      Int_t nbinsx, Axis_t xlow, Axis_t xup,
                      Int_t nbinsy, Axis_t ylow, Axis_t yup,
                      Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return  fillProfile2D(0, name, title, nbinsx, xlow, xup, (Double_t*) 0,
                          nbinsy, ylow, yup, (Double_t*) 0,
                          0., 0., option, x, y, z, w, pph);
  }
  Int_t fillProfile2D(const char* name, const char* title,
                      const PGAxis& xAxis,
                      const PGAxis& yAxis,
                      Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return  fillProfile2D(name, title,
                          xAxis.nbins, xAxis.min, xAxis.max,
                          yAxis.nbins, yAxis.min, yAxis.max,
                          option,
                          x, y, z, w, pph);
  }
  Int_t fillProfile2D(const char* name, const char* title,
                      Int_t nbinsx, Axis_t xlow, Axis_t xup,
                      Int_t nbinsy, Axis_t ylow, Axis_t yup,
                      Axis_t zlow, Axis_t zup, Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return fillProfile2D(1, name, title, nbinsx, xlow, xup, (Double_t*) 0,
                         nbinsy, ylow, yup, (Double_t*) 0,
                         zlow, zup, option, x, y, z, w, pph);
  }
  Int_t fillProfile2D(const char* name, const char* title,
                      const PGAxis& xAxis,
                      const PGAxis& yAxis,
                      Axis_t zlow, Axis_t zup, Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return fillProfile2D(name, title,
                         xAxis.nbins, xAxis.min, xAxis.max,
                         yAxis.nbins, yAxis.min, yAxis.max,
                         zlow, zup, option,
                         x, y, z, w, pph);
  }
  Int_t fillProfile2D(const char* name, const char* title,
                      Int_t nbinsx, const Double_t* xbins,
                      Int_t nbinsy, Axis_t ylow, Axis_t yup,
                      Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return fillProfile2D(2, name, title, nbinsx, 0., 0., xbins,
                         nbinsy, ylow, yup, (Double_t*) 0,
                         0., 0., option, x, y, z, w, pph);
  }
  Int_t fillProfile2D(const char* name, const char* title,
                      Int_t nbinsx, const Double_t* xbins,
                      const PGAxis& yAxis,
                      Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return fillProfile2D(name, title,
                         nbinsx, xbins,
                         yAxis.nbins, yAxis.min, yAxis.max,
                         option,
                         x, y, z, w, pph);
  }
  Int_t fillProfile2D(const char* name, const char* title,
                      Int_t nbinsx, Double_t xlow, Double_t xup,
                      Int_t nbinsy, const Double_t* ybins,
                      Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return fillProfile2D(3, name, title, nbinsx, xlow, xup, (Double_t*) 0,
                         nbinsy, 0., 0., ybins,
                         0., 0., option, x, y, z, w, pph);
  }
  Int_t fillProfile2D(const char* name, const char* title,
                      const PGAxis& xAxis,
                      Int_t nbinsy, const Double_t* ybins,
                      Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return fillProfile2D(name, title,
                         xAxis.nbins, xAxis.min, xAxis.max,
                         nbinsy, ybins,
                         option,
                         x, y, z, w, pph);
  }
  Int_t fillProfile2D(const char* name, const char* title,
                      Int_t nbinsx, const Double_t* xbins,
                      Int_t nbinsy, const Double_t* ybins,
                      Option_t* option,
                      Double_t x, Double_t y, Double_t z, Stat_t w=1.,
                      TProfile2D** pph = 0){
    return fillProfile2D(4, name, title, nbinsx, 0., 0., xbins,
                         nbinsy, 0., 0., ybins,
                         0., 0., option, x, y, z, w, pph);
  }
  //@}

  /** Gets an histo from its name. If the histogram does not
   * exits an error message is displayed and a default histo
   * is created.
   * @param name histogram name
   * @return pointer to the histogram.
   */
  TH1* operator[](const char* name);

  /** Gets an histos from its name. If the histogram does not exist,
   *  a NULL pointer is returned
   *   @param name histogram name
   *   @return pointer to the histogram or NULL
   */
  TH1* find(const char* name){
    histos_t::iterator it = histos.find(name);
    return (it==histos.end())?NULL:it->second;
  }

  /** Returns a reference to the root file where hitograms are written to.
   * @return reference to the TFile
   */
  TFile& getRootFile(){ return rootFile;}

  /** Remove a histogram from memory. The histogram will be store into the
   * file.
   */
  void release(const char* name);

private:
};

template<class T, class U>
Int_t PGHisto::fill(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, U x, Stat_t w, TH1** pph){
  TH1* h = get<T>(name, title, PGXAxis(nbinsx, xlow, xup), pph);
  return h==0?0:h->Fill(x,w);
}

template<class T>
Int_t PGHisto::fill(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup,Axis_t x, Axis_t y, Stat_t w, TH2** pph){
  TH2* ph = NULL;
  if(pph==NULL) pph = &ph;
  if(*pph==NULL){
    //create histogram if does not exist yet:
    std::pair<histos_t::iterator, bool> insertResult =
      histos.insert(std::make_pair(name, (TH1*)0));
    if(insertResult.second){//new histos
      rootFile.cd();
      if(typeid(T)==typeid(Char_t)){
        *pph = new TH2C(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
      } else if(typeid(T)==typeid(Short_t)){
        *pph = new TH2S(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
      } else if(typeid(T)==typeid(Int_t)){
        *pph = new TH2I(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
      } else if(typeid(T)==typeid(Float_t)){
        *pph = new TH2F(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
      } else if(typeid(T)==typeid(Double_t)){
        *pph = new TH2D(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
      } else{
        assert(false);
      }
      (*pph)->SetDirectory(&rootFile);
      insertResult.first->second = *pph;
    } else{
      *pph = dynamic_cast<TH2*>(insertResult.first->second);
      if(*pph==0){
        std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__
                  << ": Error: " << name << " is not a 2-D histogram!"
                  << std::endl;
        return 0;
      }
    }
  }
  //fills histogram
  return (*pph)->Fill(x,y,w);
}

template Int_t PGHisto::fill<Char_t, Double_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w, TH1** pph);
template Int_t PGHisto::fill<Short_t, Double_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w, TH1** pph);
template Int_t PGHisto::fill<Int_t, Double_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w, TH1** pph);
template Int_t PGHisto::fill<Float_t, Double_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w, TH1** pph);
template Int_t PGHisto::fill<Double_t, Double_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w, TH1** pph);

template Int_t PGHisto::fill<Char_t, const char*>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w, TH1** pph);
template Int_t PGHisto::fill<Short_t, const char*>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w, TH1** pph);
template Int_t PGHisto::fill<Int_t, const char*>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w, TH1** pph);
template Int_t PGHisto::fill<Float_t, const char*>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w, TH1** pph);
template Int_t PGHisto::fill<Double_t, const char*>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w, TH1** pph);


template Int_t PGHisto::fill<Char_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup,Axis_t x, Axis_t y, Stat_t w, TH2** pph);
template Int_t PGHisto::fill<Short_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup,Axis_t x, Axis_t y, Stat_t w, TH2** pph);
template Int_t PGHisto::fill<Int_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup,Axis_t x, Axis_t y, Stat_t w, TH2** pph);
template Int_t PGHisto::fill<Float_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup,Axis_t x, Axis_t y, Stat_t w, TH2** pph);
template Int_t PGHisto::fill<Double_t>(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup,Axis_t x, Axis_t y, Stat_t w, TH2** pph);

//--

//filling methods for 1-D histos:
// with full histo definition
template
Int_t PGHisto::fillC(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillC(const char* name, const char* title, const PGAxis& xAxis, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillS(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillS(const char* name, const char* title, const PGAxis& xAxis, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillI(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillI(const char* name, const char* title, PGAxis xAxis, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillF(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillF(const char* name, const char* title, const PGAxis& xAxis, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillD(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, const char* x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillD(const char* name, const char* title, const PGAxis& xAxis, const char* x, Stat_t w = 1., TH1** pph = NULL);

template
Int_t PGHisto::fillC(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillC(const char* name, const char* title, const PGAxis& xAxis, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillS(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillS(const char* name, const char* title, const PGAxis& xAxis, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillI(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillI(const char* name, const char* title, PGAxis xAxis, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillF(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillF(const char* name, const char* title, const PGAxis& xAxis, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillD(const char* name, const char* title, Int_t nbinsx, Axis_t xlow, Axis_t xup, Double_t x, Stat_t w = 1., TH1** pph = NULL);
template
Int_t PGHisto::fillD(const char* name, const char* title, const PGAxis& xAxis, Double_t x, Stat_t w = 1., TH1** pph = NULL);


template<class T>
TH1* PGHisto::get(const char* name, const char* title, PGXAxis xa,
                  TH1** pph){
  TH1* ph = NULL;
  if(pph==NULL) pph=&ph;
  if(*pph==NULL){
    //create histogram if does not exist yet:
    std::pair<histos_t::iterator, bool> insertResult =
      histos.insert(std::make_pair(std::string(name), (TH1*)0));
    if(insertResult.second){//new histos
      rootFile.cd();
      if(typeid(T)==typeid(Char_t)){
        *pph = new TH1C(name, title, xa.nbins, xa.min, xa.max);
      } else if(typeid(T)==typeid(Short_t)){
        *pph = new TH1S(name, title, xa.nbins, xa.min, xa.max);
      } else if(typeid(T)==typeid(Int_t)){
        *pph = new TH1I(name, title, xa.nbins, xa.min, xa.max);
      } else if(typeid(T)==typeid(Float_t)){
        *pph = new TH1F(name, title, xa.nbins, xa.min, xa.max);
      } else if(typeid(T)==typeid(Double_t)){
        *pph = new TH1D(name, title, xa.nbins, xa.min, xa.max);
      } else{
        assert(false);
      }
      (*pph)->SetDirectory(&rootFile);
      insertResult.first->second = *pph;
    } else{
      *pph = insertResult.first->second;
    }
  }
  return *pph;
}

#endif
