//Emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-

/*
 * Author: Ph. Gras CEA/Saclay
 */

#include "EventFilter/EcalRawToDigi/interface/PGHisto.h"
#include "EventFilter/EcalRawToDigi/interface/printMacros.h"
#include "TH1C.h"
#include "TH1S.h"
#include "TH1I.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2.h"
#include "TH3C.h"
#include <sstream>

using namespace std;

void PGHisto::dumpList(ostream& out){
  for(histos_t::iterator it = histos.begin();
      it != histos.end();
      ++it){
    //    VASSERT(it->first,==,string(it->second->GetName()));
    out << it->second->GetName() << "\t"
        << it->second->GetTitle() << "\t";
    //note: order of below if conditions is important
    if(dynamic_cast<TH3*>(it->second)){//3D-histo
      out << "3D" << "\t";
    } else if(dynamic_cast<TH3*>(it->second)){//3D-histo
      out << "3D" << "\t";
    } else if(dynamic_cast<TProfile2D*>(it->second)){//2D-profule
      out << "2D P" << "\t";
    } else if(dynamic_cast<TH2*>(it->second)){//2D-histo
      out << "2D" << "\t";
    } else if(dynamic_cast<TProfile*>(it->second)){//1D profile
      out << "1D P" << "\t";
    } else if(dynamic_cast<TH1*>(it->second)){//1D profile
      out << "1D" << "\t";
    }
    out  << "\n";
  }
  out << flush;
}

template<class U>
Int_t PGHisto::fill(const char* name, U x, Stat_t w, TH1** pph){
  TH1* ph = NULL;
  if(pph==NULL) pph = &ph;
  if(*pph==NULL){
    histos_t::iterator it = histos.find(name);
    if(it==histos.end()){
      cerr << __func__ << ": Histo '" << name << "' does not exit!. It will not be filled";
      return 0;
    } else{
      *pph = it->second;
    }
  }
  //creates sumw2 structure on first weighted events:
  if(w!=1. && (*pph)->GetSumw2N()==0) (*pph)->Sumw2();
  return (*pph)->Fill(x,w);
}

Int_t PGHisto::fill(const char* name, Axis_t x, Axis_t y, Stat_t w, TH2** pph){
  TH2* ph = NULL;
  if(pph==NULL) pph = &ph;
  if(*pph==NULL){
    histos_t::iterator it = histos.find(name);
    if(it==histos.end()){
      cerr << __func__ << ": Histo '" << name << "' does not exit!. It will not be filled";
      return 0;
    } else{
      *pph = dynamic_cast<TH2*>(it->second);
      if(*pph==NULL){//histo is not a 2-D histos!
        cerr << __HERE__ << ": "<< __func__
             << ": " << "Error. Histogram '" << name
             << "' is not a 2-D histograms"
             << endl;
        return 0;
      }
    }
  }
  //creates sumw2 structure on first weighted events:
  if(w!=1. && (*pph)->GetSumw2N()==0) (*pph)->Sumw2();
  return (*pph)->Fill(x, y, w);
}

TH1* PGHisto::operator[](const char* name){
  //in principle the histo must already exist
  pair<histos_t::iterator, bool> insertResult =
    histos.insert(make_pair(name, (TH1*)0));
  if(insertResult.second){//histos did not exit. This is anormal
    cerr << __HERE__ << ": " << __func__ << ": Error: histogram " << name
         << " does not exist!" << endl;
    stringstream buffer;
    buffer << "*error* This histogram was created because PGHisto::operator[]"
      " was called with a wrong name";
    insertResult.first->second = new TH3C(name, buffer.str().c_str(),
                                          0,0,0,0,0,0,0,0,0);
  }
  return insertResult.first->second;
}

Int_t PGHisto::fillProfile(int ctorId, const char* name, const char* title,
                           Int_t nbinsx, Axis_t xlow, Axis_t xup,
                           const Double_t* xbins,
                           Axis_t ylow, Axis_t yup,
                           Option_t* option, Double_t x, Double_t y,
                           Stat_t w,
                           TProfile** pph){
  TProfile* ph = 0;
  if(pph==0) pph = &ph;
  if(*pph==0){
    //create histogram if does not exist yet:
    std::pair<histos_t::iterator, bool> insertResult =
      histos.insert(std::make_pair(std::string(name), (TH1*)0));
    if(insertResult.second){//new histos
      rootFile.cd();
      switch(ctorId){
      case 0:
        *pph = new TProfile(name, title, nbinsx, xlow, xup, option);
        break;
      case 1:
        *pph = new TProfile(name, title, nbinsx, xlow, xup, ylow, yup,
                            option);
        break;
      case 2:
        *pph = new TProfile(name, title, nbinsx, xbins,
                            option);
        break;
      case 3:
        *pph = new TProfile(name, title, nbinsx, xbins, ylow, yup,
                            option);
        break;
      default:
        assert(false);
      }
      insertResult.first->second = *pph;
      (*pph)->SetDirectory(&rootFile);
    } else{
      *pph = dynamic_cast<TProfile*>(insertResult.first->second);
      if(*pph==0){
        std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__
                  << ": Error: " << name << " is not a 1-D profile!"
                  << std::endl;
        return 0;
      }
    }
  }
  //creates sumw2 structure on first weighted events:
  if(w!=1. && (*pph)->GetSumw2N()==0) (*pph)->Sumw2();
  //fills histogram:
  return (*pph)->Fill(x, y, w);
}

Int_t PGHisto::fillProfile2D(int ctorId, const char* name, const char* title,
                             Int_t nbinsx, Axis_t xlow, Axis_t xup,
                             const Double_t* xbins,
                             Int_t nbinsy, Axis_t ylow, Axis_t yup,
                             const Double_t* ybins,
                             Axis_t zlow, Axis_t zup,
                             Option_t* option, Double_t x, Double_t y,
                             Double_t z,
                             Stat_t w, TProfile2D** pph){
  TProfile2D* ph = NULL;
  if(pph==NULL) pph = &ph;
  if(*pph==0){
    //create histogram if does not exist yet:
    std::pair<histos_t::iterator, bool> insertResult =
      histos.insert(std::make_pair(std::string(name), (TH1*)0));
    if(insertResult.second){//new histos
      rootFile.cd();
      switch(ctorId){
      case 0:
        *pph = new TProfile2D(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup, option);
        break;
      case 1:
        *pph = new TProfile2D(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup, zlow, zup,
                              option);
        break;
      case 2:
        *pph = new TProfile2D(name, title, nbinsx, xbins, nbinsy, ylow, yup,
                              option);
        break;
      case 3:
        *pph = new TProfile2D(name, title, nbinsx, xlow, xup, nbinsy, ybins,
                              option);
        break;
      case 4:
        *pph = new TProfile2D(name, title, nbinsx, xbins, nbinsy, ybins,
                              option);
        break;
      default:
        assert(false);
      }
      (*pph)->SetDirectory(&rootFile);
      insertResult.first->second = *pph;
    } else{
      *pph = dynamic_cast<TProfile2D*>(insertResult.first->second);
      if(*pph==0){
        std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__
                  << ": Error: " << name << " is not a 2-D profile!"
                  << std::endl;
        return 0;
      }
    }
  }
  //fills histogram:
  return (*pph)->Fill(x, y, z, w);
}

void PGHisto::release(const char* name){
  histos_t::iterator it = histos.find(name);
  if(it==histos.end()){
    cerr << __func__ << ": Histo '" << name << "' does not exit!. It will not be released";
    return;
  } else{
    //    TH1* h = it->second;
    rootFile.Delete(name); //delete from memory
    delete it->second;
    histos.erase(it);
  }
}

//original code from ROOT TH1::LabelsDeflate
//extended to non-label histogram
void PGHisto::axisDeflate(TH1* h, int nbins, char ax){
  TAxis *axis = 0;
  if (ax == 'X'){
    axis = h->GetXaxis();
  } else if (ax == 'Y'){
    axis = h->GetYaxis();
  } else if (ax == 'Z'){
    axis = h->GetZaxis();
  } else return;
  if(nbins < 1) nbins = 1;
  if(nbins > axis->GetNbins()) return;

  TH1 *hold = (TH1*)h->Clone();
  hold->SetDirectory(0);

  Bool_t timedisp = axis->GetTimeDisplay();
  Double_t xmin = axis->GetXmin();
  Double_t xmax = axis->GetBinUpEdge(nbins);
  if (xmax <= xmin) xmax = xmin +nbins;
  axis->SetRange(0,0);
  axis->Set(nbins,xmin,xmax);
  Int_t  nbinsx = hold->GetXaxis()->GetNbins();
  Int_t  nbinsy = hold->GetYaxis()->GetNbins();
  Int_t  nbinsz = hold->GetZaxis()->GetNbins();
  Int_t ncells = nbinsx+2;
  if (h->GetDimension() > 1) ncells *= nbinsy+2;
  if (h->GetDimension() > 2) ncells *= nbinsz+2;
  h->SetBinsLength(ncells);
  TArrayD& fSumw2 = *(h->GetSumw2());
  Int_t errors = fSumw2.GetSize();
  if (errors) fSumw2.Set(ncells);
  axis->SetTimeDisplay(timedisp);

  //now loop on all bins and refill
  Double_t err,cu;
  Int_t bin,ibin,binx,biny,binz;
  Double_t oldEntries = h->GetEntries();
  for (binz=1;binz<=nbinsz;binz++) {
    for (biny=1;biny<=nbinsy;biny++) {
      for (binx=1;binx<=nbinsx;binx++) {
        bin = hold->GetBin(binx,biny,binz);
        ibin= h->GetBin(binx,biny,binz);
        cu  = hold->GetBinContent(bin);
        h->SetBinContent(ibin,cu);
        if (errors) {
          err = hold->GetBinError(bin);
          h->SetBinError(ibin,err);
        }
      }
    }
  }
  h->SetEntries(oldEntries);
  delete hold;
}
