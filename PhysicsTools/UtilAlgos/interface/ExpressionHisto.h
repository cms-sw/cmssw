#ifndef PhysicsTools_UtilAlgos_ExpressionHisto_h
#define PhysicsTools_UtilAlgos_ExpressionHisto_h
// -*- C++ -*-
//
// Package:     UtilAlgos
// Class  :     ExpressionHisto
// 
/**\class ExpressionHisto ExpressionHisto.h PhysicsTools/UtilAlgos/interface/ExpressionHisto.h

 Description: Histogram tool using expressions 

 Usage:
    <usage>

*/
//
// Original Author: Benedikt HEGNER
//         Created:  Fri Jun  1 14:35:22 CEST 2007
// $Id: ExpressionHisto.h,v 1.3 2007/10/11 13:54:37 llista Exp $
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/Utilities/interface/StringObjectFunction.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH1.h"

template<typename T>
class ExpressionHisto {
public:
  ExpressionHisto(const edm::ParameterSet& iConfig);
  ~ExpressionHisto();
  
  void initialize(edm::Service<TFileService>& fs);
  void fill(const T& element);  
  
private:
  double min, max;
  int nbins;
  std::string name, description;
  TH1F * hist;
  StringObjectFunction<T> function;      
};

template<typename T>
ExpressionHisto<T>::ExpressionHisto(const edm::ParameterSet& iConfig):
  min(iConfig.template getUntrackedParameter<double>("min")),
  max(iConfig.template getUntrackedParameter<double>("max")),
  nbins(iConfig.template getUntrackedParameter<int>("nbins")),
  name(iConfig.template getUntrackedParameter<std::string>("name")),
  description(iConfig.template getUntrackedParameter<std::string>("description")),
  function(iConfig.template getUntrackedParameter<std::string>("plotquantity")) {
}

template<typename T>
ExpressionHisto<T>::~ExpressionHisto() {
}

template<typename T>
void ExpressionHisto<T>::initialize(edm::Service<TFileService>& fs) {
  hist = fs->template make<TH1F>(name.c_str(),description.c_str(),nbins,min,max);
}

template<typename T>
void ExpressionHisto<T>::fill(const T& element) {
  hist->Fill( function(element) );
}

#endif
