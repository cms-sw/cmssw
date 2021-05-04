#ifndef MuonIdentification_MuonCaloCompatibility_h
#define MuonIdentification_MuonCaloCompatibility_h

// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonCaloCompatibility
//
/*

 Description: test track muon hypothesis using energy deposition in ECAL,HCAL,HO

*/
//
// Original Author:  Ingo Bloch
//
//

#include "TH2.h"
#include "TH2D.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class MuonCaloCompatibility {
public:
  MuonCaloCompatibility() : isConfigured_(false) {}
  void configure(const edm::ParameterSet&);
  double evaluate(const reco::Muon&);

private:
  bool accessing_overflow(const TH2D& histo, double x, double y);
  bool isConfigured_;

  // used input templates for given eta
  std::shared_ptr<TH2D> pion_template_em;
  std::shared_ptr<TH2D> pion_template_had;
  std::shared_ptr<TH2D> pion_template_ho;
  std::shared_ptr<TH2D> muon_template_em;
  std::shared_ptr<TH2D> muon_template_had;
  std::shared_ptr<TH2D> muon_template_ho;
  // input template functions by eta
  std::shared_ptr<TH2D> pion_had_etaEpl;
  std::shared_ptr<TH2D> pion_em_etaEpl;
  std::shared_ptr<TH2D> pion_had_etaTpl;
  std::shared_ptr<TH2D> pion_em_etaTpl;
  std::shared_ptr<TH2D> pion_ho_etaB;
  std::shared_ptr<TH2D> pion_had_etaB;
  std::shared_ptr<TH2D> pion_em_etaB;
  std::shared_ptr<TH2D> pion_had_etaTmi;
  std::shared_ptr<TH2D> pion_em_etaTmi;
  std::shared_ptr<TH2D> pion_had_etaEmi;
  std::shared_ptr<TH2D> pion_em_etaEmi;

  std::shared_ptr<TH2D> muon_had_etaEpl;
  std::shared_ptr<TH2D> muon_em_etaEpl;
  std::shared_ptr<TH2D> muon_had_etaTpl;
  std::shared_ptr<TH2D> muon_em_etaTpl;
  std::shared_ptr<TH2D> muon_ho_etaB;
  std::shared_ptr<TH2D> muon_had_etaB;
  std::shared_ptr<TH2D> muon_em_etaB;
  std::shared_ptr<TH2D> muon_had_etaTmi;
  std::shared_ptr<TH2D> muon_em_etaTmi;
  std::shared_ptr<TH2D> muon_had_etaEmi;
  std::shared_ptr<TH2D> muon_em_etaEmi;

  double pbx;
  double pby;
  double pbz;

  double psx;
  double psy;
  double psz;

  double muon_compatibility;

  bool use_corrected_hcal;
  bool use_em_special;
};
#endif
