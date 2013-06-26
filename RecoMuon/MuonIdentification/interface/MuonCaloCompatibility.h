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
// $Id: MuonCaloCompatibility.h,v 1.4 2010/06/02 15:50:51 andersj Exp $
//
//

#include "TH2.h"
#include "TH2D.h"
#include "TFile.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

class MuonCaloCompatibility {
 public:
   MuonCaloCompatibility():isConfigured_(false){}
   void configure(const edm::ParameterSet&);
   double evaluate( const reco::Muon& );
 private:
   bool accessing_overflow( TH2D* histo, double x, double y );
   bool isConfigured_;
   
   /*    std::string muon_templateFileName; */
   /*    std::string pion_templateFileName; */
   std::string MuonfileName_;
   std::string PionfileName_;
   
   boost::shared_ptr<TFile> pion_templates;
   boost::shared_ptr<TFile> muon_templates;
   // used input templates for given eta
   TH2D * pion_template_em ;
   TH2D * pion_template_had;
   TH2D * pion_template_ho ;
   TH2D * muon_template_em ;
   TH2D * muon_template_had;
   TH2D * muon_template_ho ;
   // input template functions by eta
   TH2D* pion_had_etaEpl ;
   TH2D* pion_em_etaEpl  ;
   TH2D* pion_had_etaTpl ;
   TH2D* pion_em_etaTpl  ;
   TH2D* pion_ho_etaB    ;
   TH2D* pion_had_etaB   ;
   TH2D* pion_em_etaB    ;
   TH2D* pion_had_etaTmi ;
   TH2D* pion_em_etaTmi  ;
   TH2D* pion_had_etaEmi ;
   TH2D* pion_em_etaEmi  ;

   TH2D* muon_had_etaEpl ;
   TH2D* muon_em_etaEpl  ;
   TH2D* muon_had_etaTpl ;
   TH2D* muon_em_etaTpl  ;
   TH2D* muon_ho_etaB    ;
   TH2D* muon_had_etaB   ;
   TH2D* muon_em_etaB    ;
   TH2D* muon_had_etaTmi ;
   TH2D* muon_em_etaTmi  ;
   TH2D* muon_had_etaEmi ;
   TH2D* muon_em_etaEmi  ;

   double  pbx;
   double  pby;
   double  pbz;
   
   double  psx;
   double  psy;
   double  psz;

   double  muon_compatibility;

   bool use_corrected_hcal;
   bool use_em_special;
};
#endif
