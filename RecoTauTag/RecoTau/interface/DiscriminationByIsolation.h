#ifndef DiscriminationByIsolation_H_
#define DiscriminationByIsolation_H_

/* class DiscriminationByIsolation
 * created : Jul 23 2007,
 * revised :
 * contributors : Ludovic Houchu (IPHC, Strasbourg), Christian Veelken (UC Davis)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/Tau.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorByIsolation.h"

#include "RecoTauTag/TauTagTools/interface/PFTauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"

#include "TFormula.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class DiscriminationByIsolation : public EDProducer {
 public:
  explicit DiscriminationByIsolation(const ParameterSet& iConfig){   
    TauProducer_                           = iConfig.getParameter<string>("TauProducer");
    // following parameters are considered when DiscriminationByIsolation EDProducer runs on Tau objects built from either CaloJet or PFJet objects *BEGIN*  
    UsePVconstraint_                       = iConfig.getParameter<bool>("UsePVconstraint");
    MatchingConeMetric_                    = iConfig.getParameter<string>("MatchingConeMetric");
    MatchingConeSizeFormula_               = iConfig.getParameter<string>("MatchingConeSizeFormula");
    MatchingConeVariableSize_min_          = iConfig.getParameter<double>("MatchingConeVariableSize_min");
    MatchingConeVariableSize_max_          = iConfig.getParameter<double>("MatchingConeVariableSize_max");
    ApplyDiscriminationByTrackerIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByTrackerIsolation");
    TrackerSignalConeMetric_               = iConfig.getParameter<string>("TrackerSignalConeMetric");
    TrackerSignalConeSizeFormula_          = iConfig.getParameter<string>("TrackerSignalConeSizeFormula");
    TrackerSignalConeVariableSize_max_     = iConfig.getParameter<double>("TrackerSignalConeVariableSize_max");
    TrackerSignalConeVariableSize_min_     = iConfig.getParameter<double>("TrackerSignalConeVariableSize_min");
    TrackerIsolConeMetric_                 = iConfig.getParameter<string>("TrackerIsolConeMetric"); 
    TrackerIsolConeSizeFormula_            = iConfig.getParameter<string>("TrackerIsolConeSizeFormula"); 
    TrackerIsolConeVariableSize_max_       = iConfig.getParameter<double>("TrackerIsolConeVariableSize_max");
    TrackerIsolConeVariableSize_min_       = iConfig.getParameter<double>("TrackerIsolConeVariableSize_min");
    // *END*   
    
    // following parameters are considered when DiscriminationByIsolation EDProducer runs on Tau objects built from PFJet objects *BEGIN* 
    ManipulateTracks_insteadofChargedHadrCands_ = iConfig.getParameter<bool>("ManipulateTracks_insteadofChargedHadrCands");
    UseOnlyChargedHadr_for_LeadCand_    = iConfig.getParameter<bool>("UseOnlyChargedHadr_for_LeadCand"); 
    LeadCand_minPt_                     = iConfig.getParameter<double>("LeadCand_minPt"); 
    ChargedHadrCand_minPt_              = iConfig.getParameter<double>("ChargedHadrCand_minPt");
    TrackerIsolAnnulus_Candsmaxn_       = iConfig.getParameter<int>("TrackerIsolAnnulus_Candsmaxn");       
    ApplyDiscriminationByECALIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByECALIsolation");
    GammaCand_minPt_                    = iConfig.getParameter<double>("GammaCand_minPt");       
    ECALSignalConeMetric_               = iConfig.getParameter<string>("ECALSignalConeMetric");
    ECALSignalConeSizeFormula_          = iConfig.getParameter<string>("ECALSignalConeSizeFormula");    
    ECALSignalConeVariableSize_max_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_max");
    ECALSignalConeVariableSize_min_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_min");
    ECALIsolConeMetric_                 = iConfig.getParameter<string>("ECALIsolConeMetric");
    ECALIsolConeSizeFormula_            = iConfig.getParameter<string>("ECALIsolConeSizeFormula");      
    ECALIsolConeVariableSize_max_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_max");
    ECALIsolConeVariableSize_min_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_min");
    ECALIsolAnnulus_Candsmaxn_          = iConfig.getParameter<int>("ECALIsolAnnulus_Candsmaxn");
    // *END*   
    
    // following parameters are considered when
    // DiscriminationByIsolation EDProducer runs on Tau objects built from CaloJet objects 
    // OR (when DiscriminationByIsolation EDProducer runs on Tau objects built from PFJet objects AND ManipulateTracks_insteadofChargedHadrCands paremeter is set true)
    // *BEGIN*
    LeadTrack_minPt_                    = iConfig.getParameter<double>("LeadTrack_minPt");
    Track_minPt_                        = iConfig.getParameter<double>("Track_minPt");
    TrackerIsolAnnulus_Tracksmaxn_      = iConfig.getParameter<int>("TrackerIsolAnnulus_Tracksmaxn");   
    // *END* 
   
    AreaMetric_recoElements_maxabsEta_  = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");
    
    MatchingConeSizeTFormula_=computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
    TrackerSignalConeSizeTFormula_=computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
    TrackerIsolConeSizeTFormula_=computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
    ECALSignalConeSizeTFormula_=computeConeSizeTFormula(ECALSignalConeSizeFormula_,"ECAL signal cone size");
    ECALIsolConeSizeTFormula_=computeConeSizeTFormula(ECALIsolConeSizeFormula_,"ECAL isolation cone size");
    
    produces<TauDiscriminatorByIsolation>();
  }
  ~DiscriminationByIsolation(){
    //delete ;
  } 
  virtual void produce(Event&, const EventSetup&);
 private:  
  double discriminator(const TauRef&);
  // compute size of signal cone possibly depending on E(energy) and/or ET(transverse energy) of the tau-jet candidate
  double computeConeSize(const TauRef& theTau,const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax);
  TFormula computeConeSizeTFormula(const string& ConeSizeFormula,const char* errorMessage);
  void replaceSubStr(string& s,const string& oldSubStr,const string& newSubStr);
  string TauProducer_;
  bool UsePVconstraint_;
  string MatchingConeMetric_;
  string MatchingConeSizeFormula_;
  double MatchingConeVariableSize_min_;
  double MatchingConeVariableSize_max_;
  TFormula MatchingConeSizeTFormula_;
  bool ApplyDiscriminationByTrackerIsolation_;
  string TrackerSignalConeMetric_;
  string TrackerSignalConeSizeFormula_;
  TFormula TrackerSignalConeSizeTFormula_;
  double TrackerSignalConeVariableSize_max_;
  double TrackerSignalConeVariableSize_min_; 
  string TrackerIsolConeMetric_;
  string TrackerIsolConeSizeFormula_; 
  TFormula TrackerIsolConeSizeTFormula_;
  double TrackerIsolConeVariableSize_max_;
  double TrackerIsolConeVariableSize_min_;   
  //
  double LeadTrack_minPt_;
  double Track_minPt_; 
  int TrackerIsolAnnulus_Tracksmaxn_; 
  //
  bool ManipulateTracks_insteadofChargedHadrCands_;
  bool UseOnlyChargedHadr_for_LeadCand_; 
  double LeadCand_minPt_; 
  double ChargedHadrCand_minPt_; 
  int TrackerIsolAnnulus_Candsmaxn_;   
  bool ApplyDiscriminationByECALIsolation_; 
  double GammaCand_minPt_;
  string ECALSignalConeMetric_;
  string ECALSignalConeSizeFormula_;   
  TFormula ECALSignalConeSizeTFormula_;
  double ECALSignalConeVariableSize_max_;
  double ECALSignalConeVariableSize_min_; 
  string ECALIsolConeMetric_;
  string ECALIsolConeSizeFormula_;  
  TFormula ECALIsolConeSizeTFormula_;
  double ECALIsolConeVariableSize_max_;
  double ECALIsolConeVariableSize_min_; 
  double ECALIsolAnnulus_Candsmaxn_; 
  //
  double AreaMetric_recoElements_maxabsEta_; 
};
#endif
