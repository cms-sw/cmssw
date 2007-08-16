#ifndef DiscriminationByIsolation_H_
#define DiscriminationByIsolation_H_

/* class DiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Aug 16 2007,
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

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class DiscriminationByIsolation : public EDProducer {
 public:
  explicit DiscriminationByIsolation(const ParameterSet& iConfig){   
    TauProducer_                           = iConfig.getParameter<string>("TauProducer");
    // following parameters are considered when DiscriminationByIsolation EDProducer runs on Tau objects built from either CaloJet or PFJet objects *BEGIN*  
    ReCompute_leadElementSignalIsolationElements_ = iConfig.getParameter<bool>("ReCompute_leadElementSignalIsolationElements");
    ApplyDiscriminationByTrackerIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByTrackerIsolation");
    // *END*   
    // following parameters are considered when DiscriminationByIsolation EDProducer runs on Tau objects built from PFJet objects *BEGIN* 
    ManipulateTracks_insteadofChargedHadrCands_ = iConfig.getParameter<bool>("ManipulateTracks_insteadofChargedHadrCands");
    TrackerIsolAnnulus_Candsmaxn_       = iConfig.getParameter<int>("TrackerIsolAnnulus_Candsmaxn");       
    ApplyDiscriminationByECALIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByECALIsolation");
    ECALIsolAnnulus_Candsmaxn_          = iConfig.getParameter<int>("ECALIsolAnnulus_Candsmaxn");
    // *END* 

    // following parameters are considered when ReCompute_leadElementSignalIsolationElements_ is set true
    // *BEGIN*
    
    //     following parameters are considered when
    //     DiscriminationByIsolation EDProducer runs on Tau objects built from CaloJet objects 
    //     OR (when DiscriminationByIsolation EDProducer runs on Tau objects built from PFJet objects AND ManipulateTracks_insteadofChargedHadrCands_ paremeter is set true)
    //     *BEGIN*
    TrackerIsolAnnulus_Tracksmaxn_      = iConfig.getParameter<int>("TrackerIsolAnnulus_Tracksmaxn");   
    //     *END*    

    //     following parameters are considered when DiscriminationByIsolation EDProducer runs on Tau objects built from either CaloJet or PFJet objects *BEGIN* 
    UsePVconstraint_                       = iConfig.getParameter<bool>("UsePVconstraint");
    MatchingConeMetric_                    = iConfig.getParameter<string>("MatchingConeMetric");
    MatchingConeSizeFormula_               = iConfig.getParameter<string>("MatchingConeSizeFormula");
    MatchingConeVariableSize_min_          = iConfig.getParameter<double>("MatchingConeVariableSize_min");
    MatchingConeVariableSize_max_          = iConfig.getParameter<double>("MatchingConeVariableSize_max");
    TrackerSignalConeMetric_               = iConfig.getParameter<string>("TrackerSignalConeMetric");
    TrackerSignalConeSizeFormula_          = iConfig.getParameter<string>("TrackerSignalConeSizeFormula");
    TrackerSignalConeVariableSize_min_     = iConfig.getParameter<double>("TrackerSignalConeVariableSize_min");
    TrackerSignalConeVariableSize_max_     = iConfig.getParameter<double>("TrackerSignalConeVariableSize_max");
    TrackerIsolConeMetric_                 = iConfig.getParameter<string>("TrackerIsolConeMetric"); 
    TrackerIsolConeSizeFormula_            = iConfig.getParameter<string>("TrackerIsolConeSizeFormula"); 
    TrackerIsolConeVariableSize_min_       = iConfig.getParameter<double>("TrackerIsolConeVariableSize_min");
    TrackerIsolConeVariableSize_max_       = iConfig.getParameter<double>("TrackerIsolConeVariableSize_max");
    //     *END*   
    
    //     following parameters are considered when DiscriminationByIsolation EDProducer runs on Tau objects built from PFJet objects *BEGIN* 
    UseOnlyChargedHadr_for_LeadCand_    = iConfig.getParameter<bool>("UseOnlyChargedHadr_for_LeadCand"); 
    LeadCand_minPt_                     = iConfig.getParameter<double>("LeadCand_minPt"); 
    ChargedHadrCand_minPt_              = iConfig.getParameter<double>("ChargedHadrCand_minPt");
    GammaCand_minPt_                    = iConfig.getParameter<double>("GammaCand_minPt");       
    ECALSignalConeMetric_               = iConfig.getParameter<string>("ECALSignalConeMetric");
    ECALSignalConeSizeFormula_          = iConfig.getParameter<string>("ECALSignalConeSizeFormula");    
    ECALSignalConeVariableSize_min_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_min");
    ECALSignalConeVariableSize_max_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_max");
    ECALIsolConeMetric_                 = iConfig.getParameter<string>("ECALIsolConeMetric");
    ECALIsolConeSizeFormula_            = iConfig.getParameter<string>("ECALIsolConeSizeFormula");      
    ECALIsolConeVariableSize_min_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_min");
    ECALIsolConeVariableSize_max_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_max");
    //     *END*   
    
    //     following parameters are considered when
    //     DiscriminationByIsolation EDProducer runs on Tau objects built from CaloJet objects 
    //     OR (when DiscriminationByIsolation EDProducer runs on Tau objects built from PFJet objects AND ManipulateTracks_insteadofChargedHadrCands_ paremeter is set true)
    //     *BEGIN*
    LeadTrack_minPt_                    = iConfig.getParameter<double>("LeadTrack_minPt");
    Track_minPt_                        = iConfig.getParameter<double>("Track_minPt");
    //     *END* 
   
    AreaMetric_recoElements_maxabsEta_  = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");
    
    // *END* 
   
    produces<TauDiscriminatorByIsolation>();
  }
  ~DiscriminationByIsolation(){
    //delete ;
  } 
  virtual void produce(Event&, const EventSetup&);
 private:  
  double discriminator(const TauRef&);
  string TauProducer_;
  bool ReCompute_leadElementSignalIsolationElements_;
  bool ApplyDiscriminationByTrackerIsolation_;
  bool ManipulateTracks_insteadofChargedHadrCands_;
  int TrackerIsolAnnulus_Candsmaxn_;   
  bool ApplyDiscriminationByECALIsolation_; 
  int ECALIsolAnnulus_Candsmaxn_; 
  int TrackerIsolAnnulus_Tracksmaxn_;   
  bool UsePVconstraint_;
  string MatchingConeMetric_;
  string MatchingConeSizeFormula_;
  double MatchingConeVariableSize_min_;
  double MatchingConeVariableSize_max_;
  string TrackerSignalConeMetric_;
  string TrackerSignalConeSizeFormula_;
  double TrackerSignalConeVariableSize_min_; 
  double TrackerSignalConeVariableSize_max_;
  string TrackerIsolConeMetric_;
  string TrackerIsolConeSizeFormula_; 
  double TrackerIsolConeVariableSize_min_;   
  double TrackerIsolConeVariableSize_max_;
  //
  double LeadTrack_minPt_;
  double Track_minPt_; 
  //
  bool UseOnlyChargedHadr_for_LeadCand_; 
  double LeadCand_minPt_; 
  double ChargedHadrCand_minPt_; 
  double GammaCand_minPt_;
  string ECALSignalConeMetric_;
  string ECALSignalConeSizeFormula_;   
  double ECALSignalConeVariableSize_min_; 
  double ECALSignalConeVariableSize_max_;
  string ECALIsolConeMetric_;
  string ECALIsolConeSizeFormula_;  
  double ECALIsolConeVariableSize_min_; 
  double ECALIsolConeVariableSize_max_;
  //
  double AreaMetric_recoElements_maxabsEta_; 
};
#endif
