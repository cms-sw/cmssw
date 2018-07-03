/* class PFRecoTauDiscriminationAgainstElectron2
 * created : Apr 3, 2014
 * revised : ,
 * Authorss : Aruna Nayak (DESY)
 */

#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDCut2.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TMath.h>
#include "TPRegexp.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
 
using namespace reco;
typedef std::pair<double, double> pdouble;

class PFRecoTauDiscriminationAgainstElectron2 : public PFTauDiscriminationProducerBase  {
public:
  explicit PFRecoTauDiscriminationAgainstElectron2(const edm::ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig)
  {

    LeadPFChargedHadrEoP_barrel_min_ = iConfig.getParameter<double>("LeadPFChargedHadrEoP_barrel_min");
    LeadPFChargedHadrEoP_barrel_max_ = iConfig.getParameter<double>("LeadPFChargedHadrEoP_barrel_max");
    Hcal3x3OverPLead_barrel_max_ = iConfig.getParameter<double>("Hcal3x3OverPLead_barrel_max");
    GammaEtaMom_barrel_max_ = iConfig.getParameter<double>("GammaEtaMom_barrel_max");
    GammaPhiMom_barrel_max_ = iConfig.getParameter<double>("GammaPhiMom_barrel_max");
    GammaEnFrac_barrel_max_ = iConfig.getParameter<double>("GammaEnFrac_barrel_max");
    LeadPFChargedHadrEoP_endcap_min1_ = iConfig.getParameter<double>("LeadPFChargedHadrEoP_endcap_min1");
    LeadPFChargedHadrEoP_endcap_max1_ = iConfig.getParameter<double>("LeadPFChargedHadrEoP_endcap_max1");
    LeadPFChargedHadrEoP_endcap_min2_ = iConfig.getParameter<double>("LeadPFChargedHadrEoP_endcap_min2");
    LeadPFChargedHadrEoP_endcap_max2_ = iConfig.getParameter<double>("LeadPFChargedHadrEoP_endcap_max2");
    Hcal3x3OverPLead_endcap_max_ = iConfig.getParameter<double>("Hcal3x3OverPLead_endcap_max");
    GammaEtaMom_endcap_max_ = iConfig.getParameter<double>("GammaEtaMom_endcap_max");
    GammaPhiMom_endcap_max_ = iConfig.getParameter<double>("GammaPhiMom_endcap_max");
    GammaEnFrac_endcap_max_ = iConfig.getParameter<double>("GammaEnFrac_endcap_max");
    keepTausInEcalCrack_ = iConfig.getParameter<bool>("keepTausInEcalCrack");
    rejectTausInEcalCrack_ = iConfig.getParameter<bool>("rejectTausInEcalCrack"); 

    applyCut_hcal3x3OverPLead_ = iConfig.getParameter<bool>("applyCut_hcal3x3OverPLead");  
    applyCut_leadPFChargedHadrEoP_ = iConfig.getParameter<bool>("applyCut_leadPFChargedHadrEoP");  
    applyCut_GammaEtaMom_ = iConfig.getParameter<bool>("applyCut_GammaEtaMom");  
    applyCut_GammaPhiMom_ = iConfig.getParameter<bool>("applyCut_GammaPhiMom");  
    applyCut_GammaEnFrac_ = iConfig.getParameter<bool>("applyCut_GammaEnFrac");  
    applyCut_HLTSpecific_ = iConfig.getParameter<bool>("applyCut_HLTSpecific");

    etaCracks_string_ = iConfig.getParameter<std::vector<std::string>>("etaCracks");
    
    verbosity_ = ( iConfig.exists("verbosity") ) ?
      iConfig.getParameter<int>("verbosity") : 0;

    cuts2_ = new AntiElectronIDCut2();
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  double discriminate(const PFTauRef&) const override;
  
  ~PFRecoTauDiscriminationAgainstElectron2() override
  {  }

private:
  bool isInEcalCrack(double) const;
  std::vector<pdouble> etaCracks_;
  std::vector<std::string> etaCracks_string_;

  AntiElectronIDCut2* cuts2_;
  
  double LeadPFChargedHadrEoP_barrel_min_;
  double LeadPFChargedHadrEoP_barrel_max_;
  double Hcal3x3OverPLead_barrel_max_;
  double GammaEtaMom_barrel_max_;
  double GammaPhiMom_barrel_max_;
  double GammaEnFrac_barrel_max_;
  double LeadPFChargedHadrEoP_endcap_min1_;
  double LeadPFChargedHadrEoP_endcap_max1_;
  double LeadPFChargedHadrEoP_endcap_min2_;
  double LeadPFChargedHadrEoP_endcap_max2_;
  double Hcal3x3OverPLead_endcap_max_;
  double GammaEtaMom_endcap_max_;
  double GammaPhiMom_endcap_max_;
  double GammaEnFrac_endcap_max_;

  bool keepTausInEcalCrack_;
  bool rejectTausInEcalCrack_;

  bool applyCut_hcal3x3OverPLead_; 
  bool applyCut_leadPFChargedHadrEoP_; 
  bool applyCut_GammaEtaMom_; 
  bool applyCut_GammaPhiMom_; 
  bool applyCut_GammaEnFrac_; 
  bool applyCut_HLTSpecific_;

  int verbosity_;
};

void PFRecoTauDiscriminationAgainstElectron2::beginEvent(const edm::Event& evt, const edm::EventSetup& es)
{
  cuts2_->SetBarrelCutValues(LeadPFChargedHadrEoP_barrel_min_,
                            LeadPFChargedHadrEoP_barrel_max_,
                            Hcal3x3OverPLead_barrel_max_,
                            GammaEtaMom_barrel_max_,
                            GammaPhiMom_barrel_max_,
                            GammaEnFrac_barrel_max_
                            );
  
  cuts2_->SetEndcapCutValues(LeadPFChargedHadrEoP_endcap_min1_,
                            LeadPFChargedHadrEoP_endcap_max1_,
                            LeadPFChargedHadrEoP_endcap_min2_,
                            LeadPFChargedHadrEoP_endcap_max2_,
                            Hcal3x3OverPLead_endcap_max_,
                            GammaEtaMom_endcap_max_,
                            GammaPhiMom_endcap_max_,
                            GammaEnFrac_endcap_max_
                            );

  cuts2_->ApplyCut_EcalCrack(keepTausInEcalCrack_, rejectTausInEcalCrack_);

  cuts2_->ApplyCuts(applyCut_hcal3x3OverPLead_,  
		    applyCut_leadPFChargedHadrEoP_,  
		    applyCut_GammaEtaMom_,  
		    applyCut_GammaPhiMom_,  
		    applyCut_GammaEnFrac_,  
		    applyCut_HLTSpecific_ 
		    );
  //Ecal cracks in eta
  etaCracks_.clear();
  TPRegexp regexpParser_range("([0-9.e+/-]+):([0-9.e+/-]+)");
  for ( std::vector<std::string>::const_iterator etaCrack = etaCracks_string_.begin();
	etaCrack != etaCracks_string_.end(); ++etaCrack ) {
    TObjArray* subStrings = regexpParser_range.MatchS(etaCrack->data());
    if ( subStrings->GetEntries() == 3 ) {
      //std::cout << "substrings(1) = " << ((TObjString*)subStrings->At(1))->GetString() << std::endl;
      double range_begin = ((TObjString*)subStrings->At(1))->GetString().Atof();
      //std::cout << "substrings(2) = " << ((TObjString*)subStrings->At(2))->GetString() << std::endl;
      double range_end = ((TObjString*)subStrings->At(2))->GetString().Atof();
      etaCracks_.push_back(pdouble(range_begin, range_end));
    }
  }
  
  cuts2_->SetEcalCracks(etaCracks_);

}

double PFRecoTauDiscriminationAgainstElectron2::discriminate(const PFTauRef& thePFTauRef) const
{
  double discriminator = 0.;
  
  // ensure tau has at least one charged object

  if( (*thePFTauRef).leadPFChargedHadrCand().isNull() )
    {
      return 0.;
    } 
  else
    {
      discriminator = cuts2_->Discriminator(*thePFTauRef);
    }


  if ( verbosity_ ) {
    std::cout<<" Taus : "<<TauProducer_<<std::endl;
    std::cout << "<PFRecoTauDiscriminationAgainstElectron2::discriminate>:" << std::endl;
    std::cout << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta() << ", phi = " << thePFTauRef->phi() << std::endl;
    std::cout << " discriminator value = " << discriminator << std::endl;
    std::cout << " Prongs in tau: " << thePFTauRef->signalPFChargedHadrCands().size() << std::endl;
  }

  return discriminator;
}

bool
PFRecoTauDiscriminationAgainstElectron2::isInEcalCrack(double eta) const
{
  eta = fabs(eta);
  return (eta < 0.018 ||
          (eta>0.423 && eta<0.461) ||
          (eta>0.770 && eta<0.806) ||
          (eta>1.127 && eta<1.163) ||
          (eta>1.460 && eta<1.558));
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectron2);
