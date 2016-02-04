/* class CaloRecoTauDiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Sep 5 2007,
 * contributors : Ludovic Houchu (Ludovic.Houchu@cern.ch ; IPHC, Strasbourg), Christian Veelken (veelken@fnal.gov ; UC Davis), Evan Friis (UC Davis)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

using namespace reco;

class CaloRecoTauDiscriminationByIsolation : public CaloTauDiscriminationProducerBase {
 public:
  explicit CaloRecoTauDiscriminationByIsolation(const edm::ParameterSet& iConfig):CaloTauDiscriminationProducerBase(iConfig){   
    applyDiscriminationByTrackerIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByTrackerIsolation");
    TrackerIsolAnnulus_maximumOccupancy_   = iConfig.getParameter<unsigned>("TrackerIsolAnnulus_maximumOccupancy");   
    
    applyDiscriminationByECALIsolation_    = iConfig.getParameter<bool>("ApplyDiscriminationByECALIsolation");
    EcalIsolAnnulus_maximumSumEtCut_       = iConfig.getParameter<double>("ECALisolAnnulus_maximumSumEtCut");   
  }
  ~CaloRecoTauDiscriminationByIsolation(){} 
  double discriminate(const CaloTauRef&);
 private:  
  bool applyDiscriminationByTrackerIsolation_;
  unsigned TrackerIsolAnnulus_maximumOccupancy_;   
  bool applyDiscriminationByECALIsolation_;
  double EcalIsolAnnulus_maximumSumEtCut_;
};

double CaloRecoTauDiscriminationByIsolation::discriminate(const CaloTauRef& caloTau)
{
  if ( applyDiscriminationByTrackerIsolation_ ){  
    if ( caloTau->isolationTracks().size() > TrackerIsolAnnulus_maximumOccupancy_ ) return 0.;
  }
  
  if ( applyDiscriminationByECALIsolation_ ) {
    if ( caloTau->isolationECALhitsEtSum() > EcalIsolAnnulus_maximumSumEtCut_ ) return 0.;
  }
  
  // N.B. the lead track requirement must be included in the discriminants
  return 1.;
}

DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByIsolation);
