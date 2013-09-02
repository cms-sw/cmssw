#include <vector>
#include <TFile.h>
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

/* class PFRecoTauDiscriminationByMVAIsolation
 *
 * Discriminates taus based on isolation deposit rings around tau
 *  
 *
 * 
 * Authors : Matthew Chan
 */

using namespace std;
using namespace reco;

namespace reco {
  namespace tau {
    namespace cone {
      struct IsoRings
      {
	vector<int> niso;
	vector<vector<float> > rings;
	vector<vector<float> > shapes;

	vector<float> getVector()
	{
	  vector<float> all;
	  all.reserve(33);
    
	  for(unsigned int i = 0; i < niso.size(); i++)
	    all.push_back(niso[i]);

	  for(unsigned int i = 0; i < rings.size(); i++)
	    all.insert(all.end(), rings[i].begin(), rings[i].end());

	  for(unsigned int i = 0; i < shapes.size(); i++)
	    all.insert(all.end(), shapes[i].begin(), shapes[i].end());

	  return all;
	}
      };
    }
  }
}

class PFRecoTauDiscriminationByMVAIsolation : public PFTauDiscriminationProducerBase
{
public:
  explicit PFRecoTauDiscriminationByMVAIsolation(const edm::ParameterSet& iConfig) : 
    PFTauDiscriminationProducerBase(iConfig),
    rhoProducer_(iConfig.getParameter<edm::InputTag>("rhoProducer")),
    gbrfFilePath_(iConfig.getParameter<edm::FileInPath>("gbrfFilePath")),
    returnMVA_(iConfig.getParameter<bool>("returnMVA")),
    mvaMin_(iConfig.getParameter<double>("mvaMin")),
    rho_(0)
  {
    // Prediscriminant fail value
    if(returnMVA_)
      prediscriminantFailValue_ = -1;
    else
      prediscriminantFailValue_ = 0;

    // Read GBRForest
    TFile *gbrfFile = new TFile(gbrfFilePath_.fullPath().data());
    gbrfTauIso_ = (GBRForest *)(gbrfFile->Get("gbrfTauIso"));
  }

  ~PFRecoTauDiscriminationByMVAIsolation(){} 

  void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup);
  double discriminate(const PFTauRef& pfTau);
  reco::tau::cone::IsoRings computeIsoRings(const PFTauRef& pfTau);

private:
  edm::InputTag rhoProducer_;
  edm::FileInPath gbrfFilePath_;
  GBRForest *gbrfTauIso_;
  bool returnMVA_;
  double mvaMin_;
  double rho_;
};

void PFRecoTauDiscriminationByMVAIsolation::beginEvent(const edm::Event& event,
    const edm::EventSetup& eventSetup)
{
  // Get rho of event
  edm::Handle<double> hRho;
  event.getByLabel(rhoProducer_, hRho);
  rho_ = *hRho;
}

double PFRecoTauDiscriminationByMVAIsolation::discriminate(const PFTauRef& thePFTauRef)
{
  reco::tau::cone::IsoRings isoRings = computeIsoRings(thePFTauRef);
  vector<float> mvainput = isoRings.getVector();
  mvainput.push_back(rho_);
  double mvaValue = gbrfTauIso_->GetClassifier(&mvainput[0]);

  return returnMVA_ ? mvaValue : mvaValue > mvaMin_;
}

reco::tau::cone::IsoRings PFRecoTauDiscriminationByMVAIsolation::computeIsoRings(const PFTauRef& pfTau)
{
  vector<int>            niso(3);
  vector<vector<float> > rings(3, vector<float>(5));
  vector<vector<float> > shapes(3, vector<float>(5));
  vector<float>          isoptsum(3);

  for(unsigned int i = 0; i < pfTau->isolationPFCands().size(); i++)
  {
    const PFCandidatePtr pf = pfTau->isolationPFCands().at(i);

    // Angular distance between PF candidate and tau
    float deta = pfTau->eta() - pf->eta();
    float dphi = reco::deltaPhi(pfTau->phi(), pf->phi());
    float dr = reco::deltaR(pfTau->eta(), pfTau->phi(), pf->eta(), pf->phi());
    int pftype = 0;

    // Determine PF candidate type
    if(pf->charge() != 0)                           pftype = 0;
    else if(pf->particleId() == PFCandidate::gamma) pftype = 1;
    else                                            pftype = 2;

    // Number of isolation candidates by type
    niso[pftype]++;

    // Isolation Rings
    if(dr < 0.1)      rings[pftype][0] += pf->pt();
    else if(dr < 0.2) rings[pftype][1] += pf->pt();
    else if(dr < 0.3) rings[pftype][2] += pf->pt();
    else if(dr < 0.4) rings[pftype][3] += pf->pt();
    else if(dr < 0.5) rings[pftype][4] += pf->pt();

    // Angle Shape Variables
    shapes[pftype][0] += pf->pt() * deta;
    shapes[pftype][1] += pf->pt() * dphi;
    shapes[pftype][2] += pf->pt() * deta*deta;
    shapes[pftype][3] += pf->pt() * dphi*dphi;
    shapes[pftype][4] += pf->pt() * deta*dphi;
    isoptsum[pftype]  += pf->pt();
  }

  // Mean and variance of angle variables are weighted by pT
  for(unsigned int i = 0; i < shapes.size(); i++)
  {
    for(unsigned int j = 0; j < shapes[i].size(); j++)
    {
      shapes[i][j] = isoptsum[i] > 0 ? fabs(shapes[i][j]/isoptsum[i]) : 0;
    }
  }

  // Fill IsoRing object
  reco::tau::cone::IsoRings isoRings;
  isoRings.niso = niso;
  isoRings.rings = rings;
  isoRings.shapes = shapes;

  return isoRings;
}
DEFINE_FWK_MODULE(PFRecoTauDiscriminationByMVAIsolation);
