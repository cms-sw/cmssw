#include "RecoJets/JetAlgorithms/interface/FixedGridEnergyDensity.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TMath.h"

using namespace reco;
using namespace edm;
using namespace std;

float FixedGridEnergyDensity::fixedGridRho(EtaRegion etaRegion){
  //define the phi bins
  vector<float> phibins;
  for (int i=0;i<10;i++) phibins.push_back(-TMath::Pi()+(2*i+1)*TMath::TwoPi()/20.);
  //define the eta bins
  vector<float> etabins;
  if (etaRegion==Central) {
    for (int i=0;i<8;++i) etabins.push_back(-2.1+0.6*i);
  } else if (etaRegion==Forward) {
     for (int i=0;i<10;++i) {
       if (i<5) etabins.push_back(-5.1+0.6*i);
       else etabins.push_back(2.7+0.6*(i-5));
     }
  } else if (etaRegion==All) {
     for (int i=0;i<18;++i) etabins.push_back(-5.1+0.6*i);
  }
  return fixedGridRho(etabins,phibins);
}


float FixedGridEnergyDensity::fixedGridRho(std::vector<float>& etabins,std::vector<float>& phibins) {
     float etadist = etabins[1]-etabins[0];
     float phidist = phibins[1]-phibins[0];
     float etahalfdist = (etabins[1]-etabins[0])/2.;
     float phihalfdist = (phibins[1]-phibins[0])/2.;
     vector<float> sumPFNallSMDQ;
     sumPFNallSMDQ.reserve(etabins.size()*phibins.size());
     for (unsigned int ieta=0;ieta<etabins.size();++ieta) {
       for (unsigned int iphi=0;iphi<phibins.size();++iphi) {
	 float pfniso_ieta_iphi = 0;
	 for(PFCandidateCollection::const_iterator pf_it = pfCandidates->begin(); pf_it != pfCandidates->end(); pf_it++) {
	   if (fabs(etabins[ieta]-pf_it->eta())>etahalfdist) continue;
	   if (fabs(reco::deltaPhi(phibins[iphi],pf_it->phi()))>phihalfdist) continue;
	   pfniso_ieta_iphi+=pf_it->pt();
	 }
	 sumPFNallSMDQ.push_back(pfniso_ieta_iphi);
       }
     }
     float evt_smdq = 0;
     sort(sumPFNallSMDQ.begin(),sumPFNallSMDQ.end());
     if (sumPFNallSMDQ.size()%2) evt_smdq = sumPFNallSMDQ[(sumPFNallSMDQ.size()-1)/2];
     else evt_smdq = (sumPFNallSMDQ[sumPFNallSMDQ.size()/2]+sumPFNallSMDQ[(sumPFNallSMDQ.size()-2)/2])/2.;
     return evt_smdq/(etadist*phidist);
}
