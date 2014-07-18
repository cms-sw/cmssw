#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DQM/PhysicsHWW/interface/PFElToElAssMaker.h"

typedef math::XYZTLorentzVectorF LorentzVector;

void PFElToElAssMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using std::vector;

  hww.Load_pfels_elsidx();

  vector<LorentzVector> *pfels_p4_h = new vector<LorentzVector>;
  *pfels_p4_h = hww.pfels_p4();

  vector<LorentzVector> *els_p4_h = new vector<LorentzVector>;
  *els_p4_h = hww.els_p4();
  
  //loop over reco electrons and find the closest particle flow electron
  for (vector<LorentzVector>::const_iterator pfels_it = pfels_p4_h->begin(); pfels_it != pfels_p4_h->end(); pfels_it++) {       

    double pfel_eta = pfels_it->Eta();
    double pfel_phi = pfels_it->Phi();
       
    double minDR = 9999.;
    unsigned int i = 0;
    int index = -1; 

    for (vector<LorentzVector>::const_iterator els_it = els_p4_h->begin(); els_it != els_p4_h->end(); els_it++, i++) {

      double el_eta = els_it->Eta();
      double el_phi = els_it->Phi();
      double dR = deltaR(pfel_eta, pfel_phi, el_eta, el_phi);

      if(dR < minDR) {
        minDR = dR;
        index = i;
      }

    }

    if(minDR > 0.1) {
      minDR = -9999.;
      index = -1;
    }

    hww.pfels_elsidx().push_back(index);

  }

}
