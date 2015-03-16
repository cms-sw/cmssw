#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DQM/PhysicsHWW/interface/HypDilepMaker.h"


typedef math::XYZTLorentzVectorF LorentzVector;
using namespace reco;
using namespace edm;
using namespace std;

bool testJetForLeptons(const LorentzVector& jetP4, const LorentzVector& lepp4) {
  
  
  bool matched = false;
  float lepphi  = lepp4.Phi();
  float jetphi = jetP4.Phi();
   
  float lepeta  = lepp4.Eta();
  float jeteta = jetP4.Eta();
   
  float dphi = lepphi - jetphi;
  float deta = lepeta - jeteta;
  if(fabs(dphi) > TMath::Pi() ) dphi = 2*TMath::Pi() - fabs(dphi);
   
  double dR = sqrt(dphi*dphi + deta*deta);
  if (dR < 0.4) 
    matched = true;
  
  return !matched;
}

void HypDilepMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  hww.Load_hyp_jets_p4();
  hww.Load_hyp_type();
  hww.Load_hyp_p4();
  hww.Load_hyp_lt_charge();
  hww.Load_hyp_lt_index();
  hww.Load_hyp_lt_id();
  hww.Load_hyp_lt_p4();
  hww.Load_hyp_ll_charge();
  hww.Load_hyp_ll_index();
  hww.Load_hyp_ll_id();
  hww.Load_hyp_ll_p4();

  double looseptcut = 10.0;
  double tightptcut = 20.0;
  double hypJetMaxEtaCut = 5.0;
  double hypJetMinPtCut = 30.0;

  // muon charge
  const vector<int> &mus_charge = hww.mus_charge(); 

  //muon p4
  const vector<LorentzVector> &mus_p4 = hww.mus_p4();

  //muon type
  const vector<int> &mus_type =  hww.mus_type();

  //-----------------------------------------------------------
  // electron variables
  //-----------------------------------------------------------
  const vector<int> &els_charge =hww.els_charge();

  // electron p4
  const vector<LorentzVector> &els_p4 = hww.els_p4();

  unsigned int nmus = mus_p4.size();
  unsigned int nels = els_p4.size();


  const vector<LorentzVector> &jets_p4 = hww.pfjets_p4();

  //------------------------------------------------------------
  // loop over the muons
  //------------------------------------------------------------
  //get the candidates and make hypotheses 
  for(unsigned int mus_index_1 = 0; mus_index_1 < nmus; mus_index_1++) {//first muon loop
    for(unsigned int mus_index_2 = 0; mus_index_2 < nmus; mus_index_2++) {//second muon loop

      if(mus_index_1 == mus_index_2) continue;
      if(mus_index_2 < mus_index_1)  continue;  //avoid double counting

      //don't look at standalone muons
      if(mus_type[mus_index_1] == 8) continue;
      if(mus_type[mus_index_2] == 8) continue;
      
      float mu_pt1 = mus_p4[mus_index_1].Pt();
      float mu_pt2 = mus_p4[mus_index_2].Pt();
      
      //if either fail the loose cut, go to the next muon
      if(mu_pt1 < looseptcut || mu_pt2 < looseptcut) continue;
      
      //if neither one passes the tight cut, go to the next muon
      if(mu_pt1 < tightptcut && mu_pt2 < tightptcut) continue;

      int tight_index = mus_index_1;
      int loose_index = mus_index_2;

      /*
	figure out which one should be tight and which should
	be loose in case one passes the tight cut and the other 
	does not
      */
      if(mu_pt1 < tightptcut && mu_pt2 > tightptcut) {
        tight_index = mus_index_2;
        loose_index = mus_index_1;
      }
      if(mu_pt2 < tightptcut && mu_pt1 > tightptcut) {
        tight_index = mus_index_1;
        loose_index = mus_index_2;
      }


      //fill the Jet vars
      vector<int> temp_jets_idx;
      vector<LorentzVector>  temp_jets_p4;      

	
      for(unsigned int i = 0; i<jets_p4.size(); i++) {
	
        // we don't want jets that overlap with electrons
        bool overlapsWithLepton = false;
        if(!testJetForLeptons(jets_p4[i], mus_p4[loose_index])) 
          overlapsWithLepton = true;
        if(!testJetForLeptons(jets_p4[i], mus_p4[tight_index])) 
          overlapsWithLepton = true;

        double jet_eta = jets_p4[i].eta();
        double jet_pt  = jets_p4[i].Pt();
        
        if( fabs(jet_eta) < hypJetMaxEtaCut && jet_pt  > hypJetMinPtCut && !overlapsWithLepton) { //hyp jetas
          temp_jets_idx.push_back(i);
          temp_jets_p4                     .push_back(jets_p4[i]);
        }
      }

      hww.hyp_jets_p4()       .push_back(temp_jets_p4                          );
      hww.hyp_type()          .push_back(0                                     );
      hww.hyp_p4()            .push_back(mus_p4[tight_index]+mus_p4[loose_index]);
      hww.hyp_lt_charge()     .push_back(mus_charge[tight_index]  );
      hww.hyp_lt_index()      .push_back(tight_index                         );
      hww.hyp_lt_id()         .push_back(-13*(mus_charge[tight_index]));
      hww.hyp_lt_p4()         .push_back(mus_p4           [tight_index]  );
      hww.hyp_ll_charge()     .push_back(mus_charge[loose_index]  );
      hww.hyp_ll_index()      .push_back(loose_index                         );
      hww.hyp_ll_id()         .push_back(-13*(mus_charge[loose_index]));
      hww.hyp_ll_p4()         .push_back(mus_p4           [loose_index]  );
    }
  }  

  //------------------------------------------------------------
  // loop over the elecrons
  //------------------------------------------------------------
  //get the candidates and make hypotheses 
  for(unsigned int els_index_1 = 0; els_index_1 < nels; els_index_1++) {
    for(unsigned int els_index_2 = 0; els_index_2 < nels; els_index_2++) {
      
      if(els_index_1 == els_index_2) continue;
      if(els_index_2 < els_index_1)  continue;  //avoid double counting
      
      float el_pt1 = els_p4[els_index_1].Pt();
      float el_pt2 = els_p4[els_index_2].Pt();
      
      //if either fail the loose cut, go to the next muon
      if(el_pt1 < looseptcut || el_pt2 < looseptcut) continue;
      
      //if neither one passes the tight cut, continue
      if(el_pt1 < tightptcut && el_pt2 < tightptcut) continue;
      
      int tight_index = els_index_1;
      int loose_index = els_index_2;
      
      /*
	figure out which one should be tight and which should
	be loose in case one passes the tight cut and the other 
	does not
      */
      if(el_pt1 < tightptcut && el_pt2 > tightptcut) {
        tight_index = els_index_2;
        loose_index = els_index_1;
      }
      if(el_pt2 < tightptcut && el_pt1 > tightptcut) {
        tight_index = els_index_1;
        loose_index = els_index_2;
      }


      //fill the Jet vars
      vector<int> temp_jets_idx;
      vector<LorentzVector>  temp_jets_p4;      

	
      for(unsigned int i = 0; i<jets_p4.size(); i++) {
	
        // we don't want jets that overlap with electrons
        bool overlapsWithLepton = false;
        if(!testJetForLeptons(jets_p4[i], els_p4[loose_index])) 
          overlapsWithLepton = true;
        if(!testJetForLeptons(jets_p4[i], els_p4[tight_index])) 
          overlapsWithLepton = true;

        double jet_eta = jets_p4[i].eta();
        double jet_pt  = jets_p4[i].Pt();
        
        if( fabs(jet_eta) < hypJetMaxEtaCut && jet_pt  > hypJetMinPtCut && !overlapsWithLepton) { //hyp jetas
          temp_jets_idx.push_back(i);
          temp_jets_p4                     .push_back(jets_p4[i]);
        }
      }

      hww.hyp_jets_p4()       .push_back(temp_jets_p4                          );
      hww.hyp_type()          .push_back(3);
      hww.hyp_p4()            .push_back(els_p4[tight_index]+els_p4[loose_index]);
      hww.hyp_lt_charge()     .push_back(els_charge       [tight_index]  );
      hww.hyp_lt_index()      .push_back(tight_index                         );
      hww.hyp_lt_id()         .push_back(-11*(els_charge   [tight_index]));
      hww.hyp_lt_p4()         .push_back(els_p4           [tight_index]  );
      hww.hyp_ll_charge()     .push_back(els_charge       [loose_index]  );
      hww.hyp_ll_index()      .push_back(loose_index                         );
      hww.hyp_ll_id()         .push_back(-11*(els_charge   [loose_index]));
      hww.hyp_ll_p4()         .push_back(els_p4           [loose_index]  );
    }
  }  
  
  /*------------------------------------------------------------
    The EMu, MuE cases
    To avoid double counting, only make MuE if Mu is tight and E is loose
  */

  for(unsigned int els_index = 0; els_index < nels; els_index++) {
    for(unsigned int mus_index = 0; mus_index < nmus; mus_index++) {

      if(mus_type[mus_index] == 8) continue;

      float el_pt = els_p4[els_index].Pt();
      float mu_pt = mus_p4[mus_index].Pt();

      //if either fail the loose cut, go to the next muon
      if(el_pt < looseptcut || mu_pt < looseptcut) continue;

      //if both fail the tight cut, continue
      if(el_pt < tightptcut && mu_pt < tightptcut) continue;
      
      //fill the Jet vars
      vector<int> temp_jets_idx;
      vector<LorentzVector>  temp_jets_p4;      

	
      for(unsigned int i = 0; i<jets_p4.size(); i++) {
	
        // we don't want jets that overlap with electrons
        bool overlapsWithLepton = false;
        if(!testJetForLeptons(jets_p4[i], els_p4[els_index])) 
          overlapsWithLepton = true;
        if(!testJetForLeptons(jets_p4[i], mus_p4[mus_index])) 
          overlapsWithLepton = true;

        double jet_eta = jets_p4[i].eta();
        double jet_pt  = jets_p4[i].Pt();
        
        if( fabs(jet_eta) < hypJetMaxEtaCut && jet_pt  > hypJetMinPtCut && !overlapsWithLepton) { //hyp jetas
          temp_jets_idx.push_back(i);
          temp_jets_p4                     .push_back(jets_p4[i]);
        }
      }

      hww.hyp_jets_p4()       .push_back(temp_jets_p4                          );
      hww.hyp_p4()            .push_back(mus_p4[mus_index]+els_p4[els_index]                 );
	
      if(el_pt < tightptcut && mu_pt > tightptcut) {
        hww.hyp_type()            .push_back(1);
        hww.hyp_lt_charge()       .push_back(mus_charge[mus_index]  );
        hww.hyp_lt_index()        .push_back(mus_index                         );
        hww.hyp_lt_id()           .push_back(-13*(mus_charge[mus_index]));
        hww.hyp_lt_p4()           .push_back(mus_p4           [mus_index]  );
        hww.hyp_ll_charge()       .push_back(els_charge       [els_index]  );
        hww.hyp_ll_index()        .push_back(els_index                         );
        hww.hyp_ll_id()           .push_back(-11*(els_charge   [els_index]));
        hww.hyp_ll_p4()           .push_back(els_p4           [els_index]  );
	
	  
      } else {
        hww.hyp_type()            .push_back(2);
        hww.hyp_lt_charge()       .push_back(els_charge       [els_index]  );
        hww.hyp_lt_index()        .push_back(els_index                         );
        hww.hyp_lt_id()           .push_back(-11*(els_charge   [els_index]));
        hww.hyp_lt_p4()           .push_back(els_p4           [els_index]  );
        hww.hyp_ll_charge()       .push_back(mus_charge[mus_index]  );
        hww.hyp_ll_index()        .push_back(mus_index                         );
        hww.hyp_ll_id()           .push_back(-13*(mus_charge[mus_index]));
        hww.hyp_ll_p4()           .push_back(mus_p4           [mus_index]  );
	
      }
    }
  }
}

