#include "../interface/PFJetAlgorithm.h"
#include <iostream>
#include <set>

#include <TVector2.h>

using namespace std;

ostream& operator<<(ostream& out, const PFJetAlgorithm::Jet& jet) {
  if(!out) return out;
  cout<<"jet "<<jet.fVecIndexes.size()<<" particles, E_T = "<<jet.fMomentum.Et()<<" eta/phi "
      <<jet.fMomentum.Eta()<<" "<<jet.fMomentum.Phi();
  return out;
}


const vector< PFJetAlgorithm::Jet >&
PFJetAlgorithm::FindJets( const vector<TLorentzVector>* vecs) {


  fAllVecs = vecs;

  fAssignedVecs.clear(); 
  fAssignedVecs.reserve( vecs->size() );
  
  fEtOrderedSeeds.clear();
  
  for(unsigned i = 0; i<vecs->size(); i++) {
    // cout<<"i = "<<i<<endl;
    double et = (*vecs)[i].Et();
    
    int assigned = -1;
    if( et >= fSeedEt) {
      fJets.push_back( Jet(i, fAllVecs) );
      fEtOrderedSeeds.insert( make_pair(et, fJets.size()-1) );
      assigned = i;
    }
    fAssignedVecs.push_back(assigned);
  }

  
  // loop on seeds 
  for(IV iv = fEtOrderedSeeds.begin(); iv != fEtOrderedSeeds.end(); iv++ ) {
    
    Jet& currentjet = fJets[iv->second];

    double etaseed = currentjet.GetMomentum().Eta();
    double phiseed = currentjet.GetMomentum().Phi();
    
    
    // adding particles
    for(unsigned i = 0; i<fAllVecs->size(); i++) {
      
      if( fAssignedVecs[i] > -1 ) continue;

      double dr = DeltaR(etaseed, phiseed, 
                         (*fAllVecs)[i].Eta(), (*fAllVecs)[i].Phi());
      // cout<<"\t\tparticle "<<i<<" "<<dr<<" "<<(*fAllVecs)[i].Et()<<endl; 
                         
      if( dr < fConeAngle) {
        // cout<<"\t\tadding"<<endl;
        currentjet.Add(i);
        fAssignedVecs[i] = iv->second;
      }
    }
    
    // cout<<"\t seed processed"<<endl;
  }
  // cout<<"end find jets"<<endl;

  Update();
  CleanUp();

  return fJets;
}


double PFJetAlgorithm::DeltaR(double eta1, double phi1, double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = TVector2::Phi_mpi_pi(phi1 - phi2);
  return sqrt(deta*deta + dphi*dphi);
}


void PFJetAlgorithm::Update() {
  // use existing jets as seeds 
  //   cout<<"clearing seeds"<<endl;
  fEtOrderedSeeds.clear();
  for(unsigned ij = 0; ij<fJets.size(); ij++ ) {
    double et = fJets[ij].GetMomentum().Et();
    if(et >= fSeedEt)
      fEtOrderedSeeds.insert( make_pair(et, ij) );
  }

  // loop on seeds and add particles 
  //   cout<<"clearing assigned"<<endl;
  for(unsigned i = 0; i<fAssignedVecs.size(); i++) {
    fAssignedVecs[i] = -1;
  }

  //   cout<<"loop on seeds"<<endl;
  bool needupdate = false;
  for(IV iv = fEtOrderedSeeds.begin(); iv != fEtOrderedSeeds.end(); iv++ ) {
    
    Jet& currentjet = fJets[iv->second];

    TLorentzVector seedmom = currentjet.GetMomentum();

    double etaseed = seedmom.Eta();
    double phiseed = seedmom.Phi();
    //     cout<<"SEED\t"<<etaseed<<" "<<phiseed<<endl;

    currentjet.Clear();

    // adding particles
    for(unsigned i = 0; i<fAllVecs->size(); i++) {
      
      if( fAssignedVecs[i] > -1 ) continue;

      double dr = DeltaR(etaseed, phiseed, 
                         (*fAllVecs)[i].Eta(), (*fAllVecs)[i].Phi());
      // cout<<"\t\tparticle "<<i<<" "<<dr<<" "<<(*fAllVecs)[i].Et()<<endl; 
                         
      if( dr < fConeAngle) {
        // cout<<"\t\tadding"<<endl;
        currentjet.Add(i);
        fAssignedVecs[i] = iv->second;
      }
    }
    
    TLorentzVector deltav = currentjet.GetMomentum();
    deltav -= seedmom;
    if(deltav.M() > 0.001) needupdate = true;
  }
  
  if(needupdate) Update();
}



void PFJetAlgorithm::CleanUp() {
  
  //   cout<<"CleanUp : -----------------------------------------"<<endl;
  
  //   for(unsigned i=0; i<fJets.size(); i++) {
  //     cout<<fJets[i]<<endl;
  //   }

  vector< PFJetAlgorithm::Jet >  tmp = fJets;

  map< double, PFJetAlgorithm::Jet, greater<double> > etjets;
  // typedef map< double, PFJetAlgorithm::Jet, greater<double> >::iterator IJ;
  
  vector< PFJetAlgorithm::Jet >  tmp2;
  fJets.clear();

  for(unsigned i=0; i<tmp.size(); i++) {
    if( tmp[i].GetMomentum().Et() > 0 ) 
      // tmp2.push_back( tmp[i]); 
      etjets.insert( make_pair(tmp[i].GetMomentum().Et(), tmp[i]) );
  }

  //   cout<<"et map : "<<endl;
  //   for(IJ ij = etjets.begin(); ij != etjets.end(); ij++) {
  //     cout<<ij->second<<endl;
  //   }  

  MergeJets( etjets );

  //   for(IJ ij = etjets.begin(); ij != etjets.end(); ij++) {

  //     const TLorentzVector& mom1 = ij->second.GetMomentum();

  //     double eta1 = mom1.Eta();
  //     double phi1 = mom1.Phi();

  //     IJ closest = etjets.end();
  //     double drmin = 99999;
  //     for(IJ jj = etjets.begin(); jj != etjets.end(); jj++) {
   
  //       if( jj == ij ) continue;  
   
  //       const TLorentzVector& mom2 = jj->second.GetMomentum();
      
  //       double eta2 = mom2.Eta();
  //       double phi2 = mom2.Phi();

  //       double dr = DeltaR(eta1, phi1, eta2, phi2);

  //       if(dr<drmin) {
  //    drmin = dr; 
  //    closest = jj;
  //       }

  //       if(closest != etjets.end() ) {
  //    if ( dr < fConeMerge ) {
  //      ij->second += jj->second;
  //    }               
  //       } 
  //     }
  //   }

  //   cout<<"et map 2: "<<endl;
  for(IJ ij = etjets.begin(); ij != etjets.end(); ij++) {
    //     cout<<ij->second<<endl;
    fJets.push_back(ij->second);
  }  

  //   set<int> used;
  //   for(unsigned i=0; i<tmp2.size(); i++) {


  //     set<int>::iterator isused = used.find(i);
  //     if( isused != used.end() ) continue;
    
  //     Jet& jet1 = tmp2[i];
    
  //     // cout<<"\t jet "<<jet1<<endl;


  //     TLorentzVector mom1 = jet1.GetMomentum();

  //     double eta1 = mom1.Eta();
  //     double phi1 = mom1.Phi();
    
  //     // merge close jets
  //     for(unsigned j=0; j<tmp2.size(); j++) {
  //       if(i==j) continue;
  //       Jet& jet2 = tmp2[j];

      
  //       TLorentzVector mom2 = jet2.GetMomentum();
      
  //       double eta2 = mom2.Eta();
  //       double phi2 = mom2.Phi();
      
  //       double dr = DeltaR(eta1, phi1, eta2, phi2);

  //       // cout<<"\t\t test merge with "<<jet2<<", dr = "<<dr<<endl;
      
  //       if ( dr < fConeMerge ) {
  //    jet1 += jet2;
  //    used.insert( j );
  //    // cout<<"\t\t  yes "<<endl;
  //       }    
  //       else {
  //    // cout<<"\t\t  no "<<endl;
  //       }
  //     }
    
  //     used.insert( i );
  //     fJets.push_back(jet1);
  //   }
}


void PFJetAlgorithm::MergeJets(map< double, PFJetAlgorithm::Jet, greater<double> >& etjets) {

  // look for smallest distance between 2 jets : 
  
  IJ j1 = etjets.end();
  IJ j2 = etjets.end();
  double smallestdistance = 999999;

  for(IJ ij = etjets.begin(); ij != etjets.end(); ij++) {
    
    const TLorentzVector& mom1 = ij->second.GetMomentum();
    
    double eta1 = mom1.Eta();
    double phi1 = mom1.Phi();
    
    for(IJ jj = etjets.begin(); jj != etjets.end(); jj++) {
   
      if( jj == ij ) continue;  
      
      const TLorentzVector& mom2 = jj->second.GetMomentum();
      
      double eta2 = mom2.Eta();
      double phi2 = mom2.Phi();

      double dr = DeltaR(eta1, phi1, eta2, phi2);
      
      if(dr<smallestdistance) {
        smallestdistance = dr; 
        j1 = ij;
        j2 = jj;
      }
    }
  } 

  //   cout<<"smallest distance is between : "<<endl;
  //   cout<<j1->second<<endl;
  //   cout<<j2->second<<endl;

  if( smallestdistance < fConeMerge ) {
    j1->second += j2->second;
    etjets.erase(j2);
    
    MergeJets( etjets ); 
  }
}
