//File: analyzeJets.C
//Author: R. Harris 
//Date: June 22, 2006.
//Description: Framework lite analysis of jets in root using TChain
//Pre-requisites: To use with CINT:
//                   .x analyzeJets_headChain.C
//                   .x analyzeJetsChain.C
//                To use with ACliC in compiled mode
//                   .x prepareCompile.C
//                   .x analyzeJets_headChain.C
//                   .x analyzeJetsChain.C++
//     You will need to replace evtgen_jets*.root with 
//     your chain of root files in analyzeJets*.C
//
#ifndef __CINT__
#include <vector>
#include "TChain.h"
#include "TH1.h"
#include "TFile.h"
#include <iostream>
#include "DataFormats/JetReco/interface/CaloJet.h"
#endif

void analyzeJetsChain()
{

  // Create histo file and book histograms
  TFile histofile("jet_hists.root","RECREATE");  
  TH1F* h_pt = new TH1F("pt","Leading Jets pT (GeV)",100,0.0,1000.0);
  TH1F* h_eta = new TH1F("eta","Leading Jets eta",100,-5.0,5.0);
  TH1F* h_phi = new TH1F("phi","Leading Jets phi",72,-3.141527,3.141527);
  TH1F* h_m2j = new TH1F("m2j","Dijet Mass",100,0.0,1000.0);

  // Declare CaloJetCollection.
  std::vector<reco::CaloJet> CaloJetCollection;

  #ifndef __CINT__
    // For the compiled version we need to define the chain here
    TChain chain("Events");
    chain.Add("evtgen_jets.root");
    chain.Add("evtgen_jets2.root");
   #endif

   // Number of entries in chain
  Int_t   nevent = chain.GetEntries();

  // Tell root we only want the CaloJets branches.
  chain.SetBranchStatus("*",0);
  chain.SetBranchStatus("recoCaloJets*",1);

  // Loop over events
  for ( int index = 0; index < nevent; ++index ) {

    // Load the TTree corresponding to event index and return the entry with respect to that tree.
    int current = chain.LoadTree(index);

    // Check if we are on the first entry in the tree
    if (current==0) {

       //Read the first entry in this new tree, needed to set branch address.
       chain.GetEvent(index); 

       //Set the branch address for this new tree
       chain.SetBranchAddress(chain.GetAlias("MC5CaloJet"),&CaloJetCollection);
    }
    
    // Read the event.
    chain.GetEvent(index);

    double px[2], py[2], pz[2], E[2];
    std::cout << "Entry index: " << index << std::endl;  
    //chain.SetBranchAddress("CaloJets_midPointCone5CaloJets.obj",&CaloJetCollection);
    int numJets = CaloJetCollection.size();
    std::cout << "Num Jets: " << numJets << std::endl;

    //Loop over jets
    for ( unsigned int jetIndex = 0; jetIndex < CaloJetCollection.size(); ++jetIndex ) {
      std::cout << "jet" << jetIndex  ;
     #ifndef __CINT__
       reco::CaloJet* Jet= &(CaloJetCollection[jetIndex]);
     #else
       reco::CaloJet* Jet = (reco::CaloJet*)CaloJetCollection[jetIndex];
     #endif

      //Get and printout jet pt, eta, phi for all jets
      double pt = Jet->pt();    std::cout << ": pt=" << pt; 
      double eta = Jet->eta();  std::cout << ", eta=" << eta;
      double phi = Jet->phi();  std::cout << ", phi=" << phi << std::endl;

      if(jetIndex<2)
      {

        //Fill Histograms for two highest pt jets
        h_pt->Fill(pt); 
	h_eta->Fill(eta); 
	h_phi->Fill(phi);       
        
       //Get Lorentz Vector components of two highest pt jets
       px[jetIndex] = Jet->px();
       py[jetIndex] = Jet->py();
       pz[jetIndex] = Jet->pz();
       E[jetIndex]  = Jet->energy();
      }
    }
    //Printout Dijet Mass and Fill Dijet Mass histogram
    if( numJets >= 2 ){
      double DijetMass = sqrt( pow(E[0]+E[1],2) - pow(px[0]+px[1],2)
                                                - pow(py[0]+py[1],2)
                                                - pow(pz[0]+pz[1],2) );
      std::cout << "Dijet Mass = " << DijetMass  << std::endl;
      h_m2j->Fill(DijetMass);    
      
    }
  }
  // save histograms
  histofile.Write();
  histofile.Close();
}  
