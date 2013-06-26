{ //File: analyzeJets.C
  //Author: R. Harris from example of Oliver Gutsche
  //Date: June 16, 2006.
  //Description: Framework lite analysis of jets in root
  //Pre-requisites: Loading and enabling framework lite libraries and event file with analyzeJets_head.C

using namespace reco;

  // Create histo file and book histograms
  TFile histofile("jet_hists.root","RECREATE");  
  TH1F* h_pt = new TH1F("pt","Leading Jets pT (GeV)",100,0.0,500.0);
  TH1F* h_eta = new TH1F("eta","Leading Jets eta",100,-5.0,5.0);
  TH1F* h_phi = new TH1F("phi","Leading Jets phi",72,-3.141527,3.141527);
  TH1F* h_m2j = new TH1F("m2j","Dijet Mass",100,0.0,1000.0);

  //Get event tree and jet collection branch
  TTree *tree = (TTree*)file->Get("Events");
  std::vector<CaloJet> CaloJetCollection;
  TBranch *branch = tree->GetBranch(tree->GetAlias("MC5CaloJet"));
  branch->SetAddress(&CaloJetCollection);  

  // Loop over events
  for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) {
    double px[2], py[2], pz[2], E[2];
    std::cout << "Entry index: " << index << std::endl;  
    branch->GetEntry(index);
    int numJets = CaloJetCollection.size();
    std::cout << "Num Jets: " << numJets << std::endl;

    //Loop over jets
    for ( unsigned int jetIndex = 0; jetIndex < CaloJetCollection.size(); ++jetIndex ) {
      std::cout << "jet" << jetIndex  ;
      CaloJet* Jet = (CaloJet*)CaloJetCollection[jetIndex];

      //Get and printout jet pt, eta, phi for all jets
      double pt = Jet->ptUncached();    std::cout << ": pt=" << pt; 
      double eta = Jet->etaUncached();  std::cout << ", eta=" << eta;
      double phi = Jet->phiUncached();  std::cout << ", phi=" << phi << std::endl;

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
