
{

// Attention :
// This example will ONLY work if you execute GenH190... cfg
// with the option of writing edm::Event out (via PoolOutputModule)


// No need for this here anymore, 
// since it already in the rootlogon.C
//
//   gSystem->Load("libFWCoreFWLite") ;
//   FWLiteEnabler::enable() ;
   
   TFile* f = new TFile("../../../Configuration/Examples/data/pythiaH190ZZ4mu.root") ;
   
   TTree* tevt = f->Get("Events") ;
   
   int nevt = tevt->GetEntries() ;
   std::cout << " Number of Events = " << nevt << std::endl ;
   
   cout << " We will print out some information for the 1st event" << endl ;
   
   
   edm::HepMCProduct EvtProd ;
   TBranch* bhepmc =
      tevt->GetBranch( "edmHepMCProduct_source__Gen.obj") ;

   bhepmc->SetAddress( & EvtProd ) ;
   
   HepMC::GenEvent*    Evt = 0 ;
   HepMC::GenVertex*   Vtx = 0 ;
   HepMC::GenParticle* Part = 0 ;
   int                 NVtx = 0 ;
   int                 NPrt = 0 ;
   
   HepMC::GenVertex*   HiggsDecVtx = 0 ;
   
   HepMC::GenEvent::particle_iterator pit ;
   HepMC::GenEvent::vertex_iterator vit ;
   HepMC::GenVertex::particle_iterator vpit ;
      
   // Somehow I'm having problems when trying to store (pointers to) selected muons
   // in a containers std::vector<HepMC::GenParticle*> (despite existing dictionary...),
   // thus I'm doing a dirty trick of storing momenta in one container (std::vector) 
   // and the PDG's (for checking charge) in another (TArrayI)
   //
   // I'm still using vector of CLHEP 4-vectors because there's NO dictionary for
   // std::vector<HepMC::FourVector>
   // I'll fix it later   
   //
   std::vector<CLHEP::HepLorentzVector> StableMuMom ;
      
   //
   // I'm reserving it for 20 int's... I wish it was automatic
   //
   TArrayI StablePDG(20) ;
   
   TH1D* Hist2muMass = new TH1D( "Hist2muMass", "test 2-mu mass", 100,  60., 120. ) ;
   TH1D* Hist4muMass = new TH1D( "Hist4muMass", "test 4-mu mass", 100, 170., 210. ) ;
   

   for ( int iev=0; iev<nevt; iev++ )
   {

      bhepmc->GetEntry(iev) ;
      
      Evt = EvtProd->GetEvent() ;

      if ( iev == 0 ) Evt->print() ;       

      HiggsDecVtx = 0 ;
      StableMuMom.clear() ;
      
      for ( pit=Evt->particles_begin(); pit!=Evt->particles_end(); pit++)
      {
	 Part = (*pit) ;
	 if ( Part->pdg_id() == 25 )
	 {
	    HiggsDecVtx = Part->end_vertex() ;
	    if ( HiggsDecVtx != 0 )
	    {
	       if ( iev == 0 ) cout << " Found Higgs with valid decay vertex : " 
	                            << HiggsDecVtx->barcode() 
		                    << " in event " << iev << endl ;
	       break ;
	    }
	 }
      }

      if ( HiggsDecVtx == 0 )
      {
         cout << "There is NO Higgs in the event " << iev << endl ;
      }
      else
      {
         for ( pit=Evt->particles_begin(); pit!=Evt->particles_end(); pit++)
	 {
	    Part = (*pit) ;
	    if ( abs(Part->pdg_id()) == 13 && Part->status() == 1 )
	    {
	       // here's the "dirty trick" itself, storing info
	       // in 2 differnt containers...
	       //
	       StableMuMom.push_back( CLHEP::HepLorentzVector(Part->momentum().px(),
	                                                      Part->momentum().py(),
							      Part->momentum().pz(),
							      Part->momentum().e() ) ) ;
	       StablePDG[StableMuMom.size()-1] = Part->pdg_id() ;
	    } 
	 }
      }
      
      if ( iev == 0 ) cout << " Found " << StableMuMom.size() << " stable muons" << endl ;
      
      HepMC::FourVector Mom2mu ;
                  
      for ( unsigned int i=0; i<StableMuMom.size(); i++ )
      {
	 for ( unsigned int j=i+1; j<StableMuMom.size(); j++ )
	 {
	    if ( StablePDG.At(i)*StablePDG.At(j) > 0 ) continue ; // skip same charge pairs
	    if ( iev == 0 )
	    {
	       cout << " Stable particles id-s: " << StablePDG.At(i) << " " 
	                                          << StablePDG.At(j) << endl ;
            }				  
	    
	    double XMass2mu = HepMC::FourVector( (StableMuMom[i].px()+StableMuMom[j].px()),
	                                (StableMuMom[i].py()+StableMuMom[j].py()),
		                        (StableMuMom[i].pz()+StableMuMom[j].pz()),
			                (StableMuMom[i].e() +StableMuMom[j].e()) ).m() ;
	    
	    if ( iev == 0 ) cout << " 2-mu inv.mass = " << XMass2mu << endl ;
	    Hist2muMass->Fill( XMass2mu ) ;
	 }
      } 
      
      if ( StableMuMom.size() == 4 )
      {
         double px, py, pz, e ;
	 px=py=pz=e=0. ;
	 for ( unsigned int i=0; i<4; i++ )
	 {
	    //Mom4mu += StableMuMom[i] ;
	    px += StableMuMom[i].px() ;
	    py += StableMuMom[i].py() ;
	    pz += StableMuMom[i].pz() ;
	    e  += StableMuMom[i].e() ;
	 }
	 double XMass4mu = HepMC::FourVector( px, py, pz, e ).m() ;
	 Hist4muMass->Fill( XMass4mu ) ;
         if ( iev == 0 ) cout << " 4-mu inv.mass = " << XMass4mu << endl ;                    
      }
   }
   
   TCanvas* cnv = new TCanvas("cnv") ;
   
   cnv->Divide(2,1) ;
   
   cnv->cd(1) ;
   Hist2muMass->Draw() ;
   cnv->cd(2) ;
   Hist4muMass->Draw() ;
   
   
}
