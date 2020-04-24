
{

   gSystem->Load("libFWCoreFWLite") ;
   FWLiteEnabler::enable() ;
   
   TFile* f = new TFile("pgun.root") ;
   
   TTree* tevt = f->Get("Events") ;
   
   int nevt = tevt->GetEntries() ;
   std::cout << "Number of Events = " << nevt << std::endl ;

   edm::HepMCProduct EvtProd ;
   TBranch* bhepmc =
      tevt->GetBranch( "edmHepMCProduct_source__PROD.obj") ;

   bhepmc->SetAddress( & EvtProd ) ;
   
   HepMC::GenEvent*  Evt = 0 ;
   HepMC::GenVertex* Vtx = 0 ;
   int               NVtx = 0 ;
   TH1D*             VtxZSpread = new TH1D("VtxZSpread", "test", 
                                            50, -250., 250. ) ;
   
   for ( int iev=0; iev<nevt; iev++ )
   {
      bhepmc->GetEntry(iev) ;
      Evt = EvtProd->GetEvent() ;
      NVtx = Evt->vertices_size() ;
      for ( int iv=-1; iv>=-NVtx; iv-- )
      {
         Vtx = Evt->barcode_to_vertex( iv ) ;
         // std::cout << " Z = " << Vtx->position().z() << std::endl ; 
	 VtxZSpread->Fill( Vtx->position().z() ) ;
      }
   }
   
   VtxZSpread->Draw() ;
   
}
