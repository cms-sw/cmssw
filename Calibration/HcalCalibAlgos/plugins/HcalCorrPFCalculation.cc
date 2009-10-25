//#include "Validation/HcalCorrPF/interface/HcalCorrPFCalculation.h"
#include "Calibration/HcalCalibAlgos/plugins/HcalCorrPFCalculation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


HcalCorrPFCalculation::HcalCorrPFCalculation(edm::ParameterSet const& conf) {
  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will NOT be saved";
  }
  
  nevtot = 0;
  
  dbe_ = 0;
  // get hold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
   
  Char_t histo[20];

  hcalselector_ = conf.getUntrackedParameter<std::string>("hcalselector", "all");
  ecalselector_ = conf.getUntrackedParameter<std::string>("ecalselector", "yes");
  eventype_     = conf.getUntrackedParameter<std::string>("eventype", "single");
  sign_         = conf.getUntrackedParameter<std::string>("sign", "*");
  mc_           = conf.getUntrackedParameter<std::string>("mc", "yes");
  Respcorr_        = conf.getUntrackedParameter<bool>("RespcorrAdd", false);
  PFcorr_        = conf.getUntrackedParameter<bool>("PFcorrAdd", false);
  Conecorr_        = conf.getUntrackedParameter<bool>("ConeCorrAdd", true);
  famos_        = conf.getUntrackedParameter<bool>("Famos", false);
  radius_       = conf.getUntrackedParameter<double>("ConeRadiusCm", 40.);
  //  std::cout << "*** famos_ = " << famos_ << std::endl; 

  subdet_ = 5;
  if (hcalselector_ == "noise") subdet_ = 0;
  if (hcalselector_ == "HB"   ) subdet_ = 1;
  if (hcalselector_ == "HE"   ) subdet_ = 2;
  if (hcalselector_ == "HO"   ) subdet_ = 3;
  if (hcalselector_ == "HF"   ) subdet_ = 4;
  if (hcalselector_ == "all"  ) subdet_ = 5;
  if (hcalselector_ == "ZS"   ) subdet_ = 6;

  etype_ = 1;
  if (eventype_ == "multi") etype_ = 2;

  iz = 1;
  if(sign_ == "-") iz = -1;
  if(sign_ == "*") iz = 0;

  imc = 1;
  if(mc_ == "no") imc = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("HcalCorrPF");

      
  //======================= PFcorr related profiles. ===================

    if(subdet_ != 0 && imc != 0) { 
      
      sprintf (histo, "HcalCorrPF_Ncells_vs_ieta_cone%icm",int(radius_));
      meEnConeEtaProfile_depth1 = dbe_->bookProfile(histo, histo, 83, -41.5, 41.5, 1000, 0., 1000.);   

      sprintf (histo, "HcalCorrPF_Ncells_vs_ieta_Noise_cone%icm",int(radius_));
      meEnConeEtaProfile_depth1Noise = dbe_->bookProfile(histo, histo, 83, -41.5, 41.5, 1000, 0., 1000.);   
      
      sprintf (histo, "HcalCorrPF_En_rechits_cone%icm_profile_vs_ieta_all_depths_woHO",int(radius_));
      meEnConeEtaProfile = dbe_->bookProfile(histo, histo, 83, -41.5, 41.5, 210, -10., 200.);  

      sprintf (histo, "HcalCorrPF_En_rechits_cone%icm_profile_vs_ieta_all_depths_woHO_Noise",int(radius_));
      meEnConeEtaProfileNoise = dbe_->bookProfile(histo, histo, 83, -41.5, 41.5, 210, -10., 200.);  


    }
    
  //======================= End of PFcorr related profiles. ===================

  }  //end-of if(_dbe) 

}


HcalCorrPFCalculation::~HcalCorrPFCalculation() {

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
  
}

void HcalCorrPFCalculation::endJob() { }

void HcalCorrPFCalculation::beginJob(const edm::EventSetup& c){
  edm::ESHandle<MagneticField> bField;
  c.get<IdealMagneticFieldRecord>().get(bField);
  stepPropF  = new SteppingHelixPropagator(&*bField,alongMomentum);
  stepPropF->setMaterialMode(false);
  stepPropF->applyRadX0Correction(true);


}
void HcalCorrPFCalculation::analyze(edm::Event const& ev, edm::EventSetup const& c) {

  using namespace edm;
 
  GlobalPoint barrelMC(0,0,0);
  GlobalPoint endcapMC(0,0,0);
  GlobalPoint forwardMC1(0,0,0);
  GlobalPoint forwardMC2(0,0,0);

  double eta_bin[42]={0.,.087,.174,.261,.348,.435,.522,.609,.696,.783,
  .870,.957,1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.740,1.830,1.930,2.043,2.172,
  2.322,2.500,2.650,2.853,3.000,3.139,3.314,3.489,3.664,3.839,4.013,4.191,4.363,4.538,4.716,4.889,5.191};

  // Response corrections w/o re-rechitting
  hasresp=0;
  //  const HcalRespCorrs* myRecalib;
  try{

  edm::ESHandle <HcalRespCorrs> recalibCorrs;
  c.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
  myRecalib = recalibCorrs.product();

  edm::ESHandle <HcalPFCorrs> pfCorrs;
  c.get<HcalPFCorrsRcd>().get("recalibrate",pfCorrs);
  pfRecalib = pfCorrs.product();

  hasresp=1;
  }catch(const cms::Exception & e) {
//    errMsg = errMsg + "  -- No  recalibrate\n" + e.what();
  }


  // cuts for each subdet_ector mimiking  "Scheme B"
  //  double cutHB = 0.9, cutHE = 1.4, cutHO = 1.1, cutHFL = 1.2, cutHFS = 1.8; 

  // energy in HCAL
  double eHcal        = 0.;
  // Total numbet of RecHits in HCAL, in the cone, above 1 GeV theshold
  int nrechits       = 0;
  int nrechitsThresh = 0;

  // energy in ECAL
  double eEcalCone   = 0.;
  int numrechitsEcal = 0;

  // MC info 
  double phi_MC = -999999.;  // phi of initial particle from HepMC
  double eta_MC = -999999.;  // eta of initial particle from HepMC
  double mom_MC = 50.;  // P of initial particle from HepMC
  bool MC = false;

  // HCAL energy around MC eta-phi at all depths;
  double partR = 0.3;

  // Cone size for serach of the hottest HCAL cell around MC
  double searchR = 1.0; 

  // Single particle samples: actual eta-phi position of cluster around
  // hottest cell
  double etaHot  = 99999.; 
  double phiHot  = 99999.; 
  int    ietahot = 1000;
  int    iphihot = 1000;

  // MC information

  //  std::cout << "*** 1" << std::endl; 


  if(imc != 0) { 

  edm::Handle<edm::HepMCProduct> evtMC;
  //  ev.getByLabel("VtxSmeared",evtMC);
  ev.getByLabel("generator",evtMC);
  if (!evtMC.isValid()) {
    std::cout << "no HepMCProduct found" << std::endl;    
  } else {
    MC=true;
    //    std::cout << "*** source HepMCProduct found"<< std::endl;
  }  

  // MC particle with highest pt is taken as a direction reference  
  double maxPt = -99999.;
  int npart    = 0;

  GlobalPoint pos (0,0,0);

  HepMC::GenEvent * myGenEvent = new  HepMC::GenEvent(*(evtMC->GetEvent()));
  for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) {
    double phip = (*p)->momentum().phi();
    double etap = (*p)->momentum().eta();
    double pt  = (*p)->momentum().perp();
    mom_MC = (*p)->momentum().mag();
    if(pt > maxPt) { npart++; maxPt = pt; phi_MC = phip; eta_MC = etap; }
    GlobalVector mom ((*p)->momentum().x(),(*p)->momentum().y(),(*p)->momentum().z());
    int charge = -1;

    if(abs((*p)->pdg_id())==211) charge = (*p)->pdg_id()/abs((*p)->pdg_id()); // pions only !!!

    const FreeTrajectoryState *freetrajectorystate_ =
       new FreeTrajectoryState(pos, mom ,charge , &(*theMagField));

    if(fabs(etap)<1.7) {
       Cylinder *cylinder = new Cylinder(Surface::PositionType(0,0,0),
//                                    Surface::RotationType(), 177.7);  // a different thing ...
//                                    Surface::RotationType(), 190);    // obsolated ...
                                    Surface::RotationType(), 181.1);

       TrajectoryStateOnSurface steppingHelixstateinfo_ = stepPropF->propagate(*freetrajectorystate_, (*cylinder));
       if(steppingHelixstateinfo_.isValid() ) {
       barrelMC = steppingHelixstateinfo_.freeState()->position();
       }
       
    }

    int doHF=0;
    if(fabs(etap)>1.2&&fabs(etap)<10.) {

	Surface::RotationType rot(GlobalVector(1,0,0),GlobalVector(0,1,0));
	
	  Surface::PositionType pos(0., 0.,400.5*fabs(etap)/etap);
	  PlaneBuilder::ReturnType aPlane = PlaneBuilder().plane(pos,rot);


       TrajectoryStateOnSurface steppingHelixstateinfo_ = stepPropF->propagate(*freetrajectorystate_, (*aPlane));
       if(steppingHelixstateinfo_.isValid() && fabs(steppingHelixstateinfo_.freeState()->position().eta())<3.0 ) 
       endcapMC = steppingHelixstateinfo_.freeState()->position();
       if(steppingHelixstateinfo_.isValid() && fabs(steppingHelixstateinfo_.freeState()->position().eta())>3.0 )
       doHF=1;
    }
    if(doHF) {

 	  Surface::RotationType rot(GlobalVector(1,0,0),GlobalVector(0,1,0));
	
	  Surface::PositionType pos1(0., 0.,1115*fabs(etap)/etap);
	  Surface::PositionType pos2(0., 0.,1137*fabs(etap)/etap);
	  PlaneBuilder::ReturnType aPlane1 = PlaneBuilder().plane(pos1,rot);
	  PlaneBuilder::ReturnType aPlane2 = PlaneBuilder().plane(pos2,rot);


       TrajectoryStateOnSurface steppingHelixstateinfo_ = stepPropF->propagate(*freetrajectorystate_, (*aPlane1));
       if(steppingHelixstateinfo_.isValid() ) 
       forwardMC1 = steppingHelixstateinfo_.freeState()->position();

       steppingHelixstateinfo_ = stepPropF->propagate(*freetrajectorystate_, (*aPlane2));
       if(steppingHelixstateinfo_.isValid() ) 
       forwardMC2 = steppingHelixstateinfo_.freeState()->position();
    }

  }
  //  std::cout << "*** Max pT = " << maxPt <<  std::endl;  

  }

// MC_ieta calculation - for the PFcorr profiles
  int MC_ieta = -100;
  if(barrelMC.mag()>1.&&fabs(barrelMC.eta())<1.392) {
   for(int ie=0;ie<17;ie++)
   if(fabs(barrelMC.eta())>eta_bin[ie]&&fabs(barrelMC.eta())<eta_bin[ie+1]) MC_ieta = ie+1;
  }else if(endcapMC.mag()>1.&&fabs(endcapMC.eta())<3.0&&fabs(endcapMC.eta())>1.392) {
   for(int ie=15;ie<30;ie++)
   if(fabs(endcapMC.eta())>eta_bin[ie]&&fabs(endcapMC.eta())<eta_bin[ie+1]) MC_ieta = ie+1;
  }else if(forwardMC1.mag()>1&&fabs(forwardMC1.eta())>3.0) {
   for(int ie=28;ie<42;ie++)
   if(fabs(forwardMC1.eta())>eta_bin[ie]&&fabs(forwardMC1.eta())<eta_bin[ie+1]) MC_ieta = ie+1;
  }
// end of MC_ieta

  if(eta_MC<0) MC_ieta = - MC_ieta ;

  c.get<CaloGeometryRecord>().get (geometry);

  // Fill working vectors of HCAL RecHits quantities 
  fillRecHitsTmp(subdet_, ev); 


  //===========================================================================
  // IN ALL other CASES : ieta-iphi maps 
  //===========================================================================

  // ECAL 
  if(ecalselector_ == "yes" && (subdet_ == 1 || subdet_ == 2 || subdet_ == 5)) {
    Handle<EBRecHitCollection> rhitEB;

    if(famos_)
      ev.getByLabel("caloRecHits","EcalRecHitsEB", rhitEB);
    else
      ev.getByLabel("ecalRecHit","EcalRecHitsEB", rhitEB);

    EcalRecHitCollection::const_iterator RecHit = rhitEB.product()->begin();  
    EcalRecHitCollection::const_iterator RecHitEnd = rhitEB.product()->end();  
    
    for (; RecHit != RecHitEnd ; ++RecHit) {
      EBDetId EBid = EBDetId(RecHit->id());
       
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (EBid)->getGeometry (EBid) ;
      double eta = cellGeometry->getPosition ().eta () ;
      double phi = cellGeometry->getPosition ().phi () ;
      double en  = RecHit->energy();

      double r   = dR(eta_MC, phi_MC, eta, phi);
      if( r < partR)  {
	eEcalCone += en;
	numrechitsEcal++; 
      }
    }

    
    Handle<EERecHitCollection> rhitEE;
 
    if(famos_)
      ev.getByLabel("caloRecHits", "EcalRecHitsEE", rhitEE );
    else
      ev.getByLabel("ecalRecHit","EcalRecHitsEE", rhitEE);

    RecHit = rhitEE.product()->begin();  
    RecHitEnd = rhitEE.product()->end();  
    
    for (; RecHit != RecHitEnd ; ++RecHit) {
      EEDetId EEid = EEDetId(RecHit->id());
      
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (EEid)->getGeometry (EEid) ;
      double eta = cellGeometry->getPosition ().eta () ;
      double phi = cellGeometry->getPosition ().phi () ;	
      double en   = RecHit->energy();

      double r   = dR(eta_MC, phi_MC, eta, phi);
      if( r < partR)  {
	eEcalCone += en;
	numrechitsEcal++; 
      }
    }
  }     // end of ECAL selection 


  // Filling HCAL maps  ----------------------------------------------------
  double maxE = -99999.;
  
  for (unsigned int i = 0; i < cen.size(); i++) {
    
    int ieta   = cieta[i]; 
    int iphi   = ciphi[i]; 
    double en  = cen[i]; 
    double eta = ceta[i]; 
    double phi = cphi[i]; 
    
    double emin = 1.;
    if(fabs(eta) > 3.) emin = 5.; 
    
    double r  = dR(eta_MC, phi_MC, eta, phi);
    if( r < searchR ) { // search for hottest cell in a big cone
      if(maxE < en && en > emin) {
	maxE    = en;
	etaHot  = eta;
	phiHot  = phi;
	ietahot = ieta;
	iphihot = iphi;
      }
    }
    
  } // end cells loop


  //===========================================================================
  // SUBSYSTEMS,  
  //===========================================================================
  
  if ((subdet_ != 6) && (subdet_ != 0)) {

    double HcalCone    = 0.;
    double HcalConeNoise    = 0.;

    int ietaMax   =  9999;
    double enMax  = -9999.;


    //   CYCLE over cells ====================================================
    //  std::cout<<"cen size"<<cen.size()<<std::endl;
    int UsedCells = 0;
    int UsedCellsNoise = 0;
    for (unsigned int i = 0; i < cen.size(); i++) {
      int depth  = cdepth[i];
      double eta = ceta[i]; 
      double phi = cphi[i]; 
      double en  = cen[i]; 
      int   ieta = cieta[i];
      double xc = cx[i];
      double yc = cy[i];
      double zc = cz[i];


          GlobalPoint gposC(xc,yc,zc);
          if(abs(ieta)>16&&fabs(zc)<1000)
           {
             double scale = 400.5/fabs(zc);
             gposC = GlobalPoint (xc*scale,yc*scale,zc*scale); // Fold HE to the front face d1
           }

          if(abs(ieta)<17)
          {
            double scale = 181.1/gposC.perp(); 
             gposC = GlobalPoint (xc*scale,yc*scale,zc*scale); // Fold HB to the front face d1
          }

          int usecell = 0;
          int usecellNoise = 0;

///  dR corrections
          if(Conecorr_) {   // shift cells along IP to do dR on Sphere
            double xcc = gposC.x();
            double ycc = gposC.y();
            double zcc = gposC.z();
            if(barrelMC.mag()>1.&&fabs(barrelMC.eta())<1.392) {
              double scale = barrelMC.mag()/gposC.mag();
              gposC = GlobalPoint (xcc*scale,ycc*scale,zcc*scale); 
            }
            if(endcapMC.mag()>1.&&fabs(endcapMC.eta())>1.392&&fabs(endcapMC.eta())<3.000) {
              double scale = endcapMC.mag()/gposC.mag();
              gposC = GlobalPoint (xcc*scale,ycc*scale,zcc*scale); // Fold HB to the front face d1
            }
          // For HF is not necessary...
          }  // end of dR corrections

          if((gposC-barrelMC).mag()<radius_) usecell = 1;
          if((gposC-endcapMC).mag()<radius_) usecell = 1;
          if((gposC-forwardMC1).mag()<radius_&&depth==1) usecell = 1;
          if((gposC-forwardMC2).mag()<radius_&&depth==2) usecell = 1;

            double xcc = -gposC.x();
            double ycc = -gposC.y();
            double zcc = -gposC.z();
              gposC = GlobalPoint (xcc,ycc,zcc); // Fold HB to the front face d1

          if((gposC-barrelMC).mag()<radius_) usecellNoise = 1;
          if((gposC-endcapMC).mag()<radius_) usecellNoise = 1;
          if((gposC-forwardMC1).mag()<radius_&&depth==1) usecellNoise = 1;
          if((gposC-forwardMC2).mag()<radius_&&depth==2) usecellNoise = 1;

           if(usecell) HcalCone += en;
           if(usecellNoise) HcalConeNoise += en;
           if(usecell) UsedCells++;
           if(usecellNoise) UsedCellsNoise++;


      nrechits++;	    
      eHcal += en;
      if(en > 1. ) nrechitsThresh++;

      double r    = dR(eta_MC, phi_MC, eta, phi);
      if( r < partR ){
	if(enMax   < en) {
	  enMax   = en;
	  ietaMax = ieta;
	}
      }
    }

    if(imc != 0) {
      if(eEcalCone<1.0||(abs(ietaMax)>16&&abs(ietaMax)<30&&eEcalCone<4.))
       {
      meEnConeEtaProfile       ->Fill(double(MC_ieta),  HcalCone);
      meEnConeEtaProfile_depth1->Fill(double(MC_ieta),  double(UsedCells));
      meEnConeEtaProfileNoise       ->Fill(double(-MC_ieta),  HcalConeNoise);
      meEnConeEtaProfile_depth1Noise->Fill(double(-MC_ieta),  double(UsedCellsNoise));
       }
    }
  }

  nevtot++;
}


///////////////////////////////////////////////////////////////////////////////
void HcalCorrPFCalculation::fillRecHitsTmp(int subdet_, edm::Event const& ev){
  
  using namespace edm;
  
  
  // initialize data vectors
  csub.clear();
  cen.clear();
  ceta.clear();
  cphi.clear();
  ctime.clear();
  cieta.clear();
  ciphi.clear();
  cdepth.clear();
  cx.clear();
  cy.clear();
  cz.clear();


  if( subdet_ == 1 || subdet_ == 2  || subdet_ == 5 || subdet_ == 6 || subdet_ == 0) {
    
    //HBHE
    std::vector<edm::Handle<HBHERecHitCollection> > hbhecoll;
    ev.getManyByType(hbhecoll);

    std::vector<edm::Handle<HBHERecHitCollection> >::iterator i;
    
    int count = 0;
    for (i=hbhecoll.begin(); i!=hbhecoll.end(); i++) {
      
      count ++;  
    //      std::cout << "*** HBHE collection No. " <<  count << std::endl;     
      if ( count == 1) {
      for (HBHERecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	HcalDetId cell(j->id());

      float icalconst=1.;
      float pfconst=1.;
      DetId mydetid = cell.rawId();
      if(hasresp) {
        if(Respcorr_)icalconst=myRecalib->getValues(mydetid)->getValue();
        if(PFcorr_)pfconst=pfRecalib->getValues(mydetid)->getValue();
      }

	const CaloCellGeometry* cellGeometry =
	  geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	double eta  = cellGeometry->getPosition().eta () ;
	double phi  = cellGeometry->getPosition().phi () ;
        double xc   = cellGeometry->getPosition().x ();
        double yc   = cellGeometry->getPosition().y ();
        double zc   = cellGeometry->getPosition().z ();
	int sub     = cell.subdet();
	int depth   = cell.depth();
	int inteta  = cell.ieta();
	int intphi  = cell.iphi()-1;
	double en   = j->energy()*icalconst*pfconst;
	double t    = j->time();

	if((iz > 0 && eta > 0.) || (iz < 0 && eta <0.) || iz == 0) { 
	
	  csub.push_back(sub);
	  cen.push_back(en);
	  ceta.push_back(eta);
	  cphi.push_back(phi);
	  ctime.push_back(t);
	  cieta.push_back(inteta);
	  ciphi.push_back(intphi);
	  cdepth.push_back(depth);
	  cx.push_back(xc);
	  cy.push_back(yc);
	  cz.push_back(zc);
	}
      }

      }
    }
  }

  if( subdet_ == 4 || subdet_ == 5 || subdet_ == 6 || subdet_ == 0) {

    //HF
    std::vector<edm::Handle<HFRecHitCollection> > hfcoll;
    ev.getManyByType(hfcoll);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator ihf;

    int count = 0;
    for (ihf=hfcoll.begin(); ihf!=hfcoll.end(); ihf++) {      
      count++;
      if(count == 1) {
      for (HFRecHitCollection::const_iterator j=(*ihf)->begin(); j!=(*ihf)->end(); j++) {
	HcalDetId cell(j->id());

      float icalconst=1.;
      float pfconst=1.;
      DetId mydetid = cell.rawId();
      if(hasresp)
      {
       if(Respcorr_)icalconst=myRecalib->getValues(mydetid)->getValue();
       if(PFcorr_) pfconst=pfRecalib->getValues(mydetid)->getValue();
      }

	const CaloCellGeometry* cellGeometry =
	  geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	double eta   = cellGeometry->getPosition().eta () ;
	double phi   = cellGeometry->getPosition().phi () ;
        double xc     = cellGeometry->getPosition().x ();
        double yc     = cellGeometry->getPosition().y ();
        double zc     = cellGeometry->getPosition().z ();
	int sub      = cell.subdet();
	int depth    = cell.depth();
	int inteta   = cell.ieta();
	int intphi   = cell.iphi()-1;
	double en    = j->energy()*icalconst*pfconst;
	double t     = j->time();

	if((iz > 0 && eta > 0.) || (iz < 0 && eta <0.) || iz == 0) { 
	
	  csub.push_back(sub);
	  cen.push_back(en);
	  ceta.push_back(eta);
	  cphi.push_back(phi);
	  ctime.push_back(t);
	  cieta.push_back(inteta);
	  ciphi.push_back(intphi);
	  cdepth.push_back(depth);
	  cx.push_back(xc);
	  cy.push_back(yc);
	  cz.push_back(zc);
       
	}
      }
      }
    }
  }

  
}

double HcalCorrPFCalculation::dR(double eta1, double phi1, double eta2, double phi2) { 
  double PI = 3.1415926535898;
  double deltaphi= phi1 - phi2;
  if( phi2 > phi1 ) { deltaphi= phi2 - phi1;}
  if(deltaphi > PI) { deltaphi = 2.*PI - deltaphi;}
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta* deltaeta + deltaphi*deltaphi);
  return tmp;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalCorrPFCalculation);

