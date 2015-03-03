#include "../interface/RetinaTrackFitter.h"

const double RetinaTrackFitter::rot_angle[8] = {  0.39269908169872414,   //  pi/8
						  -0.39269908169872414,   // -pi/8
						  -1.17809724509617242,   // -3/8 pi
						  -1.96349540849362070,   // -5/8 pi
						  -2.74889357189106898,   // -7/8 pi
						  -3.53429173528851726,   // -9/8 pi
						  -4.31968989868596509,   // -11/8 pi
						  -5.10508806208341426 }; // -13/8 pi

RetinaTrackFitter::RetinaTrackFitter():TrackFitter(0)
{
  verboseLevel  = 0;
  event_counter = 0;
  road_id       = 0;

  initialize();
}

RetinaTrackFitter::RetinaTrackFitter(int nb):TrackFitter(nb)
{
  verboseLevel  = 0;
  event_counter = 0;
  road_id       = 0;

  initialize();
}

RetinaTrackFitter::~RetinaTrackFitter(){
}


void RetinaTrackFitter::fit(vector<Hit*> hits_){
  if ( hits_.size()>1024 ){
    cout << "*** ERROR in RetinaTrackFitter::fit() at event/road = " << event_counter << "/" 
	 << road_id << ": too many stubs for fitting, fit aborted!" << endl;
    return;
  }

  if ( hits_.size()<(unsigned int) nb_layers ){
    cout << "*** ERROR in RetinaTrackFitter::fit() at event/road = " << event_counter << "/" 
	 << road_id << ": only " << hits_.size() << " stubs found, fit aborted!" << endl;
    return;
  }


  // --- Constants used in X+-X- transformation:
  double y0 =  0.0217391304347826081; // 0.5/23.
  double y1 =  0.0046296296296296294; // 0.5/108.


  // --- Determine in which phi sector and eta range the trigger tower is:
  const int phi_sector = sector_id % 8;
  const int eta_range  = sector_id / 8;

  int trigTow_type = 0;
  if ( eta_range==1 || eta_range==4 )
    trigTow_type = 1;
  else if ( eta_range==0 || eta_range==5 )
    trigTow_type = 2;


  // --- Fill the stubs vector:
  vector <Hit_t*> hits;
  vector <Hit_t*> hits_RZ;

  for(unsigned int ihit=0; ihit<hits_.size(); ihit++){
  
    Hit_t* hit = new Hit_t();
    hit->x     = hits_[ihit]->getX();
    hit->y     = hits_[ihit]->getY();
    // NB: We flip the z sign for the trigger towers with negative eta in
    //     order to use the same retina setup as for eta > 0 
    hit->z     = ( eta_range > 2 ? hits_[ihit]->getZ() : -hits_[ihit]->getZ() );
    hit->rho   = sqrt( hits_[ihit]->getX()*hits_[ihit]->getX() +
		       hits_[ihit]->getY()*hits_[ihit]->getY() );
    hit->layer = (short) hits_[ihit]->getLayer();
    hit->id    = hits_[ihit]->getID();
    
    hits.push_back(hit);

    if (verboseLevel==2 )
      cout << ihit << "  -  " 
    	   << " x = " << hits_[ihit]->getX() << "   "
    	   << " y = " << hits_[ihit]->getY() << "   "
    	   << " z = " << hits_[ihit]->getZ() << " ---> " 
    	   << " R = " << sqrt(hits_[ihit]->getX()*hits_[ihit]->getX()+
    			      hits_[ihit]->getY()*hits_[ihit]->getY())
    	   << "  -  id = " <<  hits_[ihit]->getID()
    	   << endl;

  }



  // ===========================================================================
  //  XY fit
  // ===========================================================================

  // --- Phi sector rotation:
  rotateHits(hits, rot_angle[phi_sector]);

  // --- Conformal transformation: 
  confTrans(hits);

  //
  // --- First step ------------------------------------------------------------
  //

  // --- Setup the retina:
  double pbins_step1 = config[trigTow_type]["xy_pbins_step1"];
  double qbins_step1 = config[trigTow_type]["xy_qbins_step1"];
  double pmin_step1  = config[trigTow_type]["xy_pmin_step1"];
  double pmax_step1  = config[trigTow_type]["xy_pmax_step1"];
  double qmin_step1  = config[trigTow_type]["xy_qmin_step1"];
  double qmax_step1  = config[trigTow_type]["xy_qmax_step1"];
  
  double minWeight_step1 = config[trigTow_type]["xy_threshold_step1"];

  double pstep_step1 = (pmax_step1-pmin_step1)/pbins_step1;
  double qstep_step1 = (qmax_step1-qmin_step1)/qbins_step1;

  vector <double> sigma_step1(8,sqrt(pstep_step1*pstep_step1+qstep_step1*qstep_step1));
  if ( config[trigTow_type]["xy_sigma1_step1"] != 0. ) 
    sigma_step1[0] = config[trigTow_type]["xy_sigma1_step1"];
  if ( config[trigTow_type]["xy_sigma2_step1"] != 0. ) 
    sigma_step1[1] = config[trigTow_type]["xy_sigma2_step1"];
  if ( config[trigTow_type]["xy_sigma3_step1"] != 0. ) 
    sigma_step1[2] = config[trigTow_type]["xy_sigma3_step1"];
  if ( config[trigTow_type]["xy_sigma4_step1"] != 0. ) 
    sigma_step1[3] = config[trigTow_type]["xy_sigma4_step1"];
  if ( config[trigTow_type]["xy_sigma5_step1"] != 0. ) 
    sigma_step1[4] = config[trigTow_type]["xy_sigma5_step1"];
  if ( config[trigTow_type]["xy_sigma6_step1"] != 0. ) 
    sigma_step1[5] = config[trigTow_type]["xy_sigma6_step1"];
  if ( config[trigTow_type]["xy_sigma7_step1"] != 0. ) 
    sigma_step1[6] = config[trigTow_type]["xy_sigma7_step1"];
  if ( config[trigTow_type]["xy_sigma8_step1"] != 0. ) 
    sigma_step1[7] = config[trigTow_type]["xy_sigma8_step1"];


  Retina retinaXY_step1(hits, pbins_step1+2, qbins_step1+2, 
			pmin_step1-pstep_step1, pmax_step1+pstep_step1, 
			qmin_step1-qstep_step1, qmax_step1+qstep_step1, 
			sigma_step1, minWeight_step1, 1, XY);


  // --- Fill the retina and find maxima:
  retinaXY_step1.fillGrid();
  retinaXY_step1.findMaxima();
  if ( verboseLevel==1 )
    retinaXY_step1.dumpGrid(event_counter,1,0);
  if ( verboseLevel==2 )
    retinaXY_step1.printMaxima();


  // --- Get first step maxima:
  vector <pqPoint> maximaXY_step1 = retinaXY_step1.getMaxima();

  if ( maximaXY_step1.size()==0 && verboseLevel>0 ){
    cout << "*** WARNING in RetinaTrackFitter::fit() at event/road = " << event_counter << "/" 
	 << road_id << ": no maximum found in XY fit-step 1." << endl;
    retinaXY_step1.dumpGrid(event_counter,1,0);
  }

  if ( maximaXY_step1.size() > 10 ){
    cout << "*** ERROR in RetinaTrackFitter::fit() at event/road = " << event_counter 
	 << "/" << road_id << ": " << maximaXY_step1.size() 
	 << " maxima found in XY fit-step 1, fit aborted!" << endl;
    if ( verboseLevel>0 ) 
      retinaXY_step1.dumpGrid(event_counter,1,0);
    return;
  }


  //
  // --- Second step -----------------------------------------------------------
  //

  double pbins_step2 = config[trigTow_type]["xy_pbins_step2"];
  double qbins_step2 = config[trigTow_type]["xy_qbins_step2"];

  // --- Zoom around first step maxima:
  for (unsigned int imax=0; imax<maximaXY_step1.size(); ++imax){

    // --- Retina setup:
    double pmin_step2 = maximaXY_step1[imax].p - config[trigTow_type]["xy_zoom_step2"]*pstep_step1;
    double pmax_step2 = maximaXY_step1[imax].p + config[trigTow_type]["xy_zoom_step2"]*pstep_step1;
    double qmin_step2 = maximaXY_step1[imax].q - config[trigTow_type]["xy_zoom_step2"]*qstep_step1;
    double qmax_step2 = maximaXY_step1[imax].q + config[trigTow_type]["xy_zoom_step2"]*qstep_step1;
   
    double pstep_step2 = (pmax_step2-pmin_step2)/pbins_step2;
    double qstep_step2 = (qmax_step2-qmin_step2)/qbins_step2;
    
    double minWeight_step2 = config[trigTow_type]["xy_threshold_step2"];

    vector <double> sigma_step2(8,sqrt(pstep_step2*pstep_step2+qstep_step2*qstep_step2));
    if ( config[trigTow_type]["xy_sigma1_step2"] != 0. ) 
      sigma_step2[0] = config[trigTow_type]["xy_sigma1_step2"];
    if ( config[trigTow_type]["xy_sigma2_step2"] != 0. ) 
      sigma_step2[1] = config[trigTow_type]["xy_sigma2_step2"];
    if ( config[trigTow_type]["xy_sigma3_step2"] != 0. ) 
      sigma_step2[2] = config[trigTow_type]["xy_sigma3_step2"];
    if ( config[trigTow_type]["xy_sigma4_step2"] != 0. ) 
      sigma_step2[3] = config[trigTow_type]["xy_sigma4_step2"];
    if ( config[trigTow_type]["xy_sigma5_step2"] != 0. ) 
      sigma_step2[4] = config[trigTow_type]["xy_sigma5_step2"];
    if ( config[trigTow_type]["xy_sigma6_step2"] != 0. ) 
      sigma_step2[5] = config[trigTow_type]["xy_sigma6_step2"];
    if ( config[trigTow_type]["xy_sigma7_step2"] != 0. ) 
      sigma_step2[6] = config[trigTow_type]["xy_sigma7_step2"];
    if ( config[trigTow_type]["xy_sigma8_step2"] != 0. ) 
      sigma_step2[7] = config[trigTow_type]["xy_sigma8_step2"];


    Retina retinaXY_step2(hits, pbins_step2+2, qbins_step2+2, 
			  pmin_step2-pstep_step2, pmax_step2+pstep_step2, 
			  qmin_step2-qstep_step2, qmax_step2+qstep_step2, 
			  sigma_step2, minWeight_step2, 1, XY);


    // --- Fill the retina and find maxima:
    retinaXY_step2.fillGrid();
    retinaXY_step2.findMaxima();
    if ( verboseLevel==1 )
      retinaXY_step2.dumpGrid(event_counter,2,imax);
    if ( verboseLevel==2 )
      retinaXY_step2.printMaxima();


    // --- Get second step maxima:
    vector <pqPoint> maximaXY_step2 = retinaXY_step2.getMaxima();

    if ( maximaXY_step2.size()==0 && verboseLevel>0 ){
      cout << "*** WARNING in RetinaTrackFitter::fit() at event/road = " << event_counter << "/" 
	   << road_id << ": no maximum found in XY fit-step 2 (step-1 XY maximum #" << imax << ")." 
	   << endl;
      retinaXY_step2.dumpGrid(event_counter,2,imax);
    }

    if ( maximaXY_step2.size() > 10 ){
      cout << "*** ERROR in RetinaTrackFitter::fit() at event/road = " << event_counter 
	   << "/" << road_id << ": " << maximaXY_step2.size() 
	   << " maxima found in XY fit-step 2 (step-1 XY maximum #" << imax << "), fit aborted!" 
	   << endl;
      if ( verboseLevel>0 ) 
	retinaXY_step2.dumpGrid(event_counter,2,imax);
      continue;
    }


    // --- Loop over XY second step maxima:
    for (unsigned int itrk=0; itrk<maximaXY_step2.size(); ++itrk){

      // --- Invert the X+-X- transformation:
      double p = 0.5*(y1 - y0)/maximaXY_step2[itrk].q;
      double q = y0 - p*(maximaXY_step2[itrk].p-maximaXY_step2[itrk].q);


      // --- Associate stubs to this maximum:
      hits_RZ.clear();
      unsigned int n_stubsPS = 0;
      for (unsigned int ihit=0; ihit<hits.size(); ++ihit){
      
	double dist   = (p*hits[ihit]->x-hits[ihit]->y+q)/p;
	//double dist   = fabs(hits[ihit]->y-p*hits[ihit]->x-q)/sqrt(1.+p*p);
	double weight = exp(-0.5*dist*dist/(sigma_step2[0]*sigma_step2[0]));

	if ( weight > 0.5 ){
	  hits_RZ.push_back(hits[ihit]);
	  if ( hits[ihit]->rho > 60. )
	    n_stubsPS++;
	}
	else
	  if ( verboseLevel>0 )
	    cout << "*** WARNING in RetinaTrackFitter::fit() at event/road = " << event_counter 
		 << "/" << road_id << ": stub " << hits[ihit]->id << " with weight = " << weight 
		 << " has not been associated to the XY step-2 maximum #" << imax << "." << endl;
      }
      if ( hits_RZ.size() < 3 ){
	if ( verboseLevel>0 )
	  cout << "*** ERROR in RetinaTrackFitter::fit() at event/road = " << event_counter 
	       << "/" << road_id << ": only " <<  hits_RZ.size() 
	       << " stubs associated to the XY step-2 maximum #" << imax << ", fit aborted!" << endl;
	continue;
      }

      // --- Rotate back the original phi sector:
      q = q/(cos(rot_angle[phi_sector])+p*sin(rot_angle[phi_sector]));
      p = (p*cos(rot_angle[phi_sector])-sin(rot_angle[phi_sector]))/
        (cos(rot_angle[phi_sector])+p*sin(rot_angle[phi_sector]));


      // --- Invert the conformal transformation and get the track parameters:
      double a = -0.5*p/q;
      double b =  0.5/q;
    
      double c   = 1./sqrt(a*a+b*b);
      double phi = atan(p);
      //if ( phi<0. )
      //	phi += TMath::TwoPi();

      // =========================================================================
      //  RZ fit
      // =========================================================================

      y0 = 0.5/y0;
      y1 = 0.5/y1;

      double eta = -9999.;
      double z0  = -9999.;
    

      //
      // --- First step ----------------------------------------------------------
      //

      pbins_step1 = config[trigTow_type]["rz_pbins_step1"];
      qbins_step1 = config[trigTow_type]["rz_qbins_step1"];
      pmin_step1  = config[trigTow_type]["rz_pmin_step1"];
      pmax_step1  = config[trigTow_type]["rz_pmax_step1"];
      qmin_step1  = config[trigTow_type]["rz_qmin_step1"];
      qmax_step1  = config[trigTow_type]["rz_qmax_step1"];

      minWeight_step1 = config[trigTow_type]["rz_threshold_step1"];

      pstep_step1 = (pmax_step1-pmin_step1)/pbins_step1;
      qstep_step1 = (qmax_step1-qmin_step1)/qbins_step1;

      for (unsigned int ilayer=0; ilayer<8; ++ilayer)
	sigma_step1[ilayer] = sqrt(pstep_step1*pstep_step1+qstep_step1*qstep_step1);

      if ( config[trigTow_type]["rz_sigma1_step1"] != 0. ) 
	sigma_step1[0] = config[trigTow_type]["rz_sigma1_step1"];
      if ( config[trigTow_type]["rz_sigma2_step1"] != 0. ) 
	sigma_step1[1] = config[trigTow_type]["rz_sigma2_step1"];
      if ( config[trigTow_type]["rz_sigma3_step1"] != 0. ) 
	sigma_step1[2] = config[trigTow_type]["rz_sigma3_step1"];
      if ( config[trigTow_type]["rz_sigma4_step1"] != 0. ) 
	sigma_step1[3] = config[trigTow_type]["rz_sigma4_step1"];
      if ( config[trigTow_type]["rz_sigma5_step1"] != 0. ) 
	sigma_step1[4] = config[trigTow_type]["rz_sigma5_step1"];
      if ( config[trigTow_type]["rz_sigma6_step1"] != 0. ) 
	sigma_step1[5] = config[trigTow_type]["rz_sigma6_step1"];
      if ( config[trigTow_type]["rz_sigma7_step1"] != 0. ) 
	sigma_step1[6] = config[trigTow_type]["rz_sigma7_step1"];
      if ( config[trigTow_type]["rz_sigma8_step1"] != 0. ) 
	sigma_step1[7] = config[trigTow_type]["rz_sigma8_step1"];

      
      Retina retinaRZ_step1(hits_RZ, pbins_step1+2, qbins_step1+2, 
			    pmin_step1-pstep_step1, pmax_step1+pstep_step1, 
			    qmin_step1-qstep_step1, qmax_step1+qstep_step1, 
			    sigma_step1, minWeight_step1, 1, RZ);


      retinaRZ_step1.fillGrid();
      retinaRZ_step1.findMaxima();
      if ( verboseLevel==1 )
	retinaRZ_step1.dumpGrid(event_counter,1,imax);
      if ( verboseLevel==2 )
	retinaRZ_step1.printMaxima();


      // --- Get first step maximum:
      vector <pqPoint> maximaRZ_step1 = retinaRZ_step1.getMaxima();

      if ( maximaRZ_step1.size()==0 && verboseLevel>0 ){
	cout << "*** WARNING in RetinaTrackFitter::fit() at event/road = " << event_counter 
	     << "/" << road_id << ": no maximum found in RZ fit-step 1 (step-1 maximum #" 
	     << imax << ")." << endl;
	retinaXY_step2.dumpGrid(event_counter,2,imax);
      }

      if ( maximaRZ_step1.size() > 10 ){
	cout << "*** ERROR in RetinaTrackFitter::fit() at event/road = " << event_counter 
	     << "/" << road_id << ": " << maximaRZ_step1.size() 
	     << " maxima found in RZ fit-step 1 (step-1 maximum #" << imax << "), fit aborted!" 
	     << endl;
	if ( verboseLevel>0 ) 
	  retinaRZ_step1.dumpGrid(event_counter,1,imax);
	continue;
      }

      
      //
      // --- Second step ---------------------------------------------------------
      //

      pbins_step2 = config[trigTow_type]["rz_pbins_step2"];
      qbins_step2 = config[trigTow_type]["rz_qbins_step2"];

      // Zoom around first step maxima
      for (unsigned int imax_RZ=0; imax_RZ<maximaRZ_step1.size(); ++imax_RZ){

	double pmin_step2 = maximaRZ_step1[imax_RZ].p - config[trigTow_type]["rz_zoom_step2"]*pstep_step1;
	double pmax_step2 = maximaRZ_step1[imax_RZ].p + config[trigTow_type]["rz_zoom_step2"]*pstep_step1;
	double qmin_step2 = maximaRZ_step1[imax_RZ].q - config[trigTow_type]["rz_zoom_step2"]*qstep_step1;
	double qmax_step2 = maximaRZ_step1[imax_RZ].q + config[trigTow_type]["rz_zoom_step2"]*qstep_step1;
   
	double pstep_step2 = (pmax_step2-pmin_step2)/pbins_step2;
	double qstep_step2 = (qmax_step2-qmin_step2)/qbins_step2;
    
	double minWeight_step2 = config[trigTow_type]["rz_threshold_step2"];
	// If less than 3 PS stubs are available, adjust the maximum-finder threshold:
	if (  n_stubsPS < 3 )
	  minWeight_step2 *= 0.66;

	vector <double> sigma_step2(8,sqrt(pstep_step2*pstep_step2+qstep_step2*qstep_step2));
	for (unsigned int ilayer=3; ilayer<6; ++ilayer)
	  sigma_step2[ilayer] = 8.*sqrt(pstep_step2*pstep_step2+qstep_step2*qstep_step2);

	if ( config[trigTow_type]["rz_sigma1_step2"] != 0. ) 
	  sigma_step2[0] = config[trigTow_type]["rz_sigma1_step2"];
	if ( config[trigTow_type]["rz_sigma2_step2"] != 0. ) 
	  sigma_step2[1] = config[trigTow_type]["rz_sigma2_step2"];
	if ( config[trigTow_type]["rz_sigma3_step2"] != 0. ) 
	  sigma_step2[2] = config[trigTow_type]["rz_sigma3_step2"];
	if ( config[trigTow_type]["rz_sigma4_step2"] != 0. ) 
	  sigma_step2[3] = config[trigTow_type]["rz_sigma4_step2"];
	if ( config[trigTow_type]["rz_sigma5_step2"] != 0. ) 
	  sigma_step2[4] = config[trigTow_type]["rz_sigma5_step2"];
	if ( config[trigTow_type]["rz_sigma6_step2"] != 0. ) 
	  sigma_step2[5] = config[trigTow_type]["rz_sigma6_step2"];
	if ( config[trigTow_type]["rz_sigma7_step2"] != 0. ) 
	  sigma_step2[6] = config[trigTow_type]["rz_sigma7_step2"];
	if ( config[trigTow_type]["rz_sigma8_step2"] != 0. ) 
	  sigma_step2[7] = config[trigTow_type]["rz_sigma8_step2"];


	Retina retinaRZ_step2(hits_RZ, pbins_step2+2, qbins_step2+2, 
			      pmin_step2-pstep_step2, pmax_step2+pstep_step2, 
			      qmin_step2-qstep_step2, qmax_step2+qstep_step2, 
			      sigma_step2, minWeight_step2, 1, RZ);


	retinaRZ_step2.fillGrid();
	retinaRZ_step2.findMaxima();
	if ( verboseLevel==1 )
	  retinaRZ_step2.dumpGrid(event_counter,2,imax*100+imax_RZ);
	if ( verboseLevel==2 )
	  retinaRZ_step2.printMaxima();

	pqPoint bestpqRZ_step2 = retinaRZ_step2.getBestPQ();

	// --- If no RZ maximum is found, skip the road:
	if ( bestpqRZ_step2.w == -1. ) {
	  if ( verboseLevel>0 )
	    cout << "*** ERROR in RetinaTrackFitter::fit() at event/road = " << event_counter 
		 << "/" << road_id << ": no maximum found in RZ fit-step 2 (step-1 XY maximum #" 
		 << imax << ", step-1 RZ maximum #" << imax_RZ << "), fit aborted!" << endl;
	  continue;
	}

	if ( retinaRZ_step2.getMaxima().size() > 10 ){
	  cout << "*** ERROR in RetinaTrackFitter::fit() at event/road = " << event_counter 
	       << "/" << road_id << ": " << retinaRZ_step2.getMaxima().size()
	       << " maxima found in RZ fit-step 2 (step-1 maximum #" << imax 
	       << ", step-1 RZ maximum #" << imax_RZ << "), fit aborted!" << endl;
	  if ( verboseLevel>0 ) 
	    retinaRZ_step2.dumpGrid(event_counter,2,imax*100+imax_RZ);
	  continue;
	}


	// --- Invert the X+-X- transformation:
	double p = 0.5*(y1 - y0)/bestpqRZ_step2.q;
	double q = y0 - p*(bestpqRZ_step2.p-bestpqRZ_step2.q);


	// --- Get the track parameters:
	double theta = atan(p);
	if ( theta < 0. ){
	  if ( verboseLevel>0 ) 
	    cout << "*** WARNING in RetinaTrackFitter::fit() at event/road = " << event_counter 
		 << "/" << road_id << ": theta corrected from " << theta << " to " << theta+TMath::Pi() 
		 << endl;
	  theta += TMath::Pi();
	}
	eta = -log(tan(0.5*theta));
	z0  = -q/p;


	// --- Invert eta and z0 signs if we are fitting a negative-eta tower:
	if ( eta_range < 3 ){
	  eta = -eta;
	  z0  = -z0;
	}


	// --- Save the track:
	Track* trk = new Track(c, 0., phi, eta, z0, maximaXY_step2[itrk].w, bestpqRZ_step2.w);
	for(unsigned int ihit=0; ihit<hits_RZ.size(); ihit++)
	  trk->addStubIndex(hits_RZ[ihit]->id);

	tracks.push_back(trk);
	

      } // imax_RZ loop

 
    } // itrk loop
 
 
  } // imax loop


  // --- Clean-up pointers:
  for(vector<Hit_t*>::iterator it=hits.begin(); it!=hits.end(); ++it)
    delete *it;

  
}

void RetinaTrackFitter::fit(){

  vector<Hit*> activatedHits;

  //////// Get the list of unique stubs from all the patterns ///////////
  set<int> ids;
  int total=0;
  
  for(unsigned int i=0;i<patterns.size();i++){
    vector<Hit*> allHits = patterns[i]->getHits();
    total+=allHits.size();
    for(unsigned int j=0;j<allHits.size();j++){
      pair<set<int>::iterator,bool> result = ids.insert(allHits[j]->getID());
      if(result.second==true)
	activatedHits.push_back(allHits[j]);
    }
  }

  fit(activatedHits);
 
}


void RetinaTrackFitter::mergePatterns(){
  //cout<<"Merging of patterns not implemented"<<endl;
}


void RetinaTrackFitter::mergeTracks(){
}


TrackFitter* RetinaTrackFitter::clone(){
  RetinaTrackFitter* fit = new RetinaTrackFitter(nb_layers);
  fit->setPhiRotation(sec_phi);
  fit->setSectorID(sector_id);
  return fit;
}

void RetinaTrackFitter::rotateHits(vector<Hit_t*> hits, double angle){
  
  for (unsigned int ihit=0; ihit<hits.size(); ihit++) {
    double x = hits[ihit]->x*cos(angle) - hits[ihit]->y*sin(angle);
    double y = hits[ihit]->x*sin(angle) + hits[ihit]->y*cos(angle);
    hits[ihit]->x = x;
    hits[ihit]->y = y;
  }

}

void RetinaTrackFitter::confTrans(vector<Hit_t*> hits){
  
  for (unsigned int ihit=0; ihit<hits.size(); ihit++) {
    double R2 = hits[ihit]->x*hits[ihit]->x + hits[ihit]->y*hits[ihit]->y;
    hits[ihit]->x /= R2;
    hits[ihit]->y /= R2;
  }

}

void RetinaTrackFitter::initialize(){

  // Enter all the retina parameters
  // (we refer to the detector geometry in
  //  http://sviret.web.cern.ch/sviret/Images/CMS/Upgrade/Eta6_Phi8.jpg)

  // --- Central trigger tower:
  config[0]["xy_pbins_step1"]     = 40.;
  config[0]["xy_qbins_step1"]     = 40.;
  config[0]["xy_pmin_step1"]      = -0.05;
  config[0]["xy_pmax_step1"]      =  0.05;
  config[0]["xy_qmin_step1"]      = -0.05;
  config[0]["xy_qmax_step1"]      =  0.05;
  config[0]["xy_threshold_step1"] =  4.5;
  config[0]["xy_sigma1_step1"]    =  0.;
  config[0]["xy_sigma2_step1"]    =  0.;
  config[0]["xy_sigma3_step1"]    =  0.;
  config[0]["xy_sigma4_step1"]    =  0.;
  config[0]["xy_sigma5_step1"]    =  0.;
  config[0]["xy_sigma6_step1"]    =  0.;
  config[0]["xy_sigma7_step1"]    =  0.;
  config[0]["xy_sigma8_step1"]    =  0.;
  config[0]["xy_pbins_step2"]     = 100.;
  config[0]["xy_qbins_step2"]     = 100.;
  config[0]["xy_zoom_step2"]      = 1.;
  config[0]["xy_threshold_step2"] =  4.5;
  config[0]["xy_sigma1_step2"]    =  0.;
  config[0]["xy_sigma2_step2"]    =  0.;
  config[0]["xy_sigma3_step2"]    =  0.;
  config[0]["xy_sigma4_step2"]    =  0.;
  config[0]["xy_sigma5_step2"]    =  0.;
  config[0]["xy_sigma6_step2"]    =  0.;
  config[0]["xy_sigma7_step2"]    =  0.;
  config[0]["xy_sigma8_step2"]    =  0.;

  config[0]["rz_pbins_step1"]     = 20.;
  config[0]["rz_qbins_step1"]     = 20.;
  config[0]["rz_pmin_step1"]      = -20.;
  config[0]["rz_pmax_step1"]      =  60.;
  config[0]["rz_qmin_step1"]      = -60.;
  config[0]["rz_qmax_step1"]      =  60.;
  config[0]["rz_threshold_step1"] =  4.5;
  config[0]["rz_sigma1_step1"]    =  0.;
  config[0]["rz_sigma2_step1"]    =  0.;
  config[0]["rz_sigma3_step1"]    =  0.;
  config[0]["rz_sigma4_step1"]    =  0.;
  config[0]["rz_sigma5_step1"]    =  0.;
  config[0]["rz_sigma6_step1"]    =  0.;
  config[0]["rz_sigma7_step1"]    =  0.;
  config[0]["rz_sigma8_step1"]    =  0.;
  config[0]["rz_pbins_step2"]     = 80.;
  config[0]["rz_qbins_step2"]     = 80.;
  config[0]["rz_zoom_step2"]      = 1.5;
  config[0]["rz_threshold_step2"] =  4.;
  config[0]["rz_sigma1_step2"]    =  0.;
  config[0]["rz_sigma2_step2"]    =  0.;
  config[0]["rz_sigma3_step2"]    =  0.;
  config[0]["rz_sigma4_step2"]    =  0.;
  config[0]["rz_sigma5_step2"]    =  0.;
  config[0]["rz_sigma6_step2"]    =  0.;
  config[0]["rz_sigma7_step2"]    =  0.;
  config[0]["rz_sigma8_step2"]    =  0.;

  // --- Hybrid trigger tower:
  config[1]["xy_pbins_step1"]     = 40.;
  config[1]["xy_qbins_step1"]     = 40.;
  config[1]["xy_pmin_step1"]      = -0.05;
  config[1]["xy_pmax_step1"]      =  0.05;
  config[1]["xy_qmin_step1"]      = -0.05;
  config[1]["xy_qmax_step1"]      =  0.05;
  config[1]["xy_threshold_step1"] =  4.;
  config[1]["xy_sigma1_step1"]    =  0.;
  config[1]["xy_sigma2_step1"]    =  0.;
  config[1]["xy_sigma3_step1"]    =  0.;
  config[1]["xy_sigma4_step1"]    =  0.;
  config[1]["xy_sigma5_step1"]    =  0.;
  config[1]["xy_sigma6_step1"]    =  0.;
  config[1]["xy_sigma7_step1"]    =  0.;
  config[1]["xy_sigma8_step1"]    =  0.;
  config[1]["xy_pbins_step2"]     = 100.;
  config[1]["xy_qbins_step2"]     = 100.;
  config[1]["xy_zoom_step2"]      = 1.;
  config[1]["xy_threshold_step2"] =  4.;
  config[1]["xy_sigma1_step2"]    =  0.;
  config[1]["xy_sigma2_step2"]    =  0.;
  config[1]["xy_sigma3_step2"]    =  0.;
  config[1]["xy_sigma4_step2"]    =  0.;
  config[1]["xy_sigma5_step2"]    =  0.;
  config[1]["xy_sigma6_step2"]    =  0.;
  config[1]["xy_sigma7_step2"]    =  0.;
  config[1]["xy_sigma8_step2"]    =  0.;

  config[1]["rz_pbins_step1"]     = 20.;
  config[1]["rz_qbins_step1"]     = 20.;
  config[1]["rz_pmin_step1"]      = 40.;
  config[1]["rz_pmax_step1"]      = 140.;
  config[1]["rz_qmin_step1"]      = 0.;
  config[1]["rz_qmax_step1"]      = 120.;
  config[1]["rz_threshold_step1"] =  4.;
  config[1]["rz_sigma1_step1"]    =  0.;
  config[1]["rz_sigma2_step1"]    =  0.;
  config[1]["rz_sigma3_step1"]    =  0.;
  config[1]["rz_sigma4_step1"]    =  0.;
  config[1]["rz_sigma5_step1"]    =  0.;
  config[1]["rz_sigma6_step1"]    =  0.;
  config[1]["rz_sigma7_step1"]    =  0.;
  config[1]["rz_sigma8_step1"]    =  0.;
  config[1]["rz_pbins_step2"]     = 80.;
  config[1]["rz_qbins_step2"]     = 80.;
  config[1]["rz_zoom_step2"]      = 1.5;
  config[1]["rz_threshold_step2"] =  3.;
  config[1]["rz_sigma1_step2"]    =  0.;
  config[1]["rz_sigma2_step2"]    =  0.;
  config[1]["rz_sigma3_step2"]    =  0.;
  config[1]["rz_sigma4_step2"]    =  0.;
  config[1]["rz_sigma5_step2"]    =  0.;
  config[1]["rz_sigma6_step2"]    =  0.;
  config[1]["rz_sigma7_step2"]    =  0.;
  config[1]["rz_sigma8_step2"]    =  0.;

  // --- Forward trigger tower:
  config[2]["xy_pbins_step1"]     = 40.;
  config[2]["xy_qbins_step1"]     = 40.;
  config[2]["xy_pmin_step1"]      = -0.05;
  config[2]["xy_pmax_step1"]      =  0.05;
  config[2]["xy_qmin_step1"]      = -0.05;
  config[2]["xy_qmax_step1"]      =  0.05;
  config[2]["xy_threshold_step1"] =  4.;
  config[2]["xy_sigma1_step1"]    =  0.;
  config[2]["xy_sigma2_step1"]    =  0.;
  config[2]["xy_sigma3_step1"]    =  0.;
  config[2]["xy_sigma4_step1"]    =  0.;
  config[2]["xy_sigma5_step1"]    =  0.;
  config[2]["xy_sigma6_step1"]    =  0.;
  config[2]["xy_sigma7_step1"]    =  0.;
  config[2]["xy_sigma8_step1"]    =  0.;
  config[2]["xy_pbins_step2"]     = 100.;
  config[2]["xy_qbins_step2"]     = 100.;
  config[2]["xy_zoom_step2"]      = 1.;
  config[2]["xy_threshold_step2"] =  4.;
  config[2]["xy_sigma1_step2"]    =  0.;
  config[2]["xy_sigma2_step2"]    =  0.;
  config[2]["xy_sigma3_step2"]    =  0.;
  config[2]["xy_sigma4_step2"]    =  0.;
  config[2]["xy_sigma5_step2"]    =  0.;
  config[2]["xy_sigma6_step2"]    =  0.;
  config[2]["xy_sigma7_step2"]    =  0.;
  config[2]["xy_sigma8_step2"]    =  0.;

  config[2]["rz_pbins_step1"]     = 20.;
  config[2]["rz_qbins_step1"]     = 20.;
  config[2]["rz_pmin_step1"]      = 140.;
  config[2]["rz_pmax_step1"]      = 240.;
  config[2]["rz_qmin_step1"]      = 80.;
  config[2]["rz_qmax_step1"]      = 180.;
  config[2]["rz_threshold_step1"] =  4.;
  config[2]["rz_sigma1_step1"]    =  1.5;
  config[2]["rz_sigma2_step1"]    =  1.5;
  config[2]["rz_sigma3_step1"]    =  1.5;
  config[2]["rz_sigma4_step1"]    =  1.5;
  config[2]["rz_sigma5_step1"]    =  1.5;
  config[2]["rz_sigma6_step1"]    =  1.5;
  config[2]["rz_sigma7_step1"]    =  1.5;
  config[2]["rz_sigma8_step1"]    =  1.5;
  config[2]["rz_pbins_step2"]     = 80.;
  config[2]["rz_qbins_step2"]     = 80.;
  config[2]["rz_zoom_step2"]      = 1.5;
  config[2]["rz_threshold_step2"] = 3.;
  config[2]["rz_sigma1_step2"]    = 0.;
  config[2]["rz_sigma2_step2"]    = 0.;
  config[2]["rz_sigma3_step2"]    = 0.;
  config[2]["rz_sigma4_step2"]    = 0.;
  config[2]["rz_sigma5_step2"]    = 0.;
  config[2]["rz_sigma6_step2"]    = 0.;
  config[2]["rz_sigma7_step2"]    = 0.;
  config[2]["rz_sigma8_step2"]    = 0.;


}
