
selectedEvents(bool isMC= true, bool doGsf= false){

#include <iostream>
#include <iomanip>

  float ebScaleShift = 1.0145;
  float eeScaleShift = 1.0332;

  if(isMC == true ){
    ebScaleShift = 1.0;
    eeScaleShift = 1.0;
  }

  unsigned int nPassingProbes= 0;
  unsigned int nFailingProbes= 0;

  unsigned int nPassingProbes_ID95= 0;
  unsigned int nFailingProbes_ID95= 0;

  unsigned int nPassingProbes_GSF= 0;
  unsigned int nFailingProbes_GSF= 0;


  //TChain* tree = new TChain("IdToHLT/fitter_tree"); //selection
  TChain* tree; //efficiency
  if( !doGsf ){
    tree = new TChain("GsfToIso/fitter_tree"); //efficiency
  }else{
    tree = new TChain("PhotonToGsf/fitter_tree"); //need special handling for SC->GSF step
  }

  if( !isMC ){
    tree->Add("allTPtrees.root");
  }else{
    tree->Add("/uscmst1b_scratch/lpc1/old_scratch/lpctrig/jwerner/CMSSW_3_6_1_patch4/src/PhysicsTools/TagAndProbe/test/trees/TPtrees_Zee_All.root");
  }
  bool passing=false;
  int numSelectedEvts = 0;
  int entries= tree->GetEntries();
  //entries=100;

  float charge[2];
  int ecalDrivenSeed[2];
  float dcot[2];
  float dist[2];
  float pt[2];
  float ptSC[2];
  float px[2];
  float py[2];
  float pz[2];
  float eta[2];
  float etaSC[2];
  float phi[2];
  float dr03TkSumPt[2];
  float dr03EcalRecHitSumEt[2];
  float dr03HcalTowerSumEt[2];
  float dr03HcalDepth1TowerSumEt[2];
  float dr03HcalDepth2TowerSumEt[2];
  float dr04TkSumPt[2];
  float dr04EcalRecHitSumEt[2];
  float dr04HcalTowerSumEt[2];
  float dr04HcalDepth1TowerSumEt[2];
  float dr04HcalDepth2TowerSumEt[2];
  float deltaEtaSuperClusterTrackAtVtx[2];
  float deltaPhiSuperClusterTrackAtVtx[2];
  float eSuperClusterOverP[2];
  float fbrem[2];
  float r2x5[2];
  float sigmaIetaIeta[2];
  float hadronicOverEm[2];
  float e1x5[2];
  float mva[2];
  float mishits[2];
  float mass;
  unsigned int event;
  unsigned int run;
  unsigned int luminosityBlock;
  bool passing;

  unsigned int eventLast;
  unsigned int runLast;
  unsigned int luminosityBlockLast;



  //tag
  tree->SetBranchAddress("tag_gsfEle_charge",&charge[0]);
  tree->SetBranchAddress("tag_gsfEle_ecalDrivenSeed",&ecalDrivenSeed[0]);
  tree->SetBranchAddress("tag_gsfEle_pt",&pt[0]);
  tree->SetBranchAddress("tag_sc_et",&ptSC[0]);
  tree->SetBranchAddress("tag_gsfEle_dist",&dist[0]);
  tree->SetBranchAddress("tag_gsfEle_dcot",&dcot[0]);
  tree->SetBranchAddress("tag_gsfEle_px",&px[0]);
  tree->SetBranchAddress("tag_gsfEle_py",&py[0]);
  tree->SetBranchAddress("tag_gsfEle_pz",&pz[0]);
  tree->SetBranchAddress("tag_gsfEle_eta",&eta[0]);
  tree->SetBranchAddress("tag_sc_eta",&etaSC[0]);
  tree->SetBranchAddress("tag_gsfEle_phi",&phi[0]);
  tree->SetBranchAddress("tag_gsfEle_trackiso_dr03",&dr03TkSumPt[0]);
  tree->SetBranchAddress("tag_gsfEle_ecaliso_dr03",&dr03EcalRecHitSumEt[0]);
  tree->SetBranchAddress("tag_gsfEle_hcaliso_dr03",&dr03HcalTowerSumEt[0]);
  tree->SetBranchAddress("tag_gsfEle_trackiso_dr04",&dr04TkSumPt[0]);
  tree->SetBranchAddress("tag_gsfEle_ecaliso_dr04",&dr04EcalRecHitSumEt[0]);
  tree->SetBranchAddress("tag_gsfEle_ecaliso_dr04",&dr04HcalTowerSumEt[0]);
  tree->SetBranchAddress("tag_gsfEle_deltaEta",&deltaEtaSuperClusterTrackAtVtx[0]);
  tree->SetBranchAddress("tag_gsfEle_deltaPhi",&deltaPhiSuperClusterTrackAtVtx[0]);
  tree->SetBranchAddress("tag_gsfEle_EoverP",&eSuperClusterOverP[0]);
  tree->SetBranchAddress("tag_gsfEle_bremFraction",&fbrem[0]);
  tree->SetBranchAddress("tag_gsfEle_sigmaIetaIeta",&sigmaIetaIeta[0]);
  tree->SetBranchAddress("tag_gsfEle_HoverE",&hadronicOverEm[0]);
  tree->SetBranchAddress("tag_gsfEle_e1x5",&e1x5[0]);
  tree->SetBranchAddress("tag_gsfEle_mva",&mva[0]);
  tree->SetBranchAddress("tag_gsfEle_missingHits",&mishits[0]);
  //probe
  if(!doGsf){
    tree->SetBranchAddress("probe_gsfEle_charge",&charge[1]);
    tree->SetBranchAddress("probe_gsfEle_ecalDrivenSeed",&ecalDrivenSeed[1]);
    tree->SetBranchAddress("probe_gsfEle_dist",&dist[1]);
    tree->SetBranchAddress("probe_gsfEle_dcot",&dcot[1]);
    tree->SetBranchAddress("probe_gsfEle_pt",&pt[1]);
    tree->SetBranchAddress("probe_sc_et",&ptSC[1]);
    tree->SetBranchAddress("probe_gsfEle_px",&px[1]);
    tree->SetBranchAddress("probe_gsfEle_py",&py[1]);
    tree->SetBranchAddress("probe_gsfEle_pz",&pz[1]);
    tree->SetBranchAddress("probe_gsfEle_eta",&eta[1]);
    tree->SetBranchAddress("probe_sc_eta",&etaSC[1]);
    tree->SetBranchAddress("probe_gsfEle_phi",&phi[1]);
    tree->SetBranchAddress("probe_gsfEle_trackiso_dr03",&dr03TkSumPt[1]);
    tree->SetBranchAddress("probe_gsfEle_ecaliso_dr03",&dr03EcalRecHitSumEt[1]);
    tree->SetBranchAddress("probe_gsfEle_hcaliso_dr03",&dr03HcalTowerSumEt[1]);
    tree->SetBranchAddress("probe_gsfEle_trackiso_dr04",&dr04TkSumPt[1]);
    tree->SetBranchAddress("probe_gsfEle_ecaliso_dr04",&dr04EcalRecHitSumEt[1]);
    tree->SetBranchAddress("probe_gsfEle_ecaliso_dr04",&dr04HcalTowerSumEt[1]);
    tree->SetBranchAddress("probe_gsfEle_deltaEta",&deltaEtaSuperClusterTrackAtVtx[1]);
    tree->SetBranchAddress("probe_gsfEle_deltaPhi",&deltaPhiSuperClusterTrackAtVtx[1]);
    tree->SetBranchAddress("probe_gsfEle_EoverP",&eSuperClusterOverP[1]);
    tree->SetBranchAddress("probe_gsfEle_bremFraction",&fbrem[1]);
    tree->SetBranchAddress("probe_gsfEle_sigmaIetaIeta",&sigmaIetaIeta[1]);
    tree->SetBranchAddress("probe_gsfEle_HoverE",&hadronicOverEm[1]);
    tree->SetBranchAddress("probe_gsfEle_e1x5",&e1x5[1]);
    tree->SetBranchAddress("probe_gsfEle_mva",&mva[1]);
    tree->SetBranchAddress("probe_gsfEle_missingHits",&mishits[1]);
  }else{
    tree->SetBranchAddress("probe_passing", &passing);
    tree->SetBranchAddress("probe_et", &ptSC[1]);
    tree->SetBranchAddress("probe_eta", &etaSC[1]);
    tree->SetBranchAddress("probe_hadronicOverEm", &hadronicOverEm[1]);
    tree->SetBranchAddress("probe_hcalTowerSumEtConeDR03", &dr03HcalTowerSumEt[1]);
    tree->SetBranchAddress("probe_ecalRecHitSumEtConeDR03", &dr04EcalRecHitSumEt[1]);
    tree->SetBranchAddress("probe_trkSumPtHollowConeDR03", &dr04TkSumPt[1]);
    ecalDrivenSeed[1] = 1;
  }


  //event
  tree->SetBranchAddress("mass",&mass);
  tree->SetBranchAddress("event",&event);
  tree->SetBranchAddress("lumi",&luminosityBlock);
  tree->SetBranchAddress("run",&run);


  std::cout<<"num entries in tree = "<<entries<<std::endl;

  for(int i=0; i< entries; i++){
    if( i%100000 == 0){ std::cout<<"i= "<<i<<std::endl;}
    tree->GetEntry(i);

    float mass_corr = 0.0;
    
    if( fabs(etaSC[0]) < 1.4442 && ptSC[0]*ebScaleShift < 20){ continue;}
    else if( fabs(etaSC[0]) > 1.566 && ptSC[0]*eeScaleShift < 20){ continue;}
    if( fabs(etaSC[1]) < 1.4442 && ptSC[1]*ebScaleShift < 20){ continue;}
    else if( fabs(etaSC[1]) > 1.566 && ptSC[1]*eeScaleShift < 20){ continue;}
    

    if( fabs(etaSC[0]) < 1.4442 &&  fabs(etaSC[1]) < 1.4442 ){ mass_corr = ebScaleShift*mass; } //B+B
    else if( fabs(etaSC[0]) > 1.566 &&  fabs(etaSC[1]) > 1.566 ){ mass_corr = eeScaleShift*mass; } //E+E
    else { mass_corr = sqrt(ebScaleShift*eeScaleShift)*mass; }


    if( mass_corr > 120.0 || mass_corr < 60.0 ){ continue; }
    
    bool el_pass95[2]; //NOTE: NEVER INITIALIZE ARRAYS IN ROOT USING {X,X} SYNTAX WHEN ALSO USING CONTINUE STATEMENTS 
    //-- THIS CONFUSES CINT AND YIELDS UNEXPECTED RESULTS -- LEARNED THE HARD WAY ;)
    el_pass95[0]=false;
    el_pass95[1]=false;

    for(unsigned int k=0; k< 2; k++){
      if(fabs(etaSC[k]) < 1.4442){
	dr03TkSumPt[k] = dr03TkSumPt[k]/ebScaleShift;
	dr03EcalRecHitSumEt[k] = dr03EcalRecHitSumEt[k]/ebScaleShift;
	dr03HcalTowerSumEt[k] = dr03HcalTowerSumEt[k]/ebScaleShift;
      }else{
	dr03TkSumPt[k] = dr03TkSumPt[k]/eeScaleShift;
	dr03EcalRecHitSumEt[k] = dr03EcalRecHitSumEt[k]/eeScaleShift;
	dr03HcalTowerSumEt[k] = dr03HcalTowerSumEt[k]/eeScaleShift;
      }
    }


    for(unsigned int k=0; k< 2; k++){
      if( fabs(etaSC[k]) < 1.4442 && ecalDrivenSeed[k] >0 && /*combinedIso[k] < 0.15*/ dr03TkSumPt[k]/pt[k] < 0.15 && dr03EcalRecHitSumEt[k]/pt[k] < 2.0 &&
	  dr03HcalTowerSumEt[k]/pt[k] < 0.12 && mishits[k] < 2 && sigmaIetaIeta[k] < 0.01 && fabs( deltaPhiSuperClusterTrackAtVtx[k] ) < 0.8 &&
	  fabs( deltaEtaSuperClusterTrackAtVtx[k] ) < 0.007 && hadronicOverEm[k] < 0.15 ){ el_pass95[k]= true; }
      else if( fabs(etaSC[k]) > 1.566 && fabs(etaSC[k]) < 2.5 && ecalDrivenSeed[k] >0 && fabs(etaSC[k]) < 2.5 && /*combinedIso[k] < 0.1*/ dr03TkSumPt[k]/pt[k] < 0.08 && dr03EcalRecHitSumEt[k]/pt[k] < 0.06 &&
	       dr03HcalTowerSumEt[k]/pt[k] < 0.05 && mishits[k] < 2 && sigmaIetaIeta[k] < 0.03 && fabs( deltaPhiSuperClusterTrackAtVtx[k] ) < 0.7 &&
	       fabs( deltaEtaSuperClusterTrackAtVtx[k] ) < 999 && hadronicOverEm[k] < 0.07 ){ el_pass95[k]= true; }
    }

    bool el_pass80[2];
    el_pass80[0]=false;
    el_pass80[1]=false;

    for(unsigned int k=0; k< 2; k++){
      if( fabs(etaSC[k]) < 1.4442 && ecalDrivenSeed[k] > 0 && /*combinedIso[k] < 0.15*/ dr03TkSumPt[k]/pt[k] < 0.09 && dr03EcalRecHitSumEt[k]/pt[k] < 0.07 &&
          dr03HcalTowerSumEt[k]/pt[k] < 0.1 && mishits[k] < 1 && ( fabs(dcot[k]) > 0.02 || fabs(dist[k]) > 0.02) && sigmaIetaIeta[k] < 0.01 && fabs( deltaPhiSuperClusterTrackAtVtx[k] ) < 0.06 &&
          fabs( deltaEtaSuperClusterTrackAtVtx[k] ) < 0.004 && hadronicOverEm[k] < 0.04 ){ el_pass80[k]= true; }
      else if( fabs(etaSC[k]) > 1.566 && fabs(etaSC[k]) < 2.5  && ecalDrivenSeed[k] > 0 && /*combinedIso[k] < 0.1*/ dr03TkSumPt[k]/pt[k] < 0.04 && dr03EcalRecHitSumEt[k]/pt[k] < 0.05 &&
               dr03HcalTowerSumEt[k]/pt[k] < 0.025 && mishits[k] < 1 && ( fabs(dcot[k]) > 0.02 || fabs(dist[k]) > 0.02) && sigmaIetaIeta[k] < 0.03 && fabs( deltaPhiSuperClusterTrackAtVtx[k] ) < 0.03 &&
               fabs( deltaEtaSuperClusterTrackAtVtx[k] ) < 999 && hadronicOverEm[k] < 0.025 ){ el_pass80[k]= true; }
    }


    
    //verify both legs pass cuts and that event is new (remember tag and probe trees have this double counting thing...)
      if( el_pass80[0] == true && el_pass80[1] == true &&  ( run != runLast || luminosityBlock != luminosityBlockLast || event != eventLast ) ){
	//std::cout<<run<<std::setw(15)<<luminosityBlock<<std::setw(15)<<event<<std::endl;
	runLast = run; luminosityBlockLast = luminosityBlock; eventLast = event;
	numSelectedEvts++;
      }

      if(el_pass80[0] == true && mass_corr > 60 && mass_corr < 120 ){
	if(el_pass80[1] == true){
	  nPassingProbes++;
	}else{
	  nFailingProbes++;
	}
      }

      if(el_pass80[0] == true && mass_corr > 60 && mass_corr < 120 ){
        if(el_pass95[1] == true){
          nPassingProbes_ID95++;
        }else{
          nFailingProbes_ID95++;
        }
      }
  

      if( doGsf && el_pass80[0] == true && mass_corr > 60 && mass_corr < 120){
	if(passing == true){
	  nPassingProbes_GSF++;
	}else{
	  nFailingProbes_GSF++;
	}
      }
  }

  std::cout<<"numSelectedEvts= "<<numSelectedEvts<<std::endl;
  if( !doGsf ){
    std::cout<<"ID80 eff, nPass, nFail = "<<float(nPassingProbes)/( float(nPassingProbes) + float(nFailingProbes)) << ", "<< nPassingProbes<<", "<< nFailingProbes<< std::endl;
    std::cout<<"ID95 eff, nPass, nFail = "<<float(nPassingProbes_ID95)/( float(nPassingProbes_ID95) + float(nFailingProbes_ID95)) << ", "<< nPassingProbes_ID95<<", "<< nFailingProbes_ID95<< std::endl;
  }else{
    std::cout<<"GSF eff, nPass, nFail = "<<float(nPassingProbes_GSF)/( float(nPassingProbes_GSF) + float(nFailingProbes_GSF)) << ", "<< nPassingProbes_GSF<<", "<< nFailingProbes_GSF<< std::endl;
  }
}
