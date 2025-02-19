#include <iostream>
#include <iomanip>

effCalculator(const unsigned int effType=1, const unsigned int region=0, const unsigned int bothElorPos= 0, const bool isData = true, const bool isMCTruth = false){

  //efftype = 0 sc->gsf, 1 gsf->wp95, 2 gsf->wp80, 3 wp95->hlt, 4 wp80->hlt
  //region = 0 all, 1 EB, 2 EE
  //bothElorPos = 0 all, 1 e-, 2 e+

  std::cout<<"INPUT ARGS effType, region, bothElorPos, isData= "<<effType<< ", "<< region <<", "<< bothElorPos<<", "<< isData << std::endl;

  gROOT->ProcessLine(".L ./tdrstyle.C");
  setTDRStyle();
  tdrStyle->SetErrorX(0.5);
  tdrStyle->SetPadLeftMargin(0.19);
  tdrStyle->SetPadRightMargin(0.10);
  tdrStyle->SetPadBottomMargin(0.15);
  tdrStyle->SetLegendBorderSize(0);
  tdrStyle->SetTitleYOffset(1.5);
  tdrStyle->SetOptTitle(0);

  const unsigned int region = region;
  const unsigned int bothElorPos = bothElorPos;

  using namespace RooFit;

  TFile *templates_d = new TFile("templates.root"); //note we only bother w/ template numbers for data for now at least... not MC

  const int rebin = 5;

  TH1F* SignalTemplate = ( (TH1F* ) templates_d->Get("RelTrkIso_OS_tight") ->Clone());
  TH1F* BackgroundTemplate = ( (TH1F* ) templates_d->Get("RelTrkIso_SS_Sideband_loose") ->Clone());

  TH1F* DataPass_forTemplateMethod = new TH1F("DataPass", "DataPass", 100/rebin,0.0,1.0);
  TH1F* DataFail_forTemplateMethod = new TH1F("DataFail", "DataFail", 100/rebin,0.0,1.0);
  SignalTemplate->Rebin(rebin);
  BackgroundTemplate->Rebin(rebin);



  for(int k=0;k< 100/rebin + 1; k++){
    if(SignalTemplate->GetBinLowEdge(k)>= 0.15 ){
      SignalTemplate->SetBinContent(k,0);
      BackgroundTemplate->SetBinContent(k,0);
    }

  }


  TString foutName = "tpHistos";
  if(effType == 1 ){ foutName += "_ID95"; }
  else if(effType == 2 ){ foutName += "_ID80"; }
  else if(effType == 0 ){ foutName += "_GSF"; }
  else if(effType == 3 ){ foutName += "_HLT95"; }
  else if(effType == 4 ){ foutName += "_HLT80"; }
  TString c1Name = "XXXXXXXXXXX";
  TString c2Name = "XXXXXXXXXXX";
  if( region == 1){ foutName += "_eb";}
  else if( region == 2){ foutName += "_ee";}
  if( bothElorPos == 1){ foutName += "_minus";}
  else if( bothElorPos == 2 ){ foutName += "_plus";}
  c1Name = foutName;
  c2Name = foutName;
  c1Name += "_pass";
  c2Name += "_fail";
  if( !isData ){ foutName += "_MONTECARLO"; }
  foutName += ".root";

  TFile* fout = new TFile(foutName, "RECREATE");
  fout->cd();
  TH1F* hGSF_pass;
  TH1F* hGSF_fail;
  TChain* chain;

  TString type = "scratch";
  TString chainName = "GsfToIso/fitter_tree";

  if( !isData && isMCTruth ){
    chainName= "GsfToIso/mcUnbias_tree";
  }


  if( effType == 0 ){ 
    chainName= "PhotonToGsf/fitter_tree";
    if( !isData && isMCTruth ){
      chainName= "PhotonToGsf/mcUnbias_tree";
    }
  }

    TString hname = type;    
    std::cout<<hname<<std::endl;
    hname+= "_pass";
    hGSF_pass = new TH1F(hname, hname, 30, 60, 120);
    hname = type;
    hname+= "_fail";
    hGSF_fail = new TH1F(hname, hname, 12, 60, 120);
    std::cout<<"foo"<<std::endl;

    chain = new TChain(chainName);
    if( isData ){
      chain->Add("allTPtrees.root");
    }else{  chain->Add("/uscmst1b_scratch/lpc1/old_scratch/lpctrig/jwerner/CMSSW_3_6_1_patch4/src/PhysicsTools/TagAndProbe/test/trees/TPtrees_Zee_All.root"); }
    float mass=0;
    float probe_et=0;
    float probe_eta=0;
    float probe_pt=0;
    float probe_charge=0;
    float probe_combIso=0;
    float probe_hcalIso=0;
    float probe_ecalIso=0;
    float probe_trkIso=0;
    int probe_ecalDriven=0;
    float probe_missHits=0;
    float probe_dist=0;
    float probe_dcot=0;
    float probe_sigIeta=0;
    float probe_deta=0;
    float probe_dphi=0;
    float probe_hoe=0;

    float tag_et=0;
    float tag_eta=0;
    float tag_pt=0;
    float tag_charge=0;
    float tag_combIso=0;
    float tag_hcalIso=0;
    float tag_ecalIso=0;
    float tag_trkIso=0;
    float tag_dcot=0;
    float tag_dist=0;
    float tag_missHits=0;
    float tag_sigIeta=0;
    float tag_deta=0;
    float tag_dphi=0;
    float tag_hoe=0;

    bool passing=false;
    bool passingALL=false;
    float mass = 0;
    float mass_corr = 0;

     //use probe_passing for 95 and probe_passingId80 for 80

    if( effType == 1 || effType == 3){
      chain->SetBranchAddress("probe_passingId", &passing);
    }else if( effType == 2 || effType == 4){    
      chain->SetBranchAddress("probe_passingId80", &passing);
    }

    if( effType != 0 ){
      chain->SetBranchAddress("probe_passingALL", &passingALL);
      chain->SetBranchAddress("probe_gsfEle_missingHits", &probe_missHits);
      chain->SetBranchAddress("probe_gsfEle_dcot", &probe_dcot);
      chain->SetBranchAddress("probe_gsfEle_dist", &probe_dist);
      chain->SetBranchAddress("probe_gsfEle_charge", &probe_charge);
      chain->SetBranchAddress("probe_gsfEle_hcaliso_dr03", &probe_hcalIso);
      chain->SetBranchAddress("probe_gsfEle_ecaliso_dr03", &probe_ecalIso);
      chain->SetBranchAddress("probe_gsfEle_trackiso_dr03", &probe_trkIso);
      chain->SetBranchAddress("probe_gsfEle_pt", &probe_pt);
      chain->SetBranchAddress("probe_sc_et", &probe_et);
      chain->SetBranchAddress("probe_sc_eta", &probe_eta);
      chain->SetBranchAddress("probe_gsfEle_ecalDrivenSeed", &probe_ecalDriven);
      chain->SetBranchAddress("probe_gsfEle_sigmaIetaIeta", &probe_sigIeta);
      chain->SetBranchAddress("probe_gsfEle_deltaEta", &probe_deta);
      chain->SetBranchAddress("probe_gsfEle_deltaPhi", &probe_dphi);
      chain->SetBranchAddress("probe_gsfEle_ecalDrivenSeed", &probe_ecalDriven);
      chain->SetBranchAddress("probe_gsfEle_sigmaIetaIeta", &probe_sigIeta);
      chain->SetBranchAddress("probe_gsfEle_deltaEta", &probe_deta);
      chain->SetBranchAddress("probe_gsfEle_deltaPhi", &probe_dphi);
    }
    else{
      chain->SetBranchAddress("probe_passing", &passing);
      chain->SetBranchAddress("probe_et", &probe_pt);
      chain->SetBranchAddress("probe_eta", &probe_eta);
      chain->SetBranchAddress("probe_hadronicOverEm", &probe_hoe);
      chain->SetBranchAddress("probe_hcalTowerSumEtConeDR03", &probe_hcalIso);
      chain->SetBranchAddress("probe_ecalRecHitSumEtConeDR03", &probe_ecalIso);
      chain->SetBranchAddress("probe_trkSumPtHollowConeDR03", &probe_trkIso);
      probe_ecalDriven = 1;
    }
    if( isData || !isMCTruth ){
      chain->SetBranchAddress("tag_sc_et", &tag_et);
      chain->SetBranchAddress("tag_sc_eta", &tag_eta);
      chain->SetBranchAddress("tag_gsfEle_charge", &tag_charge);
      chain->SetBranchAddress("tag_gsfEle_hcaliso_dr03", &tag_hcalIso);
      chain->SetBranchAddress("tag_gsfEle_ecaliso_dr03", &tag_ecalIso);
      chain->SetBranchAddress("tag_gsfEle_trackiso_dr03", &tag_trkIso);
      chain->SetBranchAddress("tag_gsfEle_pt", &tag_pt);
      chain->SetBranchAddress("tag_gsfEle_dcot", &tag_dcot);
      chain->SetBranchAddress("tag_gsfEle_dist", &tag_dist);
      chain->SetBranchAddress("tag_gsfEle_missingHits", &tag_missHits);
      chain->SetBranchAddress("tag_gsfEle_sigmaIetaIeta", &tag_sigIeta);
      chain->SetBranchAddress("tag_gsfEle_deltaEta", &tag_deta);
      chain->SetBranchAddress("tag_gsfEle_deltaPhi", &tag_dphi);
      chain->SetBranchAddress("tag_gsfEle_HoverE", &tag_hoe);
      chain->SetBranchAddress("mass", &mass);
    }


    int entries= chain->GetEntries();
    //if( !isData ){ entries = 100000; }

    unsigned int nOSPass = 0;
    unsigned int nSSPass = 0;
    unsigned int nOSFail = 0;
    unsigned int nSSFail = 0;

    RooRealVar x("x","M_{ee}",60,120, "GeV") ;
    RooDataSet d_pass("d_pass","d_pass",RooArgSet(x));
    RooDataSet d_fail("d_fail","d_fail",RooArgSet(x));
    
    std::cout<<"entries= "<<entries<<std::endl;
    for(int i=0; i< entries; i++){
      chain->GetEntry(i);

      if(effType == 0 ){ probe_et = probe_pt; }

      if( region == 1 && fabs(probe_eta) > 1.4442 ){continue;}
      else if( region == 2 && fabs(probe_eta) < 1.566 ){continue;}

      //this is for the non sc->gsf steps
      if( effType!= 0 && bothElorPos == 1 && probe_charge > 0 ){ continue;}
      else if(  effType!= 0 && bothElorPos == 2 && probe_charge < 0 ){ continue;}

      //this is for the sc->gsf steps
      if( (isData || !isMCTruth) && effType== 0 && bothElorPos == 1 && tag_charge < 0 ){ continue;}
      else if( (isData || !isMCTruth) && effType== 0 && bothElorPos == 2 && tag_charge > 0 ){ continue;}
      //note that for the MCTruth case we lose all charge info at this point...

      //float ebScaleShift = 1.0115 - 0.0;
      //float eeScaleShift = 1.0292 - 0.0;

      float ebScaleShift = 1.0145;
      float eeScaleShift = 1.0332;

      if( !isData ){
	ebScaleShift = 1.0;
	eeScaleShift = 1.0;
      }

      //apply scale correction to iso variables
      if( fabs(probe_eta) < 1.4442 ){
        probe_trkIso = probe_trkIso/ebScaleShift;
        probe_ecalIso = probe_ecalIso/ebScaleShift;
        probe_hcalIso = probe_hcalIso/ebScaleShift;
      }else{
	probe_trkIso = probe_trkIso/eeScaleShift;
        probe_ecalIso = probe_ecalIso/eeScaleShift;
        probe_hcalIso = probe_hcalIso/eeScaleShift;
      }
      if( fabs(tag_eta) < 1.4442 ){
	tag_trkIso = tag_trkIso/ebScaleShift;
	tag_ecalIso = tag_ecalIso/ebScaleShift;
        tag_hcalIso = tag_hcalIso/ebScaleShift;
      }else{
        tag_trkIso = tag_trkIso/eeScaleShift;
        tag_ecalIso = tag_ecalIso/eeScaleShift;
        tag_hcalIso = tag_hcalIso/eeScaleShift;
      }


      if( probe_ecalDriven< 1){ continue; }
      if( fabs(probe_eta) < 1.4442 && probe_et*ebScaleShift < 20){ continue;} 
      else if( (isData || !isMCTruth) && fabs(tag_eta) < 1.4442 && tag_et*ebScaleShift < 20){ continue;} 
      else if( fabs(probe_eta) > 1.566 && probe_et*eeScaleShift < 20){ continue;} 
      else if( (isData || !isMCTruth) && fabs(tag_eta) > 1.566 && tag_et*eeScaleShift < 20){ continue;} 

      if( (isData || !isMCTruth) && fabs(probe_eta) < 1.4442 &&  fabs(tag_eta) < 1.4442 ){ mass_corr = ebScaleShift*mass; } //B+B
      else if( (isData || !isMCTruth) && fabs(probe_eta) > 1.566 &&  fabs(tag_eta) > 1.566 ){ mass_corr = eeScaleShift*mass; } //E+E
      else if( (isData || !isMCTruth) ){ mass_corr = sqrt(ebScaleShift*eeScaleShift)*mass; }
      else{ mass = 90; mass_corr = mass; }

      if( mass_corr > 120.0 || mass_corr < 60.0 ){ continue; }

      tag_combIso = (tag_trkIso + max(0, tag_ecalIso - 1.0) + tag_hcalIso ) / tag_pt;
      if( fabs(tag_eta) > 1.566){
        tag_combIso = (tag_trkIso + tag_ecalIso  + tag_hcalIso ) / tag_pt;
      }

      probe_combIso = (probe_trkIso + max(0, probe_ecalIso - 1.0) + probe_hcalIso ) / probe_pt;
      if( fabs(probe_eta) > 1.566){
        probe_combIso = (probe_trkIso + probe_ecalIso  + probe_hcalIso ) / probe_pt;
      }


      //verify that tag passes the tag criteria (WP80)
      if( (isData || !isMCTruth) && fabs(tag_eta) < 1.4442 && ( tag_missHits > 0 || ( fabs(tag_dist) < 0.02 && fabs(tag_dcot) < 0.02 ) || 
				      tag_trkIso/tag_pt > 0.09 || tag_ecalIso/tag_pt > 0.07 || tag_hcalIso/tag_pt > 0.1 || 
				      tag_sigIeta > 0.01 || fabs(tag_dphi) > 0.06 || fabs(tag_deta) > 0.004 || tag_hoe > 0.04 ) ){ continue;}
      if( (isData || !isMCTruth) && fabs(tag_eta) > 1.566 && ( tag_missHits > 0  || ( fabs(tag_dist) < 0.02 && fabs(tag_dcot) < 0.02 )  || 
				     tag_trkIso/tag_pt > 0.04 || tag_ecalIso/tag_pt > 0.05 || tag_hcalIso/tag_pt > 0.025 ||
				     tag_sigIeta > 0.03 || fabs(tag_dphi) > 0.03 || tag_hoe > 0.025 ) ){ continue;}


      if( effType == 1 || effType == 3 ){ //WP95 or HLT95

	if( fabs(probe_eta) < 1.4442 && ( probe_missHits > 1 ||
					  probe_trkIso/probe_pt > 0.15 || probe_ecalIso/probe_pt > 2.0 || probe_hcalIso/probe_pt > 0.12 ||
					  probe_sigIeta > 0.01 || fabs(probe_dphi) > 0.8 || fabs(probe_deta) > 0.007 || probe_hoe > 0.15 ) ){ passing= false;}
	if( fabs(probe_eta) > 1.566 && ( probe_missHits > 1  ||
					 probe_trkIso/probe_pt > 0.08 || probe_ecalIso/probe_pt > 0.06 || probe_hcalIso/probe_pt > 0.05 ||
					 probe_sigIeta > 0.03 || fabs(probe_dphi) > 0.7 || probe_hoe > 0.07 ) ){ passing= false;}
      }

      if( effType == 2 || effType == 4 ){ //WP80 or HLT80

        if( fabs(probe_eta) < 1.4442 && ( probe_missHits > 0 || ( fabs(probe_dist) < 0.02 && fabs(probe_dcot) < 0.02 ) ||
					  probe_trkIso/probe_pt > 0.09 || probe_ecalIso/probe_pt > 0.07 || probe_hcalIso/probe_pt > 0.1 ||
					  probe_sigIeta > 0.01 || fabs(probe_dphi) > 0.06 || fabs(probe_deta) > 0.004 || probe_hoe > 0.04 ) ){ passing= false;}
        if( fabs(probe_eta) > 1.566 && ( probe_missHits > 0  || ( fabs(probe_dist) < 0.02 && fabs(probe_dcot) < 0.02 )  ||
					 probe_trkIso/probe_pt > 0.04 || probe_ecalIso/probe_pt > 0.05 || probe_hcalIso/probe_pt > 0.025 ||
					 probe_sigIeta > 0.03 || fabs(probe_dphi) > 0.03 || probe_hoe > 0.025 ) ){ passing= false;}
      }


      if( (effType >= 3 && passing && probe_hoe<0.15 && passingALL) || (effType < 3 && passing && probe_hoe<0.15) ){ 
	  hGSF_pass->Fill(mass_corr);
	  if( probe_charge*tag_charge > 0 ){ nSSPass++;}else{ nOSPass++;}
	  DataPass_forTemplateMethod->Fill(tag_trkIso/tag_pt);
	  x = mass_corr;
	  d_pass.add( RooArgSet(x) );
	  //DataPass_forTemplateMethod->Fill(probe_trkIso/probe_pt);
      }
      else if( (effType >= 3 && passing && probe_hoe<0.15 && !passingALL) || (effType < 3 && !passing && probe_hoe<0.15) ){ 
	  hGSF_fail->Fill(mass_corr);
	  if( probe_charge*tag_charge > 0 ){ nSSFail++;}else{ nOSFail++;}
	  DataFail_forTemplateMethod->Fill(tag_trkIso/tag_pt);
	  x = mass_corr;
          d_fail.add( RooArgSet(x) );
	  //DataFail_forTemplateMethod->Fill(probe_trkIso/probe_pt);
	}
    }

    //x.setBins(30);
    TFile* file_gen =  new TFile("ZeeGenLevel.root", "READ");
    TH1* gen_hist = (TH1D*) file_gen->Get("Mass");
    gen_hist->Rebin(5);
    RooDataHist* rdh = new RooDataHist("rdh","", x, gen_hist);
    RooHistPdf* rdh_what = new RooHistPdf("rdh_what", "", RooArgSet(x), (const RooDataHist) rdh );

    RooRealVar p1("p1","cb p1",  2.0946e-01 ) ;
    RooRealVar p2("p2","cb p2",  8.5695e-04 );
    RooRealVar p3("p3","cb p3",  3.8296e-04 );
    RooRealVar p4("p4","cp p4",  6.7489e+00 );
    RooRealVar sigma("sigma","width of gaussian",   2.5849e+00);
    RooRealVar frac("frac","fraction of gauss2", 6.5704e-01 , 0, 1);
    RooCBExGaussShape cbX("cbX", "crystal ball X", x, p1, p2, p3, p4, sigma, frac);

    p1.setConstant(kFALSE);
    p2.setConstant(kFALSE);
    p3.setConstant(kTRUE);
    p4.setConstant(kTRUE);
    sigma.setConstant(kTRUE);
    frac.setConstant(kTRUE);
    RooFFTConvPdf* signalShapePdf = new RooFFTConvPdf("signalShapePdf","signalShapePdf", x, (const RooAbsPdf) rdh_what, cbX);
    float p2_init = p2.getVal();
    float p3_init = p3.getVal();
    RooRealVar alpha("alpha","alpha", 62.0,50,70 );
    RooRealVar beta("beta","beta",  0.001, 0.0, 0.1 );
    RooRealVar peak("peak","peak",  91.1876);
    RooRealVar gamma("gamma","gamma",  0.05, -0.1, 1.0 );
    CMSBkgLineShape bkgPdf("bkgPdf", "bkgPdf", x, alpha, beta, peak, gamma);
    alpha.setConstant(kTRUE);
    beta.setConstant(kTRUE);
    gamma.setConstant(kTRUE);
    peak.setConstant(kTRUE);
    float gamma_init = gamma.getVal();
    RooRealVar fracX("fracX","fraction of cbX",  1.0 , 0.0, 1.0);
    fracX.setConstant(kTRUE); //template method bkg hypothesis for pass pass case

    if(region == 2){
      gamma.setVal(1.4173e-02);
      p1.setVal(8.2633e-01);
      p1.setConstant(kTRUE);
      p2.setVal(8.5383e-04);
      p3.setVal(3.3236e-04);
      p4.setVal(5.8295e+01);
    }

    if( effType== 0 ){
      gamma.setConstant(kFALSE);
      //fracX.setVal(0.9);
      fracX.setConstant(kFALSE);
      p2.setConstant(kTRUE);
    }

    if((effType == 0 || effType == 1 || effType >= 3) && region != 0 && bothElorPos > 0 ){
      gamma.setRange(-10.0, 10.0);
      p1.setConstant(kTRUE);
      p2.setConstant(kFALSE);
      if( region == 2 ){
	fracX.setRange(-1.0,2.0);
	gamma.setConstant(kTRUE);
        //p1.setConstant(kFALSE);
        //p2.setConstant(kTRUE);
      }
    }

    if( effType == 1 && region == 2 && bothElorPos == 1 ){
      p2.setConstant(kTRUE);
      p1.setConstant(kFALSE);
    }


    if( !isData ){
      p1.setConstant(kTRUE);
      gamma.setConstant(kTRUE);
      fracX.setConstant(kTRUE);
    }

    RooAddPdf sum("sum", "sum", RooArgList(*signalShapePdf, bkgPdf), RooArgList(fracX) );
    RooDataHist* data_pass = new RooDataHist("data_pass","data_pass", RooArgList(x), hGSF_pass);
    RooDataHist* data_fail = new RooDataHist("data_fail","data_fail", RooArgList(x), hGSF_fail);
    x.setBins(30);
    //RooFitResult *fitResult_pass = sum.fitTo( *data_pass, RooFit::Save(true),RooFit::Extended(false), RooFit::PrintLevel(1),RooFit::Minos(kTRUE));//binned
    RooFitResult *fitResult_pass = sum.fitTo(d_pass, RooFit::Save(true),RooFit::Extended(false), RooFit::PrintLevel(1),RooFit::Minos(kTRUE));//unbinned

    RooPlot* frame1 = x.frame();
    frame1->SetMinimum(0);
    d_pass->plotOn(frame1,RooFit::(kBlack), RooFit::LineColor(kBlack));
    sum.plotOn(frame1, RooFit::(kBlue), RooFit::LineColor(kBlue));
    float chi2PerDof_pass = frame1->chiSquare(1);
    sum.plotOn(frame1,Components("bkgPdf"), RooFit::(kRed), RooFit::LineColor(kRed));

    frame1->Draw("e0");
    std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   pass: chi2/dof= "<<chi2PerDof_pass<<std::endl;
    
    std::cout<<"COUT PASSING FIT RESULT:"<<std::endl;
    fitResult_pass->Print("v");
    std::cout<<"DONE COUT'ING PASSING FIT RESULTS:"<<std::endl;

    float nPass = hGSF_pass->Integral() * fracX->getVal();
    float nBkgPass = hGSF_pass->Integral() *(1.0 -  fracX->getVal() );
    float nPassErr = hGSF_pass->Integral() * fracX->getError();
    if( effType == 1 || effType == 3 ){
      nPass = hGSF_pass->Integral() * 0.99; // 1% bkg from fake rate method 
      nBkgPass = hGSF_pass->Integral() *(1.0 -  0.99 ); // 1% bkg from template method 
      nPassErr = hGSF_pass->Integral() * 0.014; //1.4% error on the bkg from template method
    }

    if( effType == 2 || effType == 4 || !isData ){ //no background passes
      nPass = hGSF_pass->Integral(); 
      nBkgPass = 0.0;
      nPassErr = 0;
    }

    fracX.setVal(0.7);
    gamma.setVal(2.1027e-02);
    p1.setVal(2.8386e+00);
    p2.setVal(3.5282e-03);
    p3.setVal(4.0762e-04);
    p4.setVal(5.9024e+01);

    fracX.setConstant(kFALSE); //include bkg for fail case
    p2.setConstant(kFALSE);
    gamma.setConstant(kFALSE);//TEST                                                                                                                               
    p1.setConstant(kTRUE);//TEST 

    if(effType == 2 || effType == 4 ){
      p1.setConstant(kFALSE);
    }


    if ( (effType == 3 || effType == 4) && region!= 2 ){
      gamma.setVal(1.84739e-01);
      fracX.setRange(-1.0,2.0);
      p1.setVal(-4.4394e+00);
      p2.setVal(1.8673e-06);
      p3.setVal(5.4340e-04);
      p4.setVal(2.2096e+01);
      gamma.setConstant(kTRUE);
      p2.setConstant(kTRUE);
      p1.setConstant(kFALSE);
    }

    if(region == 1){
      gamma.setVal(3.0834e-02);
      p1.setVal(1.1237e+00);
      p2.setVal(9.2403e-04);
      p3.setVal(3.6280e-04);
      p4.setVal(2.9168e+00);
      p1.setConstant(kTRUE);
    }


    if(region == 2){
      gamma.setRange(-1.0,100.0);
      gamma.setVal(1.4173e-02);
      fracX.setRange(-1.0,2.0);
      if( effType >= 3 ){
        fracX.setRange(0.0,2.0);
        gamma.setVal(50.0);
      }
      //gamma.setVal(6.3795e+00);                                                                                                                                                                 
      p1.setVal(1.9879e+00);
      p2.setVal(2.6257e-03);
      p3.setVal(5.2198e-04);
      p4.setVal(5.9465e+01);
      p1.setConstant(kTRUE);
      gamma.setConstant(kTRUE);
    }

    if(effType == 0 && region == 2 && bothElorPos >= 1 ){
      fracX.setVal(1);
      fracX.setConstant(kTRUE); //not enough data.. don't even bother                                                                                            
    }
    if( effType >=3 && region == 2){
      fracX.setVal(1);
      fracX.setConstant(kTRUE); //not enough data.. don't even bother                                                                                            
    }


    if( effType >= 3 && region == 1 && bothElorPos == 1 ){
      fracX.setVal(1.0);
      fracX.setConstant(kTRUE); //not enough data.. don't even bother
      p2.setConstant(kFALSE);
    }

    if( effType == 2 && region == 2 && bothElorPos == 2 ){
      fracX.setRange(0,1.2);
      gamma.setVal(5.0);
      p2.setVal(2.4350e-03);
      p2.setConstant(kFALSE);
    }

    //except static for MC
    if( !isData ){
      gamma.setConstant(kTRUE);
      fracX.setConstant(kTRUE);
      p2.setConstant(kTRUE);
      p1.setConstant(kTRUE);
    }




    x.setBins(12);
    //RooFitResult *fitResult_fail = sum.fitTo(*data_fail, RooFit::Save(true),RooFit::Extended(false), RooFit::PrintLevel(1), RooFit::Minos(kTRUE));//binned
    RooFitResult *fitResult_fail = sum.fitTo(d_fail, RooFit::Save(true),RooFit::Extended(false), RooFit::PrintLevel(1),RooFit::Minos(kTRUE));//unbinned

    RooPlot* frame2 = x.frame();
    frame2->SetMinimum(0);
    d_fail->plotOn(frame2,RooFit::(kBlack), RooFit::LineColor(kBlack));
    //data_fail->plotOn(frame2,RooFit::(kBlack), RooFit::LineColor(kBlack));
    sum.plotOn(frame2, RooFit::(kBlue), RooFit::LineColor(kBlue));
    float chi2PerDof_fail = frame2->chiSquare(3);
    sum.plotOn(frame2,Components("bkgPdf"), RooFit::(kRed), RooFit::LineColor(kRed));
    std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   fail: chi2/dof= "<<chi2PerDof_fail<<std::endl;


    std::cout<<"COUT FAILING FIT RESULTS:"<<std::endl;
    fitResult_fail->Print("v");
    std::cout<<"DONE COUT'ING FAILING FIT RESULTS:"<<std::endl;
    //std::cout<<"WARNING: NOTE THAT ANY ***FIT*** PRINTOUT BELOW IS FROM AN ENTIRELY DIFFERENT FIT, NOT RELATED TO THE LINESHAPE EFF FITS"<<std::endl;
    float nFail = hGSF_fail->Integral() * fracX->getVal();
    float nBkgFail = hGSF_fail->Integral() * (1 - fracX->getVal() );
    float nFailErr = hGSF_fail->Integral() * fracX->getError();
    if(!isData){
      nPass =  hGSF_pass->Integral();
      nFail =  hGSF_fail->Integral();
    }
    float eff = nPass / (nPass + nFail); 
    float effErr = (1.0 - eff)*sqrt( (nPassErr/nPass)**2 + (nFailErr/nFail)**2 );
    //std::cout<<"nPassErr, nFailErr= "<< nPassErr <<", "<<nFailErr << std::endl;
    std::cout<< "LINESHAPE EFFICIENCY: "<< eff <<" +/- "<< effErr*sqrt(2.0)<<std::endl;
    std::cout<<"LINESHAPE RAW INTEGRAL pass, fail= "<< hGSF_pass->Integral() << std::setw(15) << hGSF_fail->Integral() << std::endl;
    std::cout<<"LINESHAPE SIGNAL pass, fail= "<< nPass <<" +/- "<< nPassErr <<std::setw(15)<<nFail <<" +/- "<< nFailErr << std::endl;
    std::cout<<"LINESHAPE BKG pass, fail= "<< nBkgPass <<" +/- "<<nPassErr <<", "<<nBkgFail<<" +/- "<< nFailErr << std::endl;


    TCanvas* c1 = new TCanvas(c1Name,c1Name,500,500);
    c1->cd();
    fout->cd();
    frame1->Draw("e0");
    char temp[100];
    sprintf(temp, "#chi^{2}/DOF = %.3f", chi2PerDof_pass);
    TPaveText *plotlabel8 = new TPaveText(0.65,0.80,0.85,0.77,"NDC");
    plotlabel8->AddText(temp);
    plotlabel8->SetTextColor(kBlack);
    plotlabel8->SetFillColor(kWhite);
    plotlabel8->SetBorderSize(0);
    plotlabel8->SetTextAlign(12);
    plotlabel8->SetTextSize(0.03);
    plotlabel8->Draw();

    TCanvas* c2 = new TCanvas(c2Name,c2Name,500,500);
    c2->cd();
    fout->cd();
    frame2->Draw("e0");
    TPaveText *plotlabel7 = new TPaveText(0.65,0.82,0.85,0.87,"NDC");
    plotlabel7->SetTextColor(kBlack);
    plotlabel7->SetFillColor(kWhite);
    plotlabel7->SetBorderSize(0);
    plotlabel7->SetTextAlign(12);
    plotlabel7->SetTextSize(0.03);
    TPaveText *plotlabel9 = new TPaveText(0.65,0.80,0.85,0.77,"NDC");
    plotlabel9->SetTextColor(kBlack);
    plotlabel9->SetFillColor(kWhite);
    plotlabel9->SetBorderSize(0);
    plotlabel9->SetTextAlign(12);
    plotlabel9->SetTextSize(0.03);
    sprintf(temp, "#chi^{2}/DOF = %.3f", chi2PerDof_fail);
    plotlabel9->AddText(temp);
    sprintf(temp, "#epsilon = %.3f #pm %.3f", eff, effErr);
    plotlabel7->AddText(temp);
    plotlabel9->Draw();
    plotlabel7->Draw();
    c1->cd();
    plotlabel7->Draw();
    c2->cd();
    plotlabel9->Draw();

    c1->Write();
    c2->Write();




    std::cout<<"nOSPass, nSSPass= "<<nOSPass<<", "<<nSSPass<<std::endl;
    std::cout<<"nOSFail, nSSFail= "<<nOSFail<<", "<<nSSFail<<std::endl;

    //from data <q1q2> = -0.965 +/- 0.009
    // ---> qMisID = (1 - sqrt(fabs(<q1q2>))/2 = 0.00883 +/- 0.00229
    //...alright, but that is just statistical.  adding in both stat and syst errors we get:
    //<q1q2> = -0.965 +/- 0.015


    //float qMisID= 100.0;//
    float qMisID= 0.00759057; //for WP70 qmisid
    //float qMisID= 0.00883; //for WP85 qmisid
    if( !isData ){ qMisID= 0.0123; }
    //float qMisID= 0.00883; //for WP80 numbers
    float qMisID_fail= qMisID;//here we correct for the pass-->fail bias from MC
    float nSigPass_fromQ = (float(nOSPass) - float(nSSPass))/(1 - 2.0*qMisID)**2;
    float nBkgPass_fromQ = (float(nOSPass) + float(nSSPass)) - nSigPass_fromQ;


    std::cout<<"nOSFail, nSSFail= "<<nOSFail<<", "<<nSSFail<<std::endl;

    float nSigFail_fromQ = (float(nOSFail) - float(nSSFail))/(1 - 2.0*qMisID_fail)**2;
    if(nSigFail_fromQ < 0 ){ nSigFail_fromQ = 0.0;}
    float nBkgFail_fromQ = (float(nOSFail) + float(nSSFail)) - nSigFail_fromQ;

    //float qMisID_biased= 0.00883 + 0.00229; //stat
    float qMisID_biased= qMisID + 0.00383; //stat+syst
    float qMisID_fail_biased= qMisID_fail + 1.0*1.97194399999999978e-02; //for WP95 numbers
    //float qMisID_fail_biased= qMisID + 1.97194399999999978e-02; //for WP80 numbers
    float nSigPass_fromQ_biased = (float(nOSPass) - float(nSSPass))/(1 - 2.0*qMisID_biased)**2;
    float nBkgPass_fromQ_biased = (float(nOSPass) + float(nSSPass)) - nSigPass_fromQ_biased;

    float nSigFail_fromQ_biased = (float(nOSFail) - float(nSSFail))/(1 - 2.0*qMisID_fail_biased)**2;
    float nBkgFail_fromQ_biased = (float(nOSFail) + float(nSSFail)) - nSigFail_fromQ_biased;

    float nSigPass_fromQ_err = fabs( nSigPass_fromQ - nSigPass_fromQ_biased );
    float nSigFail_fromQ_err = fabs( nSigFail_fromQ - nSigFail_fromQ_biased );
    float nBkgPass_fromQ_err = fabs( nBkgPass_fromQ - nBkgPass_fromQ_biased );
    float nBkgFail_fromQ_err = fabs( nBkgFail_fromQ - nBkgFail_fromQ_biased );





    std::cout<<"qMisID EFFICIENCY: "<<nSigPass_fromQ/(nSigPass_fromQ + nSigFail_fromQ)
	     << " +/- " <<
      (1.0/(nSigPass_fromQ + nSigFail_fromQ))*sqrt( nSigPass_fromQ*(1.0 - nSigPass_fromQ/(nSigPass_fromQ + nSigFail_fromQ))   )
	     << " (stat)+/- "<< 
      (1.0 - nSigPass_fromQ/(nSigPass_fromQ + nSigFail_fromQ))*sqrt( (nSigPass_fromQ_err/nSigPass_fromQ)**2 + (nSigFail_fromQ_err/((nSigFail_fromQ>0)?nSigFail_fromQ:nSigFail_fromQ_err) )**2 )<< " (syst)"<<std::endl;

    std::cout<<"qMisID SIGNAL pass, fail= "<<nSigPass_fromQ<<" +/- " << nSigPass_fromQ_err  <<", "<< nSigFail_fromQ<<" +/- " << nSigFail_fromQ_err <<std::endl;
    std::cout<<"qMisID BKG pass, fail= "<<nBkgPass_fromQ<<" +/- " << nBkgPass_fromQ_err  <<", "<< nBkgFail_fromQ<<" +/- " << nBkgFail_fromQ_err <<std::endl;

    //scratch the template method for now
    /*
    TH1F* DataPass_forTemplateMethod_test = ( (TH1F* ) templates_d->Get("RelTrkIso") ->Clone());
    DataPass_forTemplateMethod_test->Rebin(rebin);

    DataPass_forTemplateMethod_test->SetLineColor(kBlue);
    DataPass_forTemplateMethod_test->DrawNormalized();
    DataPass_forTemplateMethod->SetLineColor(kRed);
    DataPass_forTemplateMethod->DrawNormalized("same");

    TObjArray *Template = new TObjArray(2);
    Template->Add(SignalTemplate);
    Template->Add(BackgroundTemplate);
    TFractionFitter* fit = new TFractionFitter(DataPass_forTemplateMethod, Template);

    fit->Constrain(1, 0.0, 1.0);
    fit->Constrain(2, 0.0, 1.0);

    double theFrac[3], err[3];
    Int_t status = fit->Fit();

    cout << "PASS fit status: " << status << endl;
    if (status == 0 ) {                       // check on fit status                                                               
      TH2F* result = (TH2F*) fit->GetPlot();
      result->SetTitle("fit pass");

      for(int i=0; i<2; i++){
	fit->GetResult(i, theFrac[i], err[i]);
      }

      std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DONE FITTING PASSING !!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
      std::cout<<"PASS: fracS= "<< 1.0 - theFrac[1] <<" +/- "<< err[1] << std::endl;
    }

    TObjArray *TemplateF = new TObjArray(2);
    TemplateF->Add(SignalTemplate); //or could use SignalTemplate_fail for probes
    TemplateF->Add(BackgroundTemplate);//similarly...
    TFractionFitter* fitF = new TFractionFitter(DataFail_forTemplateMethod, TemplateF);
    fitF->Constrain(1, 0.0, 1.0);
    fitF->Constrain(2, 0.0, 1.0);
    double theFracF[3], errF[3];
    Int_t statusF = fitF->Fit();
    if (statusF == 0 ) {
      TH2F* resultF = (TH2F*) fitF->GetPlot();
      resultF->SetTitle("fit pass");

      for(int i=0; i<2; i++){
        fitF->GetResult(i, theFracF[i], errF[i]);
      }
      std::cout<<"FAIL: fracS= "<< 1.0 - theFracF[1] <<" +/- "<< errF[1] << std::endl;                                                                                                                                                 

      float nSigPass_temp =  DataPass_forTemplateMethod->GetEntries()*(1.0 - theFrac[1])/1.0;
      float nBkgPass_temp =  DataPass_forTemplateMethod->GetEntries()*theFrac[1]/1.0;
      float nSigFail_temp =  DataFail_forTemplateMethod->GetEntries()*(1.0 - theFracF[1])/1.0;
      float nBkgFail_temp =  DataFail_forTemplateMethod->GetEntries()*theFracF[1]/1.0;

      float nSigPass_temp_err =  DataPass_forTemplateMethod->GetEntries()*err[1]/1.0;
      float nBkgPass_temp_err =  DataPass_forTemplateMethod->GetEntries()*err[1]/1.0;
      float nSigFail_temp_err =  DataFail_forTemplateMethod->GetEntries()*errF[1]/1.0;
      float nBkgFail_temp_err =  DataFail_forTemplateMethod->GetEntries()*errF[1]/1.0;

      std::cout<<"template EFFICIENCY: "<<nSigPass_temp/(nSigPass_temp + nSigFail_temp)
	       << " +/- " << (1.0 - nSigPass_temp/(nSigPass_temp + nSigFail_temp))*sqrt( (nSigPass_temp_err/nSigPass_temp)**2 + (nSigFail_temp_err/nSigFail_temp)**2 )<< std::endl;

      std::cout<<"template SIGNAL pass, fail= "<<nSigPass_temp<<" +/- " << nSigPass_temp_err  <<", "<< nSigFail_temp<<" +/- " << nSigFail_temp_err <<std::endl;
      std::cout<<"template BKG pass, fail= "<<nBkgPass_temp<<" +/- " << nBkgPass_temp_err  <<", "<< nBkgFail_temp<<" +/- " << nBkgFail_temp_err <<std::endl;
 
      }
    */

  fout->Write();
}

