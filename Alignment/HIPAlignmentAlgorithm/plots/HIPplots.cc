#include "TROOT.h"
#include "TAttFill.h"
#include "TColor.h"
#include "HIPplots.h"
#include <string>
#include <sstream>
#include <cmath>

#include "TProfile.h"
#include "TPaveStats.h" 
#include "TList.h"
#include "TNtuple.h"
#include "TString.h"
#include "TTree.h"

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TLegend.h"

HIPplots::HIPplots(int IOV, char* path, char* outFile ){
	
	sprintf( _path, path );
	sprintf( _outFile, outFile );
	sprintf( _inFile_params, "%s/IOAlignmentParameters.root", _path ); //Id (raw id), ObjId, Par (alignable movements), CovariantMatrix
	sprintf( _inFile_uservars, "%s/IOUserVariables.root", _path ); //Id,ObjId,Nhit,Jtvj,Jtve,AlignableChi2,AlignableNdof
	sprintf( _inFile_truepos, "%s/IOTruePositions.root", _path ); //redundant
        sprintf( _inFile_alipos, "%s/IOAlignedPositions_%d.root", _path,IOV ); //aligned absolute positions
//	sprintf( _inFile_alipos, "%s/IOAlignedPositions_262922.root", _path ); //aligned absolute positions
//      sprintf( _inFile_alipos, "%s/IOAlignedPositions_272497.root", _path ); //aligned absolute positions
	sprintf( _inFile_mispos, "%s/IOMisalignedPositions.root", _path );
	sprintf( _inFile_HIPalign, "%s/HIPAlignmentAlignables.root", _path );
	sprintf( _inFile_surveys, "%s/HIPSurveyResiduals.root", _path );
	SetPeakThreshold(8.0);
	plotbadchi2=true;
}


TLegend * HIPplots::MakeLegend (double x1,
                                double y1,
                                double x2,
                                double y2)
{
    TLegend * legend = new TLegend (x1, y1, x2, y2, "", "NBNDC");
    legend->SetNColumns(6);
    legend->SetFillColor(0);
    legend->SetBorderSize(0);
    int COLOR_CODE[6]={28,2,3,4,6,7};

    TString detNames[6];
    detNames[0] = TString("PXB");
    detNames[1] = TString("PXF");
    detNames[2] = TString("TIB");
    detNames[3] = TString("TID");
    detNames[4] = TString("TOB");
    detNames[5] = TString("TEC");

    for (unsigned int isublevel = 0 ; isublevel < 6 ; isublevel++)
    {
        TGraph * g = new TGraph (0);
        g->SetLineColor(COLOR_CODE[isublevel]);
        g->SetMarkerColor(COLOR_CODE[isublevel]);
        g->SetFillColor(COLOR_CODE[isublevel]);
        g->SetFillColor(1);
        g->SetMarkerStyle(8);
        g->SetMarkerSize(1);
        legend->AddEntry(g,detNames[isublevel], "lp");
    }
    return legend;
}


void HIPplots::extractAlignParams(int currentPar, int minHits, int subDet, int doubleSided ){
	
  cout<<"--- extracting AlignParams ; Par "<<currentPar<<" ---"<<endl<<endl; //0-5: u,v,w,alpha,beta,gamma
	//void ExtractAlignPars(int pp, string SubDetString){
	//	const int par_now=pp;
	//load first tree
	
//	TFile *fp = new TFile( _inFile_params,"READ");
//	const TList* keysp = fp->GetListOfKeys();
//	const unsigned int maxIteration = keysp->GetSize() - 1;
	cout<<"Loaded par file -. OK"<<endl;
	TFile *fv = new TFile( _inFile_uservars,"READ");
	const TList* keysv = fv->GetListOfKeys();
        const unsigned int maxIteration = keysv->GetSize() - 1;

	char fileaction[16];
	if(CheckFileExistence(_outFile))sprintf(fileaction,"UPDATE");
	else sprintf(fileaction,"NEW");

	TFile *fout = new TFile( _outFile, fileaction);
	
	TTree* tree0 = (TTree*) fv->Get(keysv->At(0)->GetName() );
	//const char* tree0v_Name = keysv->At(0)->GetName();
	//tree0->AddFriend( tree0v_Name, fv );
	
	int nHit;
	unsigned int detId;
	double par[6];
	unsigned int detId0;
	
	tree0->SetBranchAddress("Id", &detId0 );
	
	const int ndets=tree0->GetEntries();
	TH1D *hpar1[ndets];
	char ppdirname[16];
	sprintf(ppdirname,"ShiftsPar%d", currentPar);
	fout->mkdir(ppdirname);
	gDirectory->cd(ppdirname);
	// delare histos


        TH1D *hpariter[maxIteration];
 	for(unsigned int a = 0; a < maxIteration; a++){
		char histoname[32],histotitle[32];
		sprintf(histoname,"Par_%d_Iter_%d",currentPar,a);
		sprintf(histotitle,"Par %d for iteration #%d",currentPar,a);
		hpariter[a]=new TH1D(histoname,histoname,1000,-1.0,1.0);
	}

	for(int i = 0; i < ndets; i++){
		char histoname[32],histotitle[32];
		sprintf(histoname,"Par_%d_SiDet_%d",currentPar,i);
		sprintf(histotitle,"Par %d for detector #%d",currentPar,i);
		hpar1[i]=new TH1D(histoname,histoname,maxIteration+1,-0.5,maxIteration+0.5);
	}

	//file where to save aligned det id list
	ofstream flist("./List_aligned_dets.txt",ios::out);
	

	//Loop on trees
	int modules_accepted = 0;
	for(unsigned int iter = 0; iter <= maxIteration; iter++){//loop on i (HIP iterations -> trees in file)
		
//		TTree* tmpTree = (TTree*) fp->Get(keysp->At(iter)->GetName() );
//		const char* tmpTreev = keysv->At(iter)->GetName();
                TTree* tmpTree = (TTree*) fv->Get(keysv->At(iter)->GetName() );
//		tmpTree->AddFriend( tmpTreev, fv ); // get access to both IOAlignmentParameters and IOUserVariables
		tmpTree->SetBranchAddress("Id", &detId );  
		tmpTree->SetBranchAddress("Nhit", &nHit ); // taken from IOUserVariables
		tmpTree->SetBranchAddress("Par", &par ); // taken from IOAlignmentParameters
		
		std::cout << "iteration: " << iter << "..." << std::endl;
		
		modules_accepted=0;
		//loop on entries
		for(int j = 0; j < tmpTree->GetEntries(); j++){ //loop on j (modules -> entries in each tree)
			
			tmpTree->GetEntry(j);
			bool passSubdetCut = true;
			if (subDet > 0){ if ( GetSubDet(detId) != subDet ) passSubdetCut = false; } // find the subDet from module detId 
			if (doubleSided > 0){ 
			  if ( GetSubDet(detId)%2 == 1 && doubleSided == 1 && GetBarrelLayer(detId) < 3 ) passSubdetCut = false;
			  if ( GetSubDet(detId)%2 == 1 && doubleSided == 2 && GetBarrelLayer(detId) > 2 ) passSubdetCut = false;
			}  

			if (( nHit >= minHits )&&( passSubdetCut )){  //cut on min nhits per module
				hpar1[j]->SetBinContent(iter+1,par[currentPar]); //x-iteration, y-movement in this iteration
                         //std::cout << "iteration: " << iter << ",par"<<currentPar<<"= "<<par[currentPar] << std::endl;

                       int COLOR_CODE[6]={28,2,3,4,6,7};
                       hpar1[j]->SetLineColor(COLOR_CODE[GetSubDet(detId)-1]);
                       hpar1[j]->SetMarkerColor(COLOR_CODE[GetSubDet(detId)-1]);

				// hpariter[iter]->Fill(par[currentPar]);
				//if(currentPar<3&&par[currentPar]>0.5)cout<<"LARGE PARCHANGE from DET "<<detId<<" in subdet "<< subDet<<". Par#"<<currentPar<<" shifted by "<<par[currentPar]<<" at iter "<<iter<<endl; 

				if((iter==maxIteration-1)&&(currentPar==0))flist<<detId<<endl;
				modules_accepted++;
				
			}
		}//end loop on entries (i.e. modules)
		delete tmpTree;
	}//end loop on iteration
	std::cout << "Modules accepted: " << modules_accepted << std::endl;
	std::cout << "Writing..." << std::endl;
	//Save 
	fout->Write();
	flist.close();
	// delete hpar1;
	std::cout << "Deleting..." << std::endl;
	delete fout;
	//	std::cout << "Deleting..." << std::endl;
//	delete fp;
	//std::cout << "Deleting..." << std::endl;
	delete fv;
}

void HIPplots::extractAlignShifts( int currentPar, int minHits, int subDet ){
	
  cout<<"\n--- extracting AlignShifts ; Par "<<currentPar<<" ---"<<endl<<endl;
	TFile *fa = new TFile( _inFile_alipos,"READ");
	const TList* keysa = fa->GetListOfKeys();
	const unsigned int maxIteration = keysa->GetSize();
	
	TFile *fv = new TFile( _inFile_uservars,"READ");
	const TList* keysv = fv->GetListOfKeys();
	
//	TFile *fm = new TFile( _inFile_mispos,"READ");
//      TFile *ft = new TFile( _inFile_truepos,"READ");
	
	char fileaction[16];
	if(CheckFileExistence(_outFile))sprintf(fileaction,"UPDATE");
	else sprintf(fileaction,"NEW");
	TFile *fout = new TFile( _outFile, fileaction);
	
	TTree* tree0 = (TTree*) fa->Get(keysa->At(0)->GetName() );
	const char* tree0v_Name = keysv->At(0)->GetName();
	tree0->AddFriend( tree0v_Name, fv );  //get access to both IOAlignedPositions and IOUserVariables
	
	unsigned int detId0;
	
	tree0->SetBranchAddress("Id", &detId0 );
	
	const int ndets=tree0->GetEntries();
	TH1D *hshift[ndets]; //hshift[i] absolute position change for det i, at all iterations
//	TH1D *hiter_mis[maxIteration]; //hiter_mis[i] misalignment position change at iteration i (defualt case : misalignment=0)
	TH1D *hiter_ali[maxIteration]; //hiter_ali[i] absolute position change at iteration i, for all detectors
	char ppdirname[16];
	sprintf(ppdirname,"Shifts%d", currentPar);
	fout->mkdir(ppdirname);
	gDirectory->cd(ppdirname);
	// delare histos
	for(unsigned int a = 0; a < maxIteration; a++){
		char histoname[64],histotitle[64];
//		sprintf(histoname,"MisalignedShift_%d_Iter_%d",currentPar,a);
//		sprintf(histotitle,"Misaligned Shift %d for iteration #%d",currentPar,a);
//		hiter_mis[a]=new TH1D(histoname,histoname,1000,-1.0,1.0);
		sprintf(histoname,"AlignedShift_%d_Iter_%d",currentPar,a);
		sprintf(histotitle,"Aligned Shift %d for iteration #%d",currentPar,a);
                hiter_ali[a]=new TH1D(histoname,histoname,ndets,0,ndets);
	}
	for(int i = 0; i < ndets; i++){
		char histoname[64],histotitle[64];
		sprintf(histoname,"Shift_%d_SiDet_%d",currentPar,i);
		sprintf(histotitle,"Shift %d for detector #%d",currentPar,i);
		hshift[i]=new TH1D(histoname,histoname,maxIteration,-0.5,maxIteration-0.5);
	}
	
	//Loop on trees
	int modules_accepted = 0;
	int nHit;
	unsigned int detId;
	double tpos[3];	
	double apos[3];	
//	double mpos[3];	
	double trot[9];	
	double arot[9];	
//	double mrot[9];	
        double aerr[6];

        TString detNames[6];
        detNames[0] = TString("PXB");
        detNames[1] = TString("PXF");
        detNames[2] = TString("TIB");
        detNames[3] = TString("TID");
        detNames[4] = TString("TOB");
        detNames[5] = TString("TEC");

	
//	TTree* tmpTree_m = (TTree*)fm->Get("AlignablesAbsPos_1"); //misaligned position 
//      TTree* tmpTree_true = (TTree*)ft->Get("AlignablesOrgPos_1");
        TTree* tmpTree_true = (TTree*)fa->Get("AlignablesAbsPos_0"); //starting position

	if (currentPar < 3){
//		tmpTree_m->SetBranchAddress("Pos", &mpos );
		tmpTree_true->SetBranchAddress("Pos", &tpos );
	}
	if (currentPar >= 3 ){
//		tmpTree_m->SetBranchAddress("Rot", &mrot );
		tmpTree_true->SetBranchAddress("Rot", &trot );
	}
	
	for(unsigned int iter = 1; iter < maxIteration; iter++){//loop on i (HIP iterations -> trees in file)
		
		TTree* tmpTree = (TTree*) fa->Get(keysa->At(iter)->GetName() );
		const char* tmpTreev = keysv->At(iter)->GetName();
		tmpTree->AddFriend( tmpTreev, fv );
		tmpTree->SetBranchAddress("Id", &detId );
		tmpTree->SetBranchAddress("Nhit", &nHit );
                tmpTree->SetBranchAddress("ParError", &aerr );

		if (currentPar < 3){ tmpTree->SetBranchAddress("Pos", &apos ); }
		tmpTree->SetBranchAddress("Rot", &arot );
		
		//std::cout << "iteration: " << iter << "..." << std::endl;
		modules_accepted=0;
		//loop on entries
		for(int j = 0; j < tmpTree->GetEntries(); j++){ //loop on j (modules -> entries in each tree)
                        tmpTree_true->GetEntry(j);
			tmpTree->GetEntry(j);
                        //cout<<"det"<<j<<","<<detId<<","<<nHit<<endl;
//			tmpTree_m->GetEntry(j);
//			tmpTree_true->GetEntry(j);


			bool passSubdetCut = true;
			int mysubDet=GetSubDet(detId);
			if (subDet > 0){ if ( GetSubDet(detId) != subDet ) passSubdetCut = false; } 

                       int COLOR_CODE[6]={28,2,3,4,6,7};
                       hshift[j]->SetLineColor(COLOR_CODE[GetSubDet(detId)-1]);
                       hshift[j]->SetMarkerColor(COLOR_CODE[GetSubDet(detId)-1]);

			if (( nHit >= minHits )&&( passSubdetCut )){
				// maths
				if (currentPar < 3){
//					TVectorD dr_mis(3, mpos);
					TVectorD dr_ali(3, apos);
					TVectorD r_true(3, tpos);
					TMatrixD R_true(3, 3, trot);
					//std::cout << "dr_ali 00 : " << dr_ali[currentPar] << std::endl;
//					dr_mis -= r_true;
					dr_ali -= r_true;
					//std::cout << "dr_ali 0 : " << dr_ali[currentPar] << std::endl;
					// to local
					//if (dr_mis != 0.) dr_mis = R_true * dr_mis;
					//if (dr_ali != 0.) dr_ali = R_true * dr_ali;
					//std::cout << "currentPar: " << currentPar << std::endl;
					//std::cout << "dr_mis: " << dr_mis[currentPar] << std::endl;
					//std::cout << "dr_ali: " << dr_ali[currentPar] << std::endl;
					//					if(currentPar<3&&dr_ali[currentPar]>1.0)cout<<"LARGE SHIFT for DET "<<detId<<" in subdet "<< mysubDet<<". Par#"<<currentPar<<" shifted by "<<dr_ali[currentPar]<<" at iter "<<iter<<endl;
					hshift[j]->SetBinContent(iter+1,dr_ali[currentPar]); //aligned position - start position
//                                        hiter_mis[iter]->SetBinContent(j+1,dr_mis[currentPar]);
                                       //if(j=0&&currentPar==5) std::cout << "iter="<<iter<<",dr_ali: " << dr_ali[currentPar] << std::endl;
//                                        cout << "iter="<<iter<<"dr_ali: " << dr_ali[currentPar] << endl;
                                        hiter_ali[iter]->SetBinContent(j+1,dr_ali[currentPar]);
//                                        cout << "bin: " <<hiter_ali[iter]->GetBinContent(j+1)<<endl;
//                                        hiter_ali[iter]->SetBinError(j+1,5);
				}
				if (currentPar >= 3){
//					TMatrixD dR_mis(3, 3, mrot);
					TMatrixD dR_ali(3, 3, arot);
					TMatrixD R_true(3, 3, trot);
//					dR_mis = dR_mis * TMatrixD(TMatrixD::kTransposed, R_true);
					dR_ali = dR_ali * TMatrixD(TMatrixD::kTransposed, R_true);
					//-std::atan2(dR(2, 1), dR(2, 2)), std::asin(dR(2, 0)), -std::atan2(dR(1, 0), dR(0, 0)));
//					double dR_mis_euler = 0;
					double dR_ali_euler = 0;
					if (currentPar == 3 ){
//						dR_mis_euler = -std::atan2(dR_mis(2, 1), dR_mis(2, 2));
						dR_ali_euler = -std::atan2(dR_ali(2, 1), dR_ali(2, 2));						
					}
					if (currentPar == 4 ){
//						dR_mis_euler = -std::asin(dR_mis(2, 0));
						dR_ali_euler = -std::asin(dR_ali(2, 0));
					}
					if (currentPar == 5 ){
//						dR_mis_euler = -std::atan2(dR_mis(1, 0), dR_mis(0, 0));
						dR_ali_euler = -std::atan2(dR_ali(1, 0), dR_ali(0, 0));
					}
					hshift[j]->SetBinContent(iter+1,dR_ali_euler);
//                                        hiter_mis[iter]->SetBinContent(j+1,dR_mis_euler);
                                        hiter_ali[iter]->SetBinContent(j+1,dR_ali_euler);
				}
				modules_accepted++;
                                        hiter_ali[iter]->GetXaxis()->SetBinLabel(j+1,detNames[GetSubDet(detId)-1]);
                                        hiter_ali[iter]->SetBinError(j+1,aerr[currentPar]);
                                        //hiter_ali[iter]->SetBinError(j+1,0.001);
                                        hiter_ali[iter]->LabelsOption("u","x");
			}
		}
//		delete tmpTree;
	}
              delete tmpTree_true;

	std::cout <<"Modules accepted: " << modules_accepted << std::endl;
	std::cout << "Writing..." << std::endl;
	//Save 
	fout->Write();
	// delete hpar1;
	std::cout << "Deleting..." << std::flush;
	delete fout;
	std::cout << "Deleting..." << std::flush;
	delete fa;
//	delete ft;
//	delete fm;
	std::cout << "Deleting..." << std::endl;
	delete fv;
	
	
}

void HIPplots::plotAlignParams(string ShiftsOrParams, char* plotName){
	

  cout<<"_|_| plotting AlignParams |_|_"<<endl<<"---> "<<ShiftsOrParams <<endl;
	bool bParams = false;
	bool bShifts = false;
	if (ShiftsOrParams == "PARAMS") bParams = true;
	if (ShiftsOrParams == "SHIFTS") bShifts = true;

	int i = 0;

	TFile *f = new TFile( _outFile,"READ" );
	//	f->ls();

        TCanvas *c_params = new TCanvas("can_params","CAN_PARAMS",1200,900);
	c_params->Divide(3,2);
	//	cout<<"(1) I am in "<<gDirectory->GetPath()<<endl;
	TDirectory* d;
	int ndets = 0;
	if (bParams){
		d = (TDirectory*)f->Get("ShiftsPar0"); 
		ndets = GetNIterations(d,"Par_0_SiDet_");
		//	std::cout << "INHERE!" << std::endl;
	}
	if (bShifts){ 
		d = (TDirectory*)f->Get("Shifts0");
		ndets = GetNIterations(d,"Shift_0_SiDet_");
		//	std::cout << "INHERE!" << std::endl;
	}
		
	for(int iPar = 0; iPar < 6; iPar++){
		
		c_params->cd( iPar+1 );
		gPad->SetTopMargin(0.15);
		gPad->SetBottomMargin(0.15);
		char ppdirname[32];
		if (bParams) sprintf( ppdirname,"ShiftsPar%d", iPar ); 
		if (bShifts) sprintf( ppdirname,"Shifts%d", iPar );
		if(iPar > 0)gDirectory->cd("../");
		gDirectory->cd(ppdirname);
		//	cout<<"(2) I am in "<<gDirectory->GetPath()<<endl;
	
		TH1D *hpar1[ndets];
		char histoname[16];
		int sampling_ratio=1;
		int ndets_plotted=(int)ndets/sampling_ratio;
		cout<<"Plotting "<<ndets_plotted<<" detectors over a total of "<<ndets<<endl;
		i=0;
		double histomax, histomin;
		if(iPar>=3){    // alpha, beta, gamma
		  histomax=0.5;// in mrad
		  histomin=-0.5;
		}
		else if (iPar>=2) {  // w
		  if(bShifts){
		    histomax=200.0;//in microns
		    histomin=-200.0;
		  }
		  else{
		    histomax=100.0;//in microns
		    histomin=-100.0;
		  }
		}
		else {  // u, v
		  if(bShifts){
		    histomax=200.0;//in microns
		    histomin=-200.0;
		  }
		  else{
		    histomax=100.0;//in microns
		    histomin=-100.0;
		  }
		}
		while((i<ndets_plotted) && (i*sampling_ratio<ndets) ){
			if (bParams) sprintf(histoname,"Par_%d_SiDet_%d",iPar,i*sampling_ratio); 
			if (bShifts) sprintf(histoname,"Shift_%d_SiDet_%d",iPar,i*sampling_ratio); 
			hpar1[i]=(TH1D*)gDirectory->Get(histoname);

			if(iPar>=3)hpar1[i]->Scale(1000.0); //convert from rad to mrad
			else hpar1[i]->Scale(10000.0); //convert from cm to um

			hpar1[i]->SetMarkerStyle(7);
			hpar1[i]->SetStats(0);

			double tmpmax=hpar1[i]->GetBinContent(hpar1[i]->GetMaximumBin());
			double tmpmin=hpar1[i]->GetBinContent(hpar1[i]->GetMinimumBin());
			
			//Auto-adjust axis range
                        if(tmpmax>histomax)histomax=tmpmax*1.2;
                        if(tmpmin<histomin)histomin=tmpmin*1.2;

			//if(i%1300==0)cout<<"Actual maximum is "<<histomax<<" Actual minimum is "<<histomin<<endl;
			if(i==0){
				
				hpar1[i]->SetXTitle("Iteration");

                                if (bParams){
				if(iPar==0)hpar1[i]->SetYTitle("#delta u param (#mum)");
				else if(iPar==1)hpar1[i]->SetYTitle("#delta v param (#mum)");
				else if(iPar==2)hpar1[i]->SetYTitle("#delta w param (#mum)");
				else if(iPar==3)hpar1[i]->SetYTitle("#delta #alpha param (mrad)");
				else if(iPar==4)hpar1[i]->SetYTitle("#delta #beta param (mrad)");
				else if(iPar==5)hpar1[i]->SetYTitle("#delta #gamma param (mrad)");
				else hpar1[i]->SetYTitle("dunno");}
                                else if (bShifts){
                                if(iPar==0)hpar1[i]->SetYTitle("#delta u shift (#mum)");
                                else if(iPar==1)hpar1[i]->SetYTitle("#delta v shift (#mum)");
                                else if(iPar==2)hpar1[i]->SetYTitle("#delta w shift (#mum)");
                                else if(iPar==3)hpar1[i]->SetYTitle("#delta #alpha shift (mrad)");
                                else if(iPar==4)hpar1[i]->SetYTitle("#delta #beta shift (mrad)");
                                else if(iPar==5)hpar1[i]->SetYTitle("#delta #gamma shift (mrad)");
                                else hpar1[i]->SetYTitle("dunno");}

                                hpar1[i]->GetYaxis()->SetTitleOffset(1.5);
				//  hpar1[i]->SetTitle("Par1: x shift");
				hpar1[i]->SetTitle("");
				hpar1[i]->SetMaximum(histomax);
				hpar1[i]->SetMinimum(histomin);
				hpar1[i]->Draw("PL");
			}

			else hpar1[i]->Draw("PLsame");
			i++;
		}//end loop on NDets
		hpar1[0]->SetMaximum(histomax);
		hpar1[0]->SetMinimum(histomin);
                //hpar1[0]->SetLineColor(1);
		cout<<"Plotted "<<i<<" aligned detectors"<<endl;
	}//end loop on pars
        c_params->cd();
        TLegend * legend = MakeLegend(.1,.93,.9,.98);
        legend->Draw();
	c_params->SaveAs(plotName);
	std::cout << "Deleting..." << std::flush;
	delete c_params;
	std::cout << "Deleting..." << std::flush;
	//delete f;
	std::cout << "Deleting..." << std::endl;
		
}//end PlotAlignPars



void HIPplots::plotAlignParamsAtIter( int iter, string ShiftsOrParams, char* plotName ){

  cout<<"Welcome to  HIPplots::plotAlignParamsAtIter "<<iter<<endl;	
	bool bParams = false;
	bool bShifts = false;
	if (ShiftsOrParams == "PARAMS") bParams = true;
	else if (ShiftsOrParams == "SHIFTS") bShifts = true;
	else {cout<<"ERROR in plotAliParamsAtIter!!! Wrong input argument: "<<ShiftsOrParams<<" . Exiting"<<endl; return;}
	
	int i = 0;
	TFile *f = new TFile( _outFile,"READ" );
	//f->ls();
	
	TCanvas *c_params = new TCanvas("can_params","CAN_PARAMS",1200,700);
	c_params->Divide(3,2);
	//cout<<"(1) I am in "<<gDirectory->GetPath()<<endl;
	
	//TDirectory *d = (TDirectory*)f->Get("ShiftsPar0");
	//const int ndets = GetNIterations(d,"Par_0_SiDet_");
	
	for(int iPar = 0; iPar < 6; iPar++){
		
		c_params->cd( iPar+1 );
		
		gPad->SetTopMargin(0.15);
		gPad->SetBottomMargin(0.15);
		char ppdirname[32];
		if (bParams) sprintf( ppdirname,"ShiftsPar%d", iPar );
		if (bShifts) sprintf( ppdirname,"Shifts%d", iPar );
		if(iPar > 0)gDirectory->cd("../");
		gDirectory->cd(ppdirname);
		//cout<<"(2) I am in "<<gDirectory->GetPath()<<endl;
		
		TH1D *hiter;
		char histoname[16];
		if (bParams) sprintf( histoname, "Par_%d_Iter_%d", iPar, iter );
		if (bShifts) sprintf( histoname, "AlignedShift_%d_Iter_%d", iPar, iter );
		hiter = (TH1D*) gDirectory->Get(histoname);
                     

		//hiter->SetMarkerStyle(1);
		hiter->SetStats(1);
		
		hiter->SetXTitle("Sub-Det");
		if(iPar==0)hiter->SetYTitle("#delta u (#mum)");
		else if(iPar==1)hiter->SetYTitle("#delta v (#mum)");
		else if(iPar==2)hiter->SetYTitle("#delta w (#mum)");
		else if(iPar==3)hiter->SetYTitle("#delta #alpha (#murad)");
		else if(iPar==4)hiter->SetYTitle("#delta #beta (#murad)");
		else if(iPar==5)hiter->SetYTitle("#delta #gamma (#murad)");
		else hiter->SetXTitle("dunno");
                hiter->GetYaxis()->SetTitleOffset(1.5);
      
                double histomax,histomin;
                  if(iPar==2||iPar==5) {histomax=200; histomin=-200;}
                  else {histomax=100; histomin=-100;}
                   
                if(iPar>=3)hiter->Scale(1000000.0); //convert from rad to micro rad
                else hiter->Scale(10000.0); //convert from cm to um
                double tmpmax=hiter->GetBinContent(hiter->GetMaximumBin());
                double tmpmin=hiter->GetBinContent(hiter->GetMinimumBin());
                if(tmpmax>histomax)histomax=tmpmax*1.2;
                if(tmpmin<histomin)histomin=tmpmin*1.2;
                hiter->SetMaximum(histomax);
                hiter->SetMinimum(histomin);
                //hiter->SetAxisRange(0, 6, "X");
                hiter->SetLineColor(1);
                TColor *col = gROOT->GetColor(kCyan+1);
                col->SetAlpha(0.3);
                hiter->SetFillColor(col->GetNumber());
//                hiter->SetFillColor(kCyan);
                hiter->SetMarkerStyle(7);
		hiter->Draw();
                hiter->Draw("PE1");
/*
		if (bShifts) {
			TH1D *hiter2;
			char histoname2[16];
                        sprintf( histoname2, "MisalignedShift_%d_Iter_%d", iPar, iter );
			hiter2 = (TH1D*) gDirectory->Get(histoname2);
			hiter2->SetLineColor(1);
                        hiter2->SetFillColor(kRed-9);
			hiter2->Draw("SAME");
		}
		
*/		
//		cout<<"Plotted "<<i<<" aligned detectors"<<endl;
	}//end loop on pars
	c_params->SaveAs(plotName);
	delete c_params;
	std::cout << "Deleting..." << std::endl;
	delete f;
	
}//end PlotAlignParamsAtIter



void HIPplots::extractAlignableChiSquare( int minHits, int subDet, int doubleSided){

  //TFile *fp = new TFile( _inFile_params,"READ");
  //	const TList* keysp = fp->GetListOfKeys();
  //	const unsigned int maxIteration = keysp->GetSize();
  cout<<"\n--- Welcome to extractAlignableChiSquare ---"<<endl;
  cout<<"\nInput parameters:\n\tMinimum number of hits per alignbale = "<<minHits<<"\n\tSubdetetctor selection"<<subDet<<endl;

  if(minHits<1){
    cout<<"Warning ! Allowing to select modules with NO hits. Chi2 not defined for them. Setting automatically minNhits=1 !!!"<<endl;
    minHits=1;
  }  
  
  TFile *fv = new TFile( _inFile_uservars,"READ");
  const TList* keysv = fv->GetListOfKeys();
  const unsigned int maxIteration = keysv->GetSize() - 1;
  cout<<"MaxIteration is "<<maxIteration <<endl;

  char fileaction[16];
  if(CheckFileExistence(_outFile))sprintf(fileaction,"UPDATE");
  else sprintf(fileaction,"NEW");
  TFile *fout = new TFile( _outFile, fileaction);
 
  TTree* tree0 = (TTree*) fv->Get(keysv->At(0)->GetName() );
  unsigned int detId0;
  tree0->SetBranchAddress("Id", &detId0 );
  const int ndets=tree0->GetEntries();
  TH1D *halichi2n[ndets];//norm chi2 for each module as a function of iteration
  TH1D *htotchi2n[maxIteration];//distrib of norm chi2 for all modules at a given iteration
  TH1D *hprobdist[maxIteration];
  char ppdirname[16];
  sprintf(ppdirname,"AlignablesChi2n");
  fout->mkdir(ppdirname);
  gDirectory->cd(ppdirname);
  
  for(int iter = 1; iter <=(int)maxIteration ; iter++){
    char histoname[64],histotitle[128];
    sprintf(histoname,"Chi2n_iter%d",iter);
    sprintf(histotitle,"Distribution of Normalised #chi^{2} for All alignables  at iter %d",iter);
    htotchi2n[iter-1]=new TH1D(histoname,histotitle,1000,0.0,10.0);
    sprintf(histoname,"ProbChi2_%d",iter);
    sprintf(histotitle,"Distribution of  Prob(#chi^{2},ndof) at iter %d",iter);
    hprobdist[iter-1]=new TH1D(histoname,histotitle,100,0.0,1.0);
  }
  gDirectory->mkdir("AlignablewiseChi2n");
  gDirectory->cd("AlignablewiseChi2n");
  //declare histos
    for(int i = 0; i < (int)ndets; i++){
    tree0->GetEntry(i);
    char histoname[64],histotitle[64];
    sprintf(histoname,"Chi2n_%d",i);
    sprintf(histotitle,"Normalised #chi^{2} for detector #%d",i);
    halichi2n[i]=new TH1D(histoname,histotitle,maxIteration,0.5,maxIteration+0.5);
  }

  //Loop on trees and fill histos
  int modules_accepted = 0;
  for(unsigned int iter = 1; iter <= maxIteration; iter++){//loop on i (HIP iterations -> trees in file)

    TTree* tmpTreeUV = (TTree*) fv->Get(keysv->At(iter)->GetName() ); //get the UserVariable tree at each iteration
    cout<<"Taking tree "<<keysv->At(iter)->GetName()<<endl; 
    //tmpTreeUV->GetListOfLeaves()->ls();
    tmpTreeUV->SetBranchStatus("*",0);
    tmpTreeUV->SetBranchStatus("Id",1);
    tmpTreeUV->SetBranchStatus("Nhit",1);
    tmpTreeUV->SetBranchStatus("AlignableChi2",1);
    tmpTreeUV->SetBranchStatus("AlignableNdof",1);
    double alichi2=0.0;
    unsigned int alindof=0;
    int nHit=0;
    unsigned int detId=0;
    tmpTreeUV->SetBranchAddress("AlignableChi2",&alichi2);
    tmpTreeUV->SetBranchAddress("AlignableNdof",&alindof);
    tmpTreeUV->SetBranchAddress("Id",&detId);
    tmpTreeUV->SetBranchAddress("Nhit",&nHit);
    
    modules_accepted=0;
    //loop on entries
    for(int j = 0; j < tmpTreeUV->GetEntries(); j++){ //loop on j (modules -> entries in each tree)
      tmpTreeUV->GetEntry(j);

      bool passSubdetCut = true;
      if (subDet > 0){ if ( GetSubDet(detId) != subDet ) passSubdetCut = false; } //get subDet Id from module detId,and plot only the selected subDet 
      if (doubleSided > 0){ 
	if ( GetSubDet(detId)%2 == 1 && doubleSided == 1 && GetBarrelLayer(detId) < 3 ) passSubdetCut = false;
	if ( GetSubDet(detId)%2 == 1 && doubleSided == 2 && GetBarrelLayer(detId) > 2 ) passSubdetCut = false;
      }

      if (( nHit >= minHits )&&( passSubdetCut ) ){ //cut on nhits per module
	halichi2n[j]->SetBinContent(iter,double(alichi2 / alindof));
//        halichi2n[j]->SetBinContent(iter,double(alichi2));
	double prob=TMath::Prob(double(alichi2),int(alindof));
	
	htotchi2n[iter-1]->Fill(double(alichi2 / alindof));
	hprobdist[iter-1]->Fill(prob);
	modules_accepted++;
      }
      
    }//end loop on j - alignables
    cout<<"alignables accepted at iteration "<<iter<<" = "<<modules_accepted<<endl;
    delete tmpTreeUV;
  }//end loop on iterations

  // Ma che e'???
  /* cout<<"Prob for chi2=0,02 ndof=40 -> "<<TMath::Prob(0.02,40)<<endl;
  cout<<"Prob for chi2=20, ndof=40 -> "<<TMath::Prob(20.0,40)<<endl;
  cout<<"Prob for chi2=40, ndof=40 -> "<<TMath::Prob(40.0,40)<<endl;
  cout<<"Prob for chi2=60, ndof=40 -> "<<TMath::Prob(60.0,40)<<endl;
  cout<<"Prob for chi2=2000, ndof=40 -> "<<TMath::Prob(2000.0,40)<<endl; */

  //Save 
  fout->Write();
  delete fout;
  delete fv;
  cout<<"Finished extractAlignableChiSquare"<<endl;
}//end extractAlignableChiSquare


void HIPplots::extractSurveyResiduals( int currentPar,int subDet ){

  cout<<"\n---  extractSurveyResiduals has been called ---"<<endl;
   
  TFile *fv = new TFile( _inFile_surveys,"READ");
  const TList* keysv = fv->GetListOfKeys();
  const unsigned int maxIteration = keysv->GetSize() ;
  cout<<"MaxIteration is "<<maxIteration <<endl;

  char fileaction[16];
  if(CheckFileExistence(_outFile))sprintf(fileaction,"UPDATE");
  else sprintf(fileaction,"NEW");
  TFile *fout = new TFile( _outFile, fileaction);
 
  TTree* tree0 = (TTree*) fv->Get(keysv->At(0)->GetName() );
  unsigned int detId0;
  tree0->SetBranchAddress("Id", &detId0 );
  const int ndets=tree0->GetEntries();
  TH1D *hsurvey[ndets];//norm chi2 for each module as a function of iteration
  TH1D *htotres[maxIteration];//distrib of norm chi2 for all modules at a given iteration

  char ppdirname[16];
  sprintf(ppdirname,"SurveyResiduals");
  fout->mkdir(ppdirname);
  gDirectory->cd(ppdirname);
  
  for(int iter = 1; iter <=(int)maxIteration ; iter++){
    char histoname[64],histotitle[128];
    sprintf(histoname,"SurveyRes_Par%d_iter%d",currentPar,iter);
    sprintf(histotitle,"Distribution of survey Residuals for All alignables ; Par=%d ; iter %d",currentPar,iter);
    htotres[iter-1]=new TH1D(histoname,histotitle,1000,0.0,10.0);
  }
  gDirectory->mkdir("SurveyResiduals_alignables");
  gDirectory->cd("SurveyResiduals_alignables");
  //declare histos
    for(int i = 0; i < (int)ndets; i++){
    tree0->GetEntry(i);
    char histoname[64],histotitle[64];
    sprintf(histoname,"SurveyRes_Par%d_%d",currentPar,i);
    sprintf(histotitle,"Survey residual for detector #%d - Par=%d",i,currentPar);
    hsurvey[i]=new TH1D(histoname,histotitle,maxIteration,0.5,maxIteration+0.5);
  }

  //Loop on trees and fill histos
  int modules_accepted = 0;
  for(unsigned int iter = 1; iter <= maxIteration; iter++){//loop on i (HIP iterations -> trees in file)

    TTree* tmpTreeUV = (TTree*) fv->Get(keysv->At(iter)->GetName() ); //get the UserVariable tree at each iteration
    cout<<"Taking tree "<<keysv->At(iter)->GetName()<<endl; 
    //tmpTreeUV->GetListOfLeaves()->ls();
    tmpTreeUV->SetBranchStatus("*",0);
    tmpTreeUV->SetBranchStatus("Id",1);
    tmpTreeUV->SetBranchStatus("Par",1);

    double par[6];
    unsigned int detId=0;
    tmpTreeUV->SetBranchAddress("Par",&par);
    tmpTreeUV->SetBranchAddress("Id",&detId);

    
    modules_accepted=0;
    //loop on entries
    for(int j = 0; j < tmpTreeUV->GetEntries(); j++){ //loop on j (modules -> entries in each tree)
      tmpTreeUV->GetEntry(j);

      bool passSubdetCut = true;
      if (subDet > 0){ if ( GetSubDet(detId) != subDet ) passSubdetCut = false; } 
      if ( passSubdetCut  ){
	hsurvey[j]->SetBinContent(iter,double(par[currentPar]));
	
	modules_accepted++;
      }
      
    }//end loop on j - alignables
    cout<<"alignables accepted at iteration "<<iter<<" = "<<modules_accepted<<endl;
    delete tmpTreeUV;
  }//end loop on iterations

  //Save 
  fout->Write();
  delete fout;
  delete fv;
  cout<<"Finished extractAlignableChiSquare"<<endl;
}//end extractAlignableChiSquare






void HIPplots::plotAlignableChiSquare(char *plotName, float minChi2n){

  int i = 0;
  TFile *f = new TFile( _outFile,"READ" );
  TCanvas *c_alichi2n=new TCanvas("can_alichi2n","CAN_ALIGNABLECHI2N",900,900);
  c_alichi2n->cd();
  TDirectory *chi_d1=(TDirectory*)f->Get("AlignablesChi2n");
  const int maxIteration= GetNIterations(chi_d1,"Chi2n_iter",1)-1;
  cout<<"N iterations "<<maxIteration<<endl;
  //take the histos prepared with extractAlignableChiSquare
  gDirectory->cd("AlignablesChi2n");
  TDirectory *chi_d=(TDirectory*)gDirectory->Get("AlignablewiseChi2n");
  const int ndets= GetNIterations(chi_d,"Chi2n_");
  
  gDirectory->cd("AlignablewiseChi2n");
  TH1D *hchi2n[ndets];
  char histoname[64];
  int sampling_ratio=1;
  int ndets_plotted=(int)ndets/sampling_ratio;
  cout<<"Sampling "<<ndets_plotted<<" detectors over a total of "<<ndets<<endl;
  double histomax=0.1, histomin=-0.1;
  bool firstplotted=false;
  int firstplottedindex=0;
  int totalplotted=0;
  bool plothisto=true;
  while((i<ndets_plotted) && (i*sampling_ratio<ndets) ){
    sprintf(histoname,"Chi2n_%d",i);
    hchi2n[i]=(TH1D*)gDirectory->Get(histoname);

    //if required, check that the chi2n exceeds at any point the threshold set as input
    bool chi2ncut=false;
    if(minChi2n>0.0){
      for(int bin=1;bin<=hchi2n[i]->GetNbinsX();bin++){
	if(hchi2n[i]->GetBinContent(bin)>minChi2n){chi2ncut=true;break;}
      }
    }
    else chi2ncut=true;
    
    //if required check that the chi2 is increasing three iterations in a row and plot only those histos
    if(plotbadchi2)plothisto=CheckHistoRising(hchi2n[i]);
    else plothisto=true;

    if(chi2ncut&&plothisto){
      double tmpmax=hchi2n[i]->GetBinContent(hchi2n[i]->GetMaximumBin());
      double tmpmin=hchi2n[i]->GetBinContent(hchi2n[i]->GetMinimumBin());
      histomax=2.0;
      histomin=0.0;
      if(tmpmax>histomax)histomax=tmpmax;
      if(tmpmin<histomin)histomin=tmpmin;

      hchi2n[i]->SetMaximum(histomax);
      hchi2n[i]->SetMinimum(histomin);
      hchi2n[i]->SetStats(0);

      
      if(!firstplotted){
      	hchi2n[i]->SetXTitle("Iteration");
	hchi2n[i]->SetYTitle("#chi^{2} / # dof");
//        hchi2n[i]->SetYTitle("#chi^{2}");
	hchi2n[i]->SetTitle("Reduced #chi^{2} for alignables");
	hchi2n[i]->Draw("PL");   
	firstplotted=true;
	firstplottedindex=i;
      }
      else hchi2n[i]->Draw("PLsame");
      totalplotted++;
    }//END IF plot it because it passed the chi2n cut
    i++;
  }//end while loop on alignables
  hchi2n[firstplottedindex]->SetMaximum(histomax*1.2);
  hchi2n[firstplottedindex]->SetMinimum(histomin*0.8);
  //override auto-resize of axis scale
  hchi2n[firstplottedindex]->SetMaximum(histomax);
  hchi2n[firstplottedindex]->SetMinimum(histomin);

  cout<<"Plotted "<<totalplotted<<" alignables over an initial sample of "<<ndets_plotted<<endl;
  TText *txtchi2n_1=new TText();
  txtchi2n_1->SetTextFont(63);
  txtchi2n_1->SetTextSize(22);
  char strchi2n_1[128];
  sprintf(strchi2n_1,"Plotted alignables (Chi2n > %.3f): %d / %d",minChi2n,totalplotted,ndets_plotted);
  txtchi2n_1->DrawText(1.2,0.0,strchi2n_1);
  char finplotname[192];
  sprintf(finplotname,"%s.png",plotName);
  c_alichi2n->SaveAs(finplotname);
  delete c_alichi2n;
  //delete hchi2n;


  cout<<"Doing distrib"<<endl;
  gDirectory->cd("../");
  TCanvas *c_chi2ndist=new TCanvas("can_chi2ndistr","CAN_CHI2N_DISTRIBUTION",900,900);
  c_chi2ndist->cd();
  TH1D *hiter[maxIteration];
  TLegend *l=new TLegend(0.7,0.7,0.9,0.9);
  l->SetFillColor(0);
  l->SetBorderSize(0);
  int colors[10]={1,2,8,4,6,7,94,52,41,45};
  int taken=0;
  float tmpmax=1.0;
  float newmax=0.0;
  for(i=0;i<maxIteration;i++){
    sprintf(histoname,"Chi2n_iter%d",i+1);
    hiter[i]=(TH1D*)gDirectory->Get(histoname);
    hiter[i]->SetXTitle("#chi^{2} / dof");
    hiter[i]->SetYTitle("Alignables");
    hiter[i]->SetTitle("Normalised #chi^{2} of Alignables");
    hiter[i]->Rebin(5);
    hiter[i]->GetXaxis()->SetRangeUser(0.0,3.0);
    hiter[i]->SetLineColor(i);

    char legend_entry[64];
    float histmean=hiter[i]->GetMean();
    if(i==0){
      hiter[i]->SetLineColor(colors[taken]);
      taken++;
      sprintf(legend_entry,"Norm #chi^{2}; Iter %d; %.4f",i,histmean);
      l->AddEntry(hiter[i],legend_entry,"L");
      tmpmax=hiter[i]->GetBinContent(hiter[i]->GetMaximumBin());
      newmax=tmpmax;
      //      hiter[i]->Draw("");
    }
    else if((i+1)%5==0){
      hiter[i]->SetLineColor(colors[taken]);
      taken++;
      sprintf(legend_entry,"Norm #chi^{2}; Iter %d; %.4f",i+1,histmean);
      l->AddEntry(hiter[i],legend_entry,"L");
      tmpmax=hiter[i]->GetBinContent(hiter[i]->GetMaximumBin());
      if(tmpmax>newmax)newmax=tmpmax;
      //      hiter[i]->Draw("same");
    }
  }
  cout<<"NewMax after 1st loop -> "<<newmax<<endl;

  for(i=0;i<maxIteration;i++){
    hiter[i]->SetMaximum(newmax*1.1);
   if(i==1)      hiter[i]->Draw("");
   else if((i+1)%5==0) hiter[i]->Draw("same");
  }
  l->Draw();

  sprintf(finplotname,"%s_distr.png",plotName);
  cout<<finplotname<<endl;
  c_chi2ndist->SaveAs(finplotname);
  c_chi2ndist->SetLogy();
  sprintf(finplotname,"%s_distrlog.png",plotName);
  cout<<finplotname<<endl;
  c_chi2ndist->SaveAs(finplotname);
  
  delete  c_chi2ndist;
  delete f;
}//end plotAlignableChiSquare



// -----------------------------------------------------------------
// private classes
int HIPplots::GetNIterations(TDirectory *f,char *tag,int startingcounter){
	int fin_iter=0,i=startingcounter;
	bool obj_exist=kTRUE;
	while(obj_exist){
		char objname[32];
		sprintf(objname,"%s%d",tag,i);
		if(!f->FindObjectAny(objname))obj_exist=kFALSE;
		fin_iter=i;
		i++;
	}
	cout<<"Max Iterations is "<<fin_iter<<endl;
	
	return fin_iter;
}

int HIPplots::GetSubDet( unsigned int id ){
	
	const int reserved_subdetectorstartbit=25;
	const int reserved_subdetectorfinalbit=27;
	
	unsigned int detID = id;
	
	int shift = 31-reserved_subdetectorfinalbit;
	detID = detID<<(shift);
	shift = reserved_subdetectorstartbit + shift;
	detID = detID>>(shift);
	
	return detID;
	
}

int HIPplots::GetBarrelLayer( unsigned int id ){
	
	const int reserved_layerstartbit=14;
	const int reserved_layermask=0x7;
	
	return int((id>>reserved_layerstartbit) & reserved_layermask);
	
}

void HIPplots::SetMinMax( TH1* h ){
	
	Double_t xmin = 10000;
	Double_t xmax = -10000;

	
	for(int i = 1; i <= h->GetNbinsX(); ++i) {
		if((h->GetBinContent(i) > 0)&&(h->GetBinCenter(i)>xmax) ) xmax = h->GetBinCenter(i);
	}
	for(int i = h->GetNbinsX(); i >= 1; --i) {
		if((h->GetBinContent(i) > 0)&&(h->GetBinCenter(i)<xmin) ) xmin = h->GetBinCenter(i);
	}
	
	h->SetAxisRange((xmin-xmin*0.1), (xmax+xmax*0.1), "X");
	//	std::cout << "xmin: " << xmin << ", xmax: " << xmax << std::endl;

}



void HIPplots::plotHitMap( char *outpath,int subDet,int minHits){
  cout<<"Starting plotHitMap"<<flush;

  TFile *falignable=new TFile(_inFile_HIPalign,"READ");
  cout<<"\tLoaded file"<<flush;
  //take the alignablewise tree and address useful branches
  TTree *talignable=(TTree*)falignable->Get("T2");
  cout<<"\t Loaded tree"<<endl;

  float eta=-999.0,phi=-55.0, xpos=-999.0,ypos=+999.0,zpos=-11111.0;
  int layer=-1,type=-1, nhit=-11111;
  int id=0;
  talignable->SetBranchAddress("Id",&id);
  talignable->SetBranchAddress("Layer",&layer);
  talignable->SetBranchAddress("Type",&type);
  talignable->SetBranchAddress("Nhit",&nhit);
  talignable->SetBranchAddress("Ypos",&ypos);
  talignable->SetBranchAddress("Eta",&eta);
  talignable->SetBranchAddress("Phi",&phi);
  talignable->SetBranchAddress("Ypos",&ypos);
  talignable->SetBranchAddress("Xpos",&xpos);
  talignable->SetBranchAddress("Zpos",&zpos);

  //loop on subdets
  char typetag[16];
  int maxLayers=0;
  if(subDet == TPBid){sprintf(typetag,"TPB");maxLayers=3;}
  else   if(subDet == TPEid){sprintf(typetag,"TPE");maxLayers=2;}
  else   if(subDet == TIBid){sprintf(typetag,"TIB");maxLayers=4;}
  else   if(subDet == TIDid){sprintf(typetag,"TID");maxLayers=3;}
  else   if(subDet == TOBid){sprintf(typetag,"TOB");maxLayers=6;}
  else   if(subDet == TECid){sprintf(typetag,"TEC");maxLayers=9;}
  else   {sprintf(typetag,"UNKNOWN");}
  cout<<"Starting to plot Hit Distributions for "<<typetag<<endl;

  bool printbinning=true;
  char psname[600];
  char ovtag[16];
   sprintf(psname,"%s/Hits_%s_Layers_Skimmed.eps",outpath,typetag);

  char binfilename[64];
  sprintf(binfilename,"./BinningHitMaps_%s.txt",typetag);
  ofstream binfile(binfilename,ios::out); 

  if(printbinning){
    binfile<<"******** Binning for Subdet "<<typetag<<" *********"<<endl<<endl;
  }

  for(int layerindex=1;layerindex<=maxLayers;layerindex++){  //loop on layers

    cout<<"\n\n*** Layer # "<<layerindex<<" ***"<<endl;
  //activate only useful branches
    talignable->SetBranchStatus("*",0);
    talignable->SetBranchStatus("Id",1);
    talignable->SetBranchStatus("Type",1);
    talignable->SetBranchStatus("Layer",1);
    talignable->SetBranchStatus("Nhit",1);
    talignable->SetBranchStatus("Eta",1);
    talignable->SetBranchStatus("Phi",1);
    talignable->SetBranchStatus("Ypos",1);
    talignable->SetBranchStatus("Zpos",1);
    talignable->SetBranchStatus("Xpos",1);

    TCut selA,selECpos,selECneg;
    char selA_str[196],selECneg_str[196],selECpos_str[196];
    char varA_str[64],varB_str[64];
    char commonsense_str[128]="Xpos>-150.0&&Xpos<150.0&&Ypos>-150.0&&Ypos<150.0&&Zpos>-400.0&&Zpos<400.0";
    TCut commonsense=commonsense_str;
   
    sprintf(selECneg_str,"(Type==%d && Layer==%d && Zpos<0.0  && Nhit>=%d && sqrt(pow(Xpos,2)+pow(Ypos,2) )<125.0 )",subDet,layerindex,minHits);
    sprintf(selECpos_str,"(Type==%d && Layer==%d && Zpos>=0.0 && Nhit>=%d && sqrt(pow(Xpos,2)+pow(Ypos,2) )<125.0 )",subDet,layerindex,minHits);
    //      sprintf(selB_str,"(Eta>0.0)&&(Eta<0.2)");
    sprintf(selA_str,"Type==%d && Layer==%d",subDet,layerindex);

    sprintf(varA_str,"Eta>>hvarx");
    sprintf(varB_str,"Phi>>hvary");

    selECneg=selECneg_str;
    selECpos=selECpos_str;
    selA=selA_str;
    cout<<"Cuts defined as "<<selA<<endl;
    ///////////////////////////
    //////////////////////////
    //////////////////////////

    //--------- (2) START bin definition -----
    //cout<<"Selection is "<<sel<<endl;
    //    char sel[96];

    int nzentries= talignable->Draw("Zpos>>hZ(360,-270.0,270.0)",commonsense&&selA,"goff");
    TH1F *hZ=(TH1F*)gDirectory->Get("hZ");
    if(subDet == TOBid)      SetPeakThreshold(8.0);
    else SetPeakThreshold(5.0);
    float Zpeaks[120];
      int bin=0;
      for(bin=0;bin<120;bin++){
	Zpeaks[bin]=-444.0;
      }
      const int nZpeaks=FindPeaks(hZ,Zpeaks,99);
      const int nZbinlims=nZpeaks+1;
      float Zwidth=(Zpeaks[nZpeaks-1]-Zpeaks[0])/ (nZpeaks-1) ;
      float Zmin=Zpeaks[0]- Zwidth/2.0 ;
      float Zmax=Zpeaks[nZpeaks-1] + Zwidth/2.0;//((Zpeaks[nZbinlims-1]-Zpeaks[nZbinlims-2])/2.0) ;
      cout<<"--> Zmin= "<<Zmin<<" - Zmax= "<<Zmax<<" Zwidth= "<<Zwidth<<" ; found "<<nZpeaks<<" Zpeaks"<<endl;
      cout<<"Zpeaks[0] is "<<Zpeaks[0]<<endl;
      
 
      float Phipeaks[120];
      if((subDet==TIBid||subDet==TOBid)&&layerindex<=2) sprintf(selA_str,"%s&&Zpos>%f&&Zpos<%f",selA_str,Zpeaks[0]-2.0,Zpeaks[0]+2.0);//DS modules 
      else sprintf(selA_str,"%s&&Zpos>%f&&Zpos<%f",selA_str,Zpeaks[0]-2.0,Zpeaks[0]+2.0);
      int  nphientries=talignable->Draw("Phi>>hPhi",selA_str,"goff");
      cout<<"N phi entries "<<nphientries<<" from sel "<<selA_str<< endl;
      TH1F *hPhi=(TH1F*)gDirectory->Get("hPhi");
      //      nPhibins=FindPeaks(hPhi,Phipeaks,99);
      if(subDet==TPBid&&layerindex==1)nphientries=nphientries-1; 
      const int  nPhibins=nphientries;
      cout<<"+ It would have found "<<nPhibins<<" phi bins"<<endl;
      
      //fill Z array with binning. special case for DS layers of TIB with non equidistant z bins
      float phibin[nPhibins+1];
      float zbin[nZbinlims];
      float Phiwidth=6.28/nPhibins;
      
      if((subDet==TIBid||subDet==TOBid)&&layerindex<=2){//DS modules in TIB and TOB
	
	///-************
	cout<<"Gonna loop over "<<nZpeaks<<" peaks / "<<nZbinlims<<" bin limits"<<endl;
	for(bin=0;bin<nZbinlims-1;bin++){
	  float zup=0.0;
	  float zdown=0.0;

	  if(bin==0){               //don't go underflow!
	    zup=(Zpeaks[bin+1]-Zpeaks[bin])/2.0;
	    zdown=zup;
	  }
	  else if(bin==nZbinlims-2){//don't go overflow!
	    cout<<"Don't go overflow !"<<endl;
	    zdown=(Zpeaks[bin]-Zpeaks[bin-1])/2.0;
	    if(layerindex==1) zup=(Zpeaks[bin-1]-Zpeaks[bin-2])/2.0;
	    else  zup=zdown;
	  }
	  else{
	    zup=(Zpeaks[bin+1]-Zpeaks[bin])/2.0;
	    zdown=(Zpeaks[bin]-Zpeaks[bin-1])/2.0;
	  }
	  zbin[bin]  =  Zpeaks[bin]-zdown;
	  zbin[bin+1]=  Zpeaks[bin]+zup;

	}//end loop on z bin
	///-************


	for(int bin=0;bin<=nPhibins;++bin){
	  if(bin==0)phibin[bin]=-3.14+bin*(Phiwidth)+Phiwidth/4;
	  else if(bin==nPhibins-1)phibin[bin]=-3.14+bin*(Phiwidth)-Phiwidth/4;
	  // else phibin[bin]=-3.14+bin*(Phiwidth);
	  else phibin[bin]=phibin[bin-1]+ Phiwidth;
	}//end loop on phi bin

       }//end if DS layers of TIB/TOB
       else{
		for(int bin=0;bin<nZbinlims;++bin){
		  zbin[bin]=Zmin+bin*(Zwidth);
		}//end loop on z bin

		for(int bin=0;bin<=nPhibins;++bin){
		  phibin[bin]=-3.14+bin*(Phiwidth);
		}//end loop on phi bin
	 }

      
      float Phimin=Phipeaks[0]- ((Phipeaks[1]-Phipeaks[0])/2.0) ;
      float Phimax=Phipeaks[nPhibins-1] + ((Phipeaks[nPhibins-1]-Phipeaks[nPhibins-2])/2.0) ;
      
      cout<<"N Z bin LIMITs = "<<nZbinlims<<" N Phi bin LIMITS = "<<nPhibins<<endl;
      //--------- END bin definition -----
      


    char histoname[64];
    sprintf(histoname,"%s_Layer%d",typetag,layerindex);
    TH2F *hetaphi=new TH2F(histoname,histoname,nZpeaks,zbin,nPhibins,phibin);


    cout<<"Starting to loop on entries"<<flush;
    int nmods=0;
    int nlowentrycells=0;

    for(int j=0;j<talignable->GetEntries();j++){ //loop on entries (-> alignables)
      if(j%1000==0)cout<<"."<<flush;
      talignable->GetEntry(j);
      if(type==subDet&&layer==layerindex){
	  if(nhit>=minHits){
	    //find the right eta bin
	    hetaphi->Fill(zpos,phi,nhit);
	    nmods++;
	  }
	  else{
	    hetaphi->Fill(zpos,phi,-99);
	    nlowentrycells++;
	  }//end else if nhits < minhits

      }//end if the type and layer are the desired ones
    }//end loop on entries(->alignables)

    //Let's start with the funny things: fancy graphics!
    hetaphi->SetXTitle("Z [cm]");
    hetaphi->SetYTitle("#phi (rad)");

    int Nxbins=hetaphi->GetXaxis()->GetNbins();
    int Nybins=hetaphi->GetYaxis()->GetNbins();
    cout<<"On x-axis there are "<<Nxbins<<" bins  "<<endl;
    cout<<"On y-axis there are "<<Nybins<<" bins  "<<endl;


    bool smooth_etaphi=false;
    if(smooth_etaphi){
      for(int i=1;i<=Nxbins;i++){
	for(int j=1;j<=Nybins;j++){
	  float bincont=hetaphi->GetBinContent(i,j);	
	  if(bincont==0){//average with the surrounding bins
	    float binup1=hetaphi->GetBinContent(i,j+1);
	    float bindown1=hetaphi->GetBinContent(i,j-1);
	    float binlx1=hetaphi->GetBinContent(i-1,j);
	    float binrx1=hetaphi->GetBinContent(i+1,j);
	    if(i==1)binlx1=binrx1;//at the edges you don't have lx or rx bins; set a def value
	    if(i==Nxbins)binrx1=binlx1;
	    if(j==1)bindown1=binup1;
	    if(j==Nybins)binup1=bindown1;
	    int adiacentbins=0;
	    if(binup1>0.0)adiacentbins++;
	    if(bindown1>0.0)adiacentbins++;
	    if(binlx1>0.0)adiacentbins++;
	    if(binrx1>0.0)adiacentbins++;
	    if(adiacentbins>=2){
	      float avg=(binup1+bindown1+binlx1+binrx1)/adiacentbins;
	      hetaphi->SetBinContent(i,j,avg);
	    }
	  }
	}
      }
    }//end if smooth_etaphi

    
    //for debugging purposes
    TGraph *gretaphi;
    bool plotAlignablePos=false;
    if(plotAlignablePos){
    const int ngrpoints=nmods;
    float etagr[ngrpoints],phigr[ngrpoints];
    nmods=0;
    for(int j=0;j<talignable->GetEntries();j++){ //loop on entries (-> alignables)
      if(j%1000==0)cout<<"."<<flush;
      talignable->GetEntry(j);
      if(type==subDet&&layer==layerindex){
	etagr[nmods]=zpos;
	phigr[nmods]=phi;
	nmods++;
      }
    }

    gretaphi=new TGraph(ngrpoints,etagr,phigr);
    gretaphi->SetMarkerStyle(20);
    gretaphi->SetMarkerColor(1);
    gretaphi->SetMarkerSize(0.75);
    }    //////



    ///////////////////
    // cout<<"Layer #"<<i<<endl;
    float Zcellgr[512];
    float Phicellgr[512];
    int nemptycells=0;
    for(int zcells=1;zcells<=hetaphi->GetNbinsX();zcells++){
      for(int phicells=1;phicells<=hetaphi->GetNbinsY();phicells++){
	if( hetaphi->GetBinContent( zcells,phicells)==-99){
	  hetaphi->SetBinContent( zcells,phicells,0);
	  Zcellgr[nemptycells]= float(hetaphi->GetXaxis()->GetBinCenter(zcells));
	  Phicellgr[nemptycells]= float(hetaphi->GetYaxis()->GetBinCenter(phicells));
	  nemptycells++;	    
	}
	//  if(a==2)cout<<"Finished Z value "<<hetaphi->GetXaxis()->GetBinCenter(zcells)<<" the emptycells are "<<nemptycells<<endl;
	}//end loop on Phi bins
    }//end loop on Z bins
    TGraph *gr_empty=new TGraph( nlowentrycells,Zcellgr,Phicellgr);
    sprintf(histoname,"gr_emptycells_subdet%d_layer%d",subDet,layerindex);
    //      cout<<"Graph name: "<<histoname<<" with "<<gr_empty->GetN()<<endl;
    gr_empty->SetName(histoname);
    gr_empty->SetMarkerStyle(5);
    /////////


    cout<<" Done! Used "<<nmods<<" alignables. Starting to plot !"<<endl;

  //plot them and save the canvas appending it to a ps file
    gStyle->SetPalette(1,0);
    TCanvas *c_barrels=new TCanvas("canvas_hits_barrels","CANVAS_HITS_BARRELS",1600,1600);
    TCanvas *c_endcaps=new TCanvas("canvas_hits_endcaps","CANVAS_HITS_ENDCAPS",3200,1600);
    TPad *pleft=new TPad("left_panel","Left Panel",0.0,0.0,0.49,0.99);
    TPad *pcent=new TPad("central_up_panel","Central Panel",0.01,0.00,0.99,0.99);
    TPad *pright=new TPad("right_panel","Right Panel",0.51,0.0,0.99,0.99);


    if(subDet==1 ||subDet==3 ||subDet==5  ){
      c_barrels->cd();
      pcent->Draw();
      pcent->cd();
      gPad->SetRightMargin(0.15);
      
      //    hbarrel->SetMarkerSize(2.0);  
      // hbarrel->SetMarkerSize(21);  
      hetaphi->SetStats(0);
      if(subDet==TPBid||subDet==TIBid||subDet==TOBid)hetaphi->Draw("COLZtext");//COLZtext
      if(plotAlignablePos) gretaphi->Draw("P");
      gr_empty->Draw("P");
      
      if(printbinning){
	binfile<<"--> Layer #"<<layerindex<<endl;
      binfile<<"Eta Binning: "<<flush;
      for(int h=1;h<=hetaphi->GetNbinsX();h++){
	binfile<<hetaphi->GetXaxis()->GetBinLowEdge(h)<<"\t";
	if(h==hetaphi->GetNbinsX())	binfile<<hetaphi->GetXaxis()->GetBinLowEdge(h)+hetaphi->GetXaxis()->GetBinWidth(h);
      }
      binfile<<endl;
      binfile<<"Phi Binning: "<<flush;
      for(int h=1;h<=hetaphi->GetNbinsY();h++){
	binfile<<hetaphi->GetYaxis()->GetBinLowEdge(h)<<"\t";
	if(h==hetaphi->GetNbinsX())	binfile<<hetaphi->GetYaxis()->GetBinLowEdge(h)+hetaphi->GetYaxis()->GetBinWidth(h);
      }
      binfile<<endl;
    }

    }
    else{
      c_endcaps->cd();
      pleft->Draw();
      pright->Draw();

    
    pleft->cd();
    gPad->SetRightMargin(0.15);
    char selEC_str[192], varEC_str[128];
    float radlimit=0.0;
    if(subDet == TPBid){radlimit=45.0;}
    else   if(subDet == TPEid){radlimit=45.0;}
    else   if(subDet == TIBid){radlimit=70.0;}
    else   if(subDet == TIDid){radlimit=70.0;}
    else   if(subDet == TOBid){radlimit=130.0;}
    else   if(subDet == TECid){radlimit=130.0;}
    else   {radlimit=0.0;}

    sprintf(varEC_str,"Ypos:Xpos>>hxy_negz(30,%f,%f)",-radlimit,radlimit);
    sprintf(selEC_str,"Nhit*(%s&&%s)",selECneg_str,commonsense_str);
    cout<<selEC_str<<endl;
    int selentriesECneg=talignable->Draw(varEC_str,selEC_str,"goff"); //total number of alignables for thys type and layer
    if(selentriesECneg>0){
    TH2F *hxy_negz=(TH2F*)gDirectory->Get("hxy_negz");
    hxy_negz->GetXaxis()->SetRangeUser(-radlimit,radlimit);
    hxy_negz->GetYaxis()->SetRangeUser(-radlimit,radlimit);
    char histoname_negz[32];
    sprintf(histoname_negz,"%s (-Z)",histoname);
    hxy_negz->SetStats(0);
    hxy_negz->SetTitle( histoname_negz);
    hxy_negz->SetXTitle("X (cm)");
    hxy_negz->SetYTitle("Y (cm)");
    hxy_negz->Draw("COLZ");
    }
    else{
      cout<<"WARNING !!!! No hits on this layer ! not plotting (-Z) !"<<endl;
    }



    cout<<"PAD 3"<<endl;
    pright->cd();
    gPad->SetRightMargin(0.15);
    sprintf(selEC_str,"Nhit*(%s&&%s)",selECpos_str,commonsense_str);
    //    selEC=selEC_str&&commonsense;
    cout<<"(2)"<<selEC_str<<endl;
    sprintf(varEC_str,"Ypos:Xpos>>hxy_posz(30,%f,%f)",-radlimit,radlimit);
    int selentriesECpos=talignable->Draw(varEC_str,selEC_str,"goff"); //total number of alignables for thys type and layer
    if(selentriesECpos>0){
      TH2F *hxy_posz=(TH2F*)gDirectory->Get("hxy_posz");
      char histoname_posz[32];
      hxy_posz->GetXaxis()->SetRangeUser(-radlimit,radlimit);
      hxy_posz->GetYaxis()->SetRangeUser(-radlimit,radlimit);
      sprintf(histoname_posz,"%s (+Z)",histoname);
      hxy_posz->SetStats(0);
      hxy_posz->SetTitle( histoname_posz);
      hxy_posz->SetXTitle("X (cm)");
      hxy_posz->SetYTitle("Y (cm)");
      hxy_posz->Draw("COLZ");
    }
    else{
      cout<<"WARNING !!!! No hits on this layer ! not plotting (+Z) !"<<endl;
    }


    }

    //save in a ps file
    cout << "******* " << typetag << endl;
    char psnamefinal[600]; 

    if(layerindex==1) sprintf(psnamefinal,"%s(",psname);
    else if(layerindex==maxLayers) sprintf(psnamefinal,"%s)",psname);
    else  sprintf(psnamefinal,"%s",psname);

    cout<<"Saving in "<<psnamefinal<<endl;
    if(subDet==1 ||subDet==3 ||subDet==5  )c_barrels->SaveAs(psnamefinal);
    else c_endcaps->SaveAs(psnamefinal);
    
    if(subDet==1 ||subDet==3 ||subDet==5  )delete c_barrels;
    else delete c_endcaps;


      //      ctmp->cd();
      //  hbarrel->Draw("scat");

  }//end loop on layers
      cout<<"Finished "<<maxLayers<<" of the "<<typetag<<endl;  

  delete talignable;
  delete falignable;
}//end PlotHitDistribution


void HIPplots::dumpAlignedModules(int nhits){

  ///

}//end  dumpAlignedModules



int HIPplots::FindPeaks(TH1F *h1,float *peaklist,const int maxNpeaks,int startbin,int endbin){

  int Npeaks=0;
  if(startbin<0)startbin=1;
  if(endbin<0)endbin=h1->GetNbinsX();
  int bin=startbin;
  int prevbin=startbin;
  bool rising=true;
  while(bin<=endbin){

    float bincont=h1->GetBinContent(bin);
    float prevbincont=h1->GetBinContent(prevbin);

    if(prevbincont>peakthreshold||bincont>1.0){
      if(bincont>=prevbincont){//we are rising, keep going until we don't go down
	rising=true;
      }
      else{//ehi! we are decreasing. But check if we were decreasing already at the step before
	if(!rising){//we were already decreasing, this is not a maximum
	  rising=true;
	}
	else{//we found a maximum
	  rising=false;
	  peaklist[Npeaks]=h1->GetBinCenter(prevbin);
	  Npeaks++;
	}
      }
    }//end if bincont>1.0
    if(Npeaks>=maxNpeaks){//max number reached
      bin=endbin;
    }
    bin++;
    prevbin=bin-1;
  }//end while loop on bins


  return Npeaks;
}//End FindPeaks


void HIPplots::SetPeakThreshold(float newpeakthreshold){
  peakthreshold=newpeakthreshold;
}

bool HIPplots::CheckFileExistence(char *filename){
  bool flag = false;
  fstream fin;
  fin.open(filename,ios::in);
  if( fin.is_open() )   flag=true;
  fin.close();
  return flag;
}

void HIPplots::CheckFiles(int &ierr){
  
  if (! CheckFileExistence(_inFile_uservars)){
    cout<<"Missing file "<<_inFile_uservars<<endl;
    ierr++;
  }

  if(! CheckFileExistence(_inFile_HIPalign)){
    cout<<"Missing file "<<_inFile_HIPalign<<endl;
    ierr++;
  }

  if(! CheckFileExistence( _inFile_alipos)){
    cout<<"Missing file "<< _inFile_alipos<<endl;
    ierr++;
  }
  
  if( CheckFileExistence( _outFile)){
    cout<<"Output file already existing !"<<endl;
    ierr=-1*(ierr+1);
  }

}


bool HIPplots::CheckHistoRising(TH1D *h){
  
  bool rise1=false,rise2=false,rise3=false;
  int totbins=h->GetNbinsX();
  //  for(int i=1;i<=totbins;i++){
  int i=1;


  for(i=1;i<=totbins-3;i++){
    double cont1 = h->GetBinContent(i);
    if( h->GetBinContent(i+1)>cont1)rise1=true;
    if(rise1){
      if( h->GetBinContent(i+2)>h->GetBinContent(i+1))rise2=true;
      if(rise2){
	if( h->GetBinContent(i+3)>h->GetBinContent(i+2)){
	  rise3=true;
	  break;
	}
      }
    }

  }//emnd loop on bins

  return rise3;

}//end CheckHistoRising

