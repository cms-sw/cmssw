#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <cmath>
#include <TString.h>

void CosmicRateTool_CosmicRates(const char* fileName, unsigned int runLow=0, unsigned int runUp=0)
{
   TString InputFile= Form("../test/%s",fileName); 
   TFile *file = new TFile(InputFile);
   TTree *tree;
   tree = (TTree*)file->Get("demo/Run");

   FILE * pFile;
   pFile = fopen ("tracksInfo.txt","w");
	
   double  run_time;
   unsigned int runnum;
   int     number_of_events;
   int     number_of_tracks;
   int     number_of_tracks_PIX;
   int     number_of_tracks_FPIX;
   int     number_of_tracks_BPIX;
   int     number_of_tracks_TID;
   int     number_of_tracks_TIDM;
   int     number_of_tracks_TIDP;
   int     number_of_tracks_TIB;
   int     number_of_tracks_TEC;
   int     number_of_tracks_TECP;
   int     number_of_tracks_TECM;
   int     number_of_tracks_TOB;
   
   tree->SetBranchAddress("run_time", &run_time);
   tree->SetBranchAddress("runnum", &runnum);
   tree->SetBranchAddress("number_of_events", &number_of_events);
   tree->SetBranchAddress("number_of_tracks", &number_of_tracks);
   tree->SetBranchAddress("number_of_tracks_PIX", &number_of_tracks_PIX);
   tree->SetBranchAddress("number_of_tracks_FPIX", &number_of_tracks_FPIX);
   tree->SetBranchAddress("number_of_tracks_BPIX", &number_of_tracks_BPIX);
   tree->SetBranchAddress("number_of_tracks_TID", &number_of_tracks_TID);
   tree->SetBranchAddress("number_of_tracks_TIDM", &number_of_tracks_TIDM);
   tree->SetBranchAddress("number_of_tracks_TIDP", &number_of_tracks_TIDP);
   tree->SetBranchAddress("number_of_tracks_TIB", &number_of_tracks_TIB);
   tree->SetBranchAddress("number_of_tracks_TEC", &number_of_tracks_TEC);
   tree->SetBranchAddress("number_of_tracks_TECP", &number_of_tracks_TECP);
   tree->SetBranchAddress("number_of_tracks_TECM", &number_of_tracks_TECM);
   tree->SetBranchAddress("number_of_tracks_TOB", &number_of_tracks_TOB);
   
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//		Various Rates Declerations				
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   Long64_t n = tree->GetEntriesFast();

   vector<double>  event_rate 		;
   vector<double>  event_rate_err	;
   vector<double>  track_rate 		;
   vector<double>  track_rate_err	;
   vector<double>  runNumber  		;
   vector<double>  runNumber_err	;
   vector<double>  track_rate_PIX 	;
   vector<double>  track_rate_PIX_err 	;
   vector<double>  track_rate_FPIX	;
   vector<double>  track_rate_FPIX_err	;
   vector<double>  track_rate_BPIX 	;
   vector<double>  track_rate_BPIX_err	;
   vector<double>  track_rate_TOB 	;
   vector<double>  track_rate_TOB_err 	;
   vector<double>  track_rate_TIB 	;
   vector<double>  track_rate_TIB_err 	;
   vector<double>  track_rate_TID 	;
   vector<double>  track_rate_TID_err 	;
   vector<double>  track_rate_TEC 	;
   vector<double>  track_rate_TEC_err 	;
   vector<double>  track_rate_TECP 	;
   vector<double>  track_rate_TECP_err	;
   vector<double>  track_rate_TECM 	;
   vector<double>  track_rate_TECM_err	;
   vector<double>  tracks		;
   vector<double>  tracks_err		;
   vector<double>  tracks_bpix		;
   vector<double>  tracks_fpix		;
   vector<double>  tracks_pix		;
   vector<double>  tracks_tec		;
   vector<double>  weight		;

   
   string Bar_Xtitle[8] = {"Event","Track","FPIX","BPIX","TIB","TID","TOB","TEC"};
   double Bar_Ytitle[8] = {0};
   
   
   int j=0;
   double total_tracks = 0;
   double bpix_tracks = 0;
   double fpix_tracks = 0;
   double pix_tracks = 0;
   double tracks_TECoff = 0;
   
   cout<<n<<endl;

   fprintf(pFile,"##################################################\n");
   fprintf(pFile,"         Track rate for each run number           \n");
   fprintf(pFile,"##################################################\n");
   
   for (Long64_t jentry=0; jentry<n;jentry++) 
   {
     tree->GetEntry(jentry);
     if (run_time == 0 ) continue;

     if (runLow != 0 && runUp != 0 ) 
     {
        if (runnum < runLow) continue;
        if (runnum > runUp) break;
     }  

      event_rate	 .push_back( number_of_events/run_time );
      runNumber 	 .push_back( runnum );
      track_rate	 .push_back( number_of_tracks/run_time );
      track_rate_PIX	 .push_back( number_of_tracks_PIX/run_time );
      track_rate_FPIX	 .push_back( number_of_tracks_FPIX/run_time );
      track_rate_BPIX	 .push_back( number_of_tracks_BPIX/run_time );
      track_rate_TOB	 .push_back( number_of_tracks_TOB/run_time );
      track_rate_TIB	 .push_back( number_of_tracks_TIB/run_time );
      track_rate_TID	 .push_back( number_of_tracks_TID/run_time );
      track_rate_TEC	 .push_back( number_of_tracks_TEC/run_time );
      track_rate_TECP	 .push_back( number_of_tracks_TECP/run_time );
      track_rate_TECM	 .push_back( number_of_tracks_TECM/run_time );
      tracks	 	 .push_back( number_of_tracks );
      tracks_bpix	 .push_back( number_of_tracks_BPIX );
      tracks_fpix	 .push_back( number_of_tracks_FPIX );
      tracks_pix	 .push_back( number_of_tracks_PIX );
      tracks_tec	 .push_back( number_of_tracks_TECM );
      total_tracks	+= tracks[j];
      bpix_tracks 	+= tracks_bpix[j];
      fpix_tracks 	+= tracks_fpix[j];
      pix_tracks 	+= tracks_pix[j];


       fprintf(pFile,"runnum :%-7.0lf, # of tracks :%-10.0lf, track rates :%-10.2lf\n",runNumber.at(j),tracks.at(j),track_rate.at(j));
       track_rate_err	 	.push_back( sqrt(float(number_of_tracks))/run_time );
       event_rate_err	 	.push_back( sqrt(float(number_of_events))/run_time );
       track_rate_PIX_err 	.push_back( sqrt(float(number_of_tracks_PIX))/run_time );
       track_rate_FPIX_err 	.push_back( sqrt(float(number_of_tracks_FPIX))/run_time );
       track_rate_BPIX_err 	.push_back( sqrt(float(number_of_tracks_BPIX))/run_time );
       track_rate_TOB_err 	.push_back( sqrt(float(number_of_tracks_TOB))/run_time );
       track_rate_TIB_err 	.push_back( sqrt(float(number_of_tracks_TIB))/run_time );
       track_rate_TID_err 	.push_back( sqrt(float(number_of_tracks_TID))/run_time );
       track_rate_TEC_err 	.push_back( sqrt(float(number_of_tracks_TEC))/run_time );
       track_rate_TECP_err 	.push_back( sqrt(float(number_of_tracks_TECP))/run_time );
       track_rate_TECM_err 	.push_back( sqrt(float(number_of_tracks_TECM))/run_time );

      runNumber_err.push_back(0);   
	if (number_of_tracks_TECM == 0){
	tracks_TECoff += tracks.at(j);}

      j++;
	

     
     }

       fprintf(pFile,"\n\n");
       fprintf(pFile,"##################################################\n");
       fprintf(pFile,"    Some information on total number of tracks    \n");
       fprintf(pFile,"##################################################\n");
       fprintf(pFile,"Total # of tracks   : %-10.0lf\n",total_tracks);
       fprintf(pFile,"# of tracks in BPIX : %-10.0lf\n",bpix_tracks);
       fprintf(pFile,"# of tracks in FPIX : %-10.0lf\n",fpix_tracks);
       fprintf(pFile,"# of tracks in PIX  : %-10.0lf\n",pix_tracks);
       fprintf(pFile,"\n\n");

   fclose (pFile);
     
//+++++++++++++++++++++++++++++       Make Directories     +++++++++++++++++++++++++++++++++++++

	gSystem->Exec("mkdir -p Rate_Plots");

//----------------------------------------------------------------------------------------------


   TCanvas c("c","c",800,600);    // Declare canvas

   TVectorD  event_rate_VecD 		;
   TVectorD  event_rate_err_VecD	;
   TVectorD  track_rate_VecD 		;
   TVectorD  track_rate_err_VecD	;
   TVectorD  runNumber_VecD 		;
   TVectorD  runNumber_err_VecD		;
   TVectorD  track_rate_PIX_VecD 	;
   TVectorD  track_rate_PIX_err_VecD 	;
   TVectorD  track_rate_FPIX_VecD	;
   TVectorD  track_rate_FPIX_err_VecD	;
   TVectorD  track_rate_BPIX_VecD 	;
   TVectorD  track_rate_BPIX_err_VecD	;
   TVectorD  track_rate_TOB_VecD 	;
   TVectorD  track_rate_TOB_err_VecD 	;
   TVectorD  track_rate_TIB_VecD 	;
   TVectorD  track_rate_TIB_err_VecD 	;
   TVectorD  track_rate_TID_VecD 	;
   TVectorD  track_rate_TID_err_VecD 	;
   TVectorD  track_rate_TEC_VecD 	;
   TVectorD  track_rate_TEC_err_VecD 	;
   TVectorD  track_rate_TECP_VecD 	;
   TVectorD  track_rate_TECP_err_VecD	;
   TVectorD  track_rate_TECM_VecD 	;
   TVectorD  track_rate_TECM_err_VecD	;


   runNumber_VecD.Use(runNumber.size(),&(runNumber[0]));
   runNumber_err_VecD.Use(runNumber_err.size(),&(runNumber_err[0]));
   event_rate_VecD.Use(event_rate.size(),&(event_rate[0]));
   event_rate_err_VecD.Use(event_rate_err.size(),&(event_rate_err[0]));

   track_rate_VecD.Use(track_rate.size(),&(track_rate[0]));
   track_rate_err_VecD.Use(track_rate_err.size(),&(track_rate_err[0]));

   track_rate_PIX_VecD.Use(track_rate_PIX.size(),&(track_rate_PIX[0]));
   track_rate_PIX_err_VecD.Use(track_rate_PIX_err.size(),&(track_rate_PIX_err[0]));
   track_rate_FPIX_VecD.Use(track_rate_FPIX.size(),&(track_rate_FPIX[0]));
   track_rate_FPIX_err_VecD.Use(track_rate_FPIX_err.size(),&(track_rate_FPIX_err[0]));
   track_rate_BPIX_VecD.Use(track_rate_BPIX.size(),&(track_rate_BPIX[0]));
   track_rate_BPIX_err_VecD.Use(track_rate_BPIX_err.size(),&(track_rate_BPIX_err[0]));
   track_rate_TOB_VecD.Use(track_rate_TOB.size(),&(track_rate_TOB[0]));
   track_rate_TOB_err_VecD.Use(track_rate_TOB_err.size(),&(track_rate_TOB_err[0]));
   track_rate_TIB_VecD.Use(track_rate_TIB.size(),&(track_rate_TIB[0]));
   track_rate_TIB_err_VecD.Use(track_rate_TIB_err.size(),&(track_rate_TIB_err[0]));
   track_rate_TID_VecD.Use(track_rate_TID.size(),&(track_rate_TID[0]));
   track_rate_TID_err_VecD.Use(track_rate_TID_err.size(),&(track_rate_TID_err[0]));
   track_rate_TEC_VecD.Use(track_rate_TEC.size(),&(track_rate_TEC[0]));
   track_rate_TEC_err_VecD.Use(track_rate_TEC_err.size(),&(track_rate_TEC_err[0]));
   track_rate_TECP_VecD.Use(track_rate_TECP.size(),&(track_rate_TECP[0]));
   track_rate_TECP_err_VecD.Use(track_rate_TECP_err.size(),&(track_rate_TECP_err[0]));
   track_rate_TECM_VecD.Use(track_rate_TECM.size(),&(track_rate_TECM[0]));
   track_rate_TECM_err_VecD.Use(track_rate_TECM_err.size(),&(track_rate_TECM_err[0]));


//+++++++++++++++++++++++++++++  Overall event event rate  +++++++++++++++++++++++++++++++++++++ 

	TGraphErrors gr_event_rate(runNumber_VecD,event_rate_VecD,runNumber_err_VecD,event_rate_err_VecD);
	gr_event_rate.GetXaxis()->SetTitle("Run Number");
	gr_event_rate.GetXaxis()->SetLabelSize(0.03);
	gr_event_rate.GetXaxis()->SetNoExponent();
	gr_event_rate.GetYaxis()->SetTitle("Event Rate (in Hz)");
	gr_event_rate.SetMarkerStyle(20);
	gr_event_rate.SetMarkerSize(1.2);
	gr_event_rate.SetMarkerColor(kBlue);
	gr_event_rate.SetTitle("Event Rate");
	gr_event_rate.GetYaxis()->SetRangeUser(0,7);
	gr_event_rate.Draw("AP");
	c.SetGrid();
    	c.SaveAs("event_rate.png");
	c.Clear();
	gSystem->Exec("mv event_rate.png Rate_Plots");

//-----------------------------------------------------------------------------------------------


//++++++++++++++++++++++++++++++  Overall track rate  +++++++++++++++++++++++++++++++++++++++++++ 

	TGraphErrors gr_track_rate(runNumber_VecD,track_rate_VecD,runNumber_err_VecD,track_rate_err_VecD);
	gr_track_rate.GetXaxis()->SetTitle("Run Number");
	gr_track_rate.GetXaxis()->SetLabelSize(0.03);
	gr_track_rate.GetXaxis()->SetNoExponent();
	gr_track_rate.GetYaxis()->SetTitle("Track Rate (in Hz)");
	gr_track_rate.SetMarkerStyle(20);
	gr_track_rate.SetMarkerSize(1.2);
	gr_track_rate.SetMarkerColor(kBlue);
	gr_track_rate.SetTitle("Track Rate");
	gr_track_rate.GetYaxis()->SetRangeUser(0,5);
	gr_track_rate.Draw("AP");
	c.SetGrid();
        c.SaveAs("track_rate.png");
	c.Clear();
	gSystem->Exec("mv track_rate.png Rate_Plots");

//-----------------------------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++  Total Pixel track rate +++++++++++++++++++++++++++++++++++++++  

	TGraphErrors gr_track_rate_PIX(runNumber_VecD,track_rate_PIX_VecD,runNumber_err_VecD,track_rate_PIX_err_VecD);
	gr_track_rate_PIX.GetXaxis()->SetTitle("Run Number");
	gr_track_rate_PIX.GetXaxis()->SetLabelSize(0.03);
	gr_track_rate_PIX.GetXaxis()->SetNoExponent();
	gr_track_rate_PIX.GetYaxis()->SetTitle("Track Rate (in Hz)");
	gr_track_rate_PIX.SetMarkerStyle(20);
	gr_track_rate_PIX.SetMarkerSize(1.2);
	gr_track_rate_PIX.SetMarkerColor(kBlue);
	gr_track_rate_PIX.SetTitle("Pixel Track Rate");
	gr_track_rate_PIX.Draw("AP");
	c.SetGrid();
        c.SaveAs("pixel_track_rate.png");
	c.Clear();
	gSystem->Exec("mv pixel_track_rate.png Rate_Plots");

//-----------------------------------------------------------------------------------------------

//++++++++++++++++++++++++++++++++  FPIX track rate  ++++++++++++++++++++++++++++++++++++++++++++ 

	TGraphErrors gr_track_rate_FPIX(runNumber_VecD,track_rate_FPIX_VecD,runNumber_err_VecD,track_rate_FPIX_err_VecD);
	gr_track_rate_FPIX.GetXaxis()->SetTitle("Run Number");
	gr_track_rate_FPIX.GetXaxis()->SetLabelSize(0.03);
	gr_track_rate_FPIX.GetXaxis()->SetNoExponent();
	gr_track_rate_FPIX.GetYaxis()->SetTitle("Track Rate (in Hz)");
	gr_track_rate_FPIX.SetMarkerStyle(20);
	gr_track_rate_FPIX.SetMarkerSize(1.2);
	gr_track_rate_FPIX.SetMarkerColor(kBlue);
	gr_track_rate_FPIX.SetTitle("FPIX Track Rate");
	gr_track_rate_FPIX.Draw("AP");
	c.SetGrid();
        c.SaveAs("fpix_track_rate.png");
	c.Clear();
	gSystem->Exec("mv fpix_track_rate.png Rate_Plots");
//-----------------------------------------------------------------------------------------------


//++++++++++++++++++++++++++++++++  BPIX track rate  ++++++++++++++++++++++++++++++++++++++++++++ 

	TGraphErrors gr_track_rate_BPIX(runNumber_VecD,track_rate_BPIX_VecD,runNumber_err_VecD,track_rate_BPIX_err_VecD);
	gr_track_rate_BPIX.GetXaxis()->SetTitle("Run Number");
	gr_track_rate_BPIX.GetXaxis()->SetLabelSize(0.03);
	gr_track_rate_BPIX.GetXaxis()->SetNoExponent();
	gr_track_rate_BPIX.GetYaxis()->SetTitle("Track Rate (in Hz)");
	gr_track_rate_BPIX.SetMarkerStyle(20);
	gr_track_rate_BPIX.SetMarkerSize(1.2);
	gr_track_rate_BPIX.SetMarkerColor(kBlue);
	gr_track_rate_BPIX.SetTitle("BPIX Track Rate");
	gr_track_rate_BPIX.Draw("AP");
	c.SetGrid();
        c.SaveAs("bpix_track_rate.png");
	c.Clear();
	gSystem->Exec("mv bpix_track_rate.png Rate_Plots");

//-----------------------------------------------------------------------------------------------


//++++++++++++++++++++++++++++++++  TOB track rate  ++++++++++++++++++++++++++++++++++++++++++++ 

	TGraphErrors gr_track_rate_TOB(runNumber_VecD,track_rate_TOB_VecD,runNumber_err_VecD,track_rate_TOB_err_VecD);
	gr_track_rate_TOB.GetXaxis()->SetTitle("Run Number");
	gr_track_rate_TOB.GetXaxis()->SetLabelSize(0.03);
	gr_track_rate_TOB.GetXaxis()->SetNoExponent();
	gr_track_rate_TOB.GetYaxis()->SetTitle("Track Rate (in Hz)");
	gr_track_rate_TOB.SetMarkerStyle(20);
	gr_track_rate_TOB.SetMarkerSize(1.2);
	gr_track_rate_TOB.SetMarkerColor(kBlue);
	gr_track_rate_TOB.SetTitle("TOB Track Rate");
	gr_track_rate_TOB.Draw("AP");
	c.SetGrid();
        c.SaveAs("tob_track_rate.png");
	c.Clear();
	gSystem->Exec("mv tob_track_rate.png Rate_Plots");

//-----------------------------------------------------------------------------------------------


//++++++++++++++++++++++++++++++++  TIB track rate  ++++++++++++++++++++++++++++++++++++++++++++ 

	TGraphErrors gr_track_rate_TIB(runNumber_VecD,track_rate_TIB_VecD,runNumber_err_VecD,track_rate_TIB_err_VecD);
	gr_track_rate_TIB.GetXaxis()->SetTitle("Run Number");
	gr_track_rate_TIB.GetXaxis()->SetLabelSize(0.03);
	gr_track_rate_TIB.GetXaxis()->SetNoExponent();
	gr_track_rate_TIB.GetYaxis()->SetTitle("Track Rate (in Hz)");
	gr_track_rate_TIB.SetMarkerStyle(20);
	gr_track_rate_TIB.SetMarkerSize(1.2);
	gr_track_rate_TIB.SetMarkerColor(kBlue);
	gr_track_rate_TIB.SetTitle("TIB Track Rate");
	gr_track_rate_TIB.Draw("AP");
	c.SetGrid();
        c.SaveAs("tib_track_rate.png");
	c.Clear();
	gSystem->Exec("mv tib_track_rate.png Rate_Plots");

//-----------------------------------------------------------------------------------------------


//++++++++++++++++++++++++++++++++  TID track rate  ++++++++++++++++++++++++++++++++++++++++++++ 

	TGraphErrors gr_track_rate_TID(runNumber_VecD,track_rate_TID_VecD,runNumber_err_VecD,track_rate_TID_err_VecD);
	gr_track_rate_TID.GetXaxis()->SetTitle("Run Number");
	gr_track_rate_TID.GetXaxis()->SetLabelSize(0.03);
	gr_track_rate_TID.GetXaxis()->SetNoExponent();
	gr_track_rate_TID.GetYaxis()->SetTitle("Track Rate (in Hz)");
	gr_track_rate_TID.SetMarkerStyle(20);
	gr_track_rate_TID.SetMarkerSize(1.2);
	gr_track_rate_TID.SetMarkerColor(kBlue);
	gr_track_rate_TID.SetTitle("TID Track Rate");
	gr_track_rate_TID.Draw("AP");
	c.SetGrid();
        c.SaveAs("tid_track_rate.png");
	c.Clear();
	gSystem->Exec("mv tid_track_rate.png Rate_Plots");

//-----------------------------------------------------------------------------------------------


//++++++++++++++++++++++++++++++++  Total TEC track rate  ++++++++++++++++++++++++++++++++++++++++++++ 

	TGraphErrors gr_track_rate_TEC(runNumber_VecD,track_rate_TEC_VecD,runNumber_err_VecD,track_rate_TEC_err_VecD);
	gr_track_rate_TEC.GetXaxis()->SetTitle("Run Number");
	gr_track_rate_TEC.GetXaxis()->SetLabelSize(0.03);
	gr_track_rate_TEC.GetXaxis()->SetNoExponent();
	gr_track_rate_TEC.GetYaxis()->SetTitle("Track Rate (in Hz)");
	gr_track_rate_TEC.SetMarkerStyle(20);
	gr_track_rate_TEC.SetMarkerSize(1.2);
	gr_track_rate_TEC.SetMarkerColor(kBlue);
	gr_track_rate_TEC.SetTitle("TEC Track Rate");
	gr_track_rate_TEC.Draw("AP");
	c.SetGrid();
        c.SaveAs("tec_track_rate.png");
	c.Clear();
	gSystem->Exec("mv tec_track_rate.png Rate_Plots");

//-----------------------------------------------------------------------------------------------


//++++++++++++++++++++++++++++++++  TEC+/- track rate  ++++++++++++++++++++++++++++++++++++++++++++ 
	TMultiGraph mg("track rate","TRack Rate TEC+/-");		// Multigraph decleration
	
	TGraphErrors *gr_track_rate_TECP = new TGraphErrors(runNumber_VecD,track_rate_TECP_VecD,runNumber_err_VecD,track_rate_TECP_err_VecD);
	gr_track_rate_TECP->SetMarkerStyle(20);
	gr_track_rate_TECP->SetMarkerSize(1.2);
	gr_track_rate_TECP->SetMarkerColor(kBlack);

	TGraphErrors *gr_track_rate_TECM = new TGraphErrors(runNumber_VecD,track_rate_TECM_VecD,runNumber_err_VecD,track_rate_TECM_err_VecD);
	gr_track_rate_TECM->SetMarkerStyle(20);
	gr_track_rate_TECM->SetMarkerSize(1.2);
	gr_track_rate_TECM->SetMarkerColor(kRed);

	mg.Add(gr_track_rate_TECP);
	mg.Add(gr_track_rate_TECM);
	mg.Draw("AP");
	mg.GetXaxis()->SetTitle("Run Number");
	mg.GetXaxis()->SetNoExponent();
	mg.GetXaxis()->SetLabelSize(0.03);
	mg.GetYaxis()->SetTitle("Track Rate (in Hz)");
	
	TLegend leg(0.8,0.8,0.94,0.92);			// Legend for TEC+/-
	leg.AddEntry(gr_track_rate_TECP, "TEC+","p");
	leg.AddEntry(gr_track_rate_TECM, "TEC-","p");
	leg.SetBorderSize(1);
	leg.SetShadowColor(0);
	leg.SetFillColor(0);
	leg.Draw();
	c.SetGrid();
        c.SaveAs("tec_track_ratePM.png");
	c.Clear();
	gSystem->Exec("mv tec_track_ratePM.png Rate_Plots");

//-----------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------
//					Weighted Mean calculation
//-----------------------------------------------------------------------------------------------

						
        double total_weight		= 0;
	double weighted_mean_track_rate;
	double weighted_mean_track_rate_TEC;
	double weighted_mean_track_rate_TOB;
	double weighted_mean_track_rate_TIB;
	double weighted_mean_track_rate_TID;
	double weighted_mean_track_rate_FPIX;
	double weighted_mean_track_rate_BPIX;
	double weighted_mean_event_rate;
	
	for (int k = 0; k < j; k++)
	{
		weight.push_back( tracks.at(k)/total_tracks );
	}


	for (int a = 0; a < j ; a++)
	{
          weighted_mean_track_rate 		+= track_rate.at(a)     * weight.at(a) ; 		
          weighted_mean_track_rate_TEC 		+= track_rate_TEC.at(a) * weight.at(a) ; 	
          weighted_mean_track_rate_TOB 		+= track_rate_TOB.at(a) * weight.at(a) ; 	
          weighted_mean_track_rate_TIB 		+= track_rate_TIB.at(a) * weight.at(a) ;	
          weighted_mean_track_rate_TID 		+= track_rate_TID.at(a) * weight.at(a) ;	
          weighted_mean_track_rate_FPIX 	+= track_rate_FPIX.at(a)* weight.at(a) ;	
          weighted_mean_track_rate_BPIX 	+= track_rate_BPIX.at(a)* weight.at(a) ; 	
          weighted_mean_event_rate 		+= event_rate.at(a)     * weight.at(a) ; 		
	  total_weight			    	+= weight.at(a) ;
	}

//-----------------------------------------------------------------------------------------------
//			Summary Plot for track rate in each Subdetector				 
//-----------------------------------------------------------------------------------------------

   	TH1F h1b("h1b","rate summary",8,0,8);
   	h1b.SetFillColor(4);
   	h1b.SetBarWidth(0.3);
   	h1b.SetBarOffset(0.35);
   	h1b.SetStats(0);

	Bar_Ytitle[0] = weighted_mean_event_rate;     
	Bar_Ytitle[1] = weighted_mean_track_rate;     
	Bar_Ytitle[2] = weighted_mean_track_rate_FPIX;     
	Bar_Ytitle[3] = weighted_mean_track_rate_BPIX; 
	Bar_Ytitle[4] = weighted_mean_track_rate_TIB; 
	Bar_Ytitle[5] = weighted_mean_track_rate_TID;
	Bar_Ytitle[6] = weighted_mean_track_rate_TOB; 
	Bar_Ytitle[7] = weighted_mean_track_rate_TEC;

      for (int i=1; i<=8; i++)
        {
      	h1b.SetBinContent(i, Bar_Ytitle[i-1]);
      	h1b.GetXaxis()->SetBinLabel(i,Bar_Xtitle[i-1].c_str());
  	}

	gStyle->SetPaintTextFormat("1.4f");
	h1b.LabelsOption("d");
	h1b.SetLabelSize(0.04);
	h1b.GetYaxis()->SetTitle("Average Rate (Hz)");
	h1b.Draw("bTEXT");
	c.SaveAs("Summary_Chart.png");
        c.Close();
	gSystem->Exec("mv Summary_Chart.png Rate_Plots");


  }
   
   
   
   
   
   
