
#include<vector>


void DrawSuperposedHistos(TFile** File, char** Histos_Name, std::vector<char*> legend, int Rebin, char* Title,  char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax, bool save, char* save_path, bool special);
//void DrawTH2D(TFile** File, char** Histos_Name, std::vector<char*> legend, char* Title, bool save, char* save_path);
void DrawTProfile(TFile** File, char** Histos_Name, std::vector<char*> legend, char* Title, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax,  bool save, char* save_path);
void DrawTH2D  (TFile** File, char** Histos_Name, std::vector<char*> legend, char* Title, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax,  bool save, char* save_path, double MarkerSize);
void DrawTH2D  (TFile** File, char** Histos_Name, std::vector<char*> legend, char* Title, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax,  bool save, char* save_path);
void DrawTH2DZ(TFile* File, char* Histos_Name, char* legend, char* Title, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax,  bool save, char* save_path, bool Normalization);


void macro()
{
	gROOT->Reset();
	gStyle->SetCanvasColor(0);
        gStyle->SetPalette(1);
        gStyle->SetBarOffset(0);
	bool save = true;


        TFile* Run62966  = new TFile("file:../SiStripCalib.root");

        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_EtaTEC";                   legend.push_back("TEC");                    Files[0] = Run62966;
        Histos_Name[1] = "MPV_Vs_EtaTIB";                   legend.push_back("TIB");                    Files[1] = Run62966;
        Histos_Name[2] = "MPV_Vs_EtaTID";                   legend.push_back("TID");                    Files[2] = Run62966;
        Histos_Name[3] = "MPV_Vs_EtaTOB";                   legend.push_back("TOB");                    Files[3] = Run62966;
        DrawTH2D(Files, Histos_Name,legend, "", "module #eta", "MPV", -3.0,3.0,150,500,save,"Pictures/Run62966_MPV_Vs_EtaSubDet.png");
        delete [] Histos_Name; Histos_Name=NULL;     

        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "APV_MPV";                         legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "APV Pair Index", "MPV", 0, 0, 150, 500, save, "Pictures/Run62966_APV_MPV.png",true);

        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "APV_Gain";                        legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "APV Pair Index", "Gain", 0, 0, 0.5, 1.5, save, "Pictures/Run62966_APV_Gain.png",true);

        DrawTH2DZ(Run62966, "APV_Charge" ,"", "", "APV Pair Index", "Charge", 0,0,0,0,save,"Pictures/Run62966_APV_Charge.png", false);


        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPVs320";                         legend.push_back("320 #mum");               Files[0] = Run62966;
        Histos_Name[1] = "MPVs500";                         legend.push_back("500 #mum");               Files[1] = Run62966;
        Histos_Name[2] = "MPVs";                            legend.push_back("320 + 500 #mum");         Files[2] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "MPV", "a.u.", 150, 500, 0, 0, save, "Pictures/Run62966_MPVs.png",false);
        delete [] Histos_Name; Histos_Name=NULL;

        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_PathLength";               legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "Pathlength", "MPV", 0, 0, 0, 400, save, "Pictures/Run62966_MPV_Vs_PathLength.png",false);
        delete [] Histos_Name; Histos_Name=NULL;     
        DrawTH2DZ(Run62966, "Charge_Vs_PathLength" ,"", "", "PathLength", "Charge", 0,0,0,0,save,"Pictures/Run62966_Charge_Vs_PathLength.png", false);


        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_PathLength320";            legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "Pathlength", "MPV", 0, 0, 0, 400, save, "Pictures/Run62966_MPV_Vs_PathLength320.png",false);
        delete [] Histos_Name; Histos_Name=NULL;
        DrawTH2DZ(Run62966, "Charge_Vs_PathLength320" ,"", "", "PathLength", "Charge", 0,0,0,0,save,"Pictures/Run62966_Charge_Vs_PathLength320.png", false);


        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_PathLength500";            legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "Pathlength", "MPV", 0, 0, 0, 400, save, "Pictures/Run62966_MPV_Vs_PathLength500.png",false);
        delete [] Histos_Name; Histos_Name=NULL;
        DrawTH2DZ(Run62966, "Charge_Vs_PathLength500" ,"", "", "PathLength", "Charge", 0,0,0,0,save,"Pictures/Run62966_Charge_Vs_PathLength500.png", false);



        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_PathTIB";                  legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "Pathlength", "MPV", 0, 0, 0, 400, save, "Pictures/Run62966_MPV_Vs_PathLengthTIB.png",false);
        delete [] Histos_Name; Histos_Name=NULL;
        DrawTH2DZ(Run62966, "Charge_Vs_PathTIB" ,"", "", "PathLength", "Charge", 0,0,0,0,save,"Pictures/Run62966_Charge_Vs_PathLengthTIB.png", false);


        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_PathTID";                  legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "Pathlength", "MPV", 0, 0, 0, 400, save, "Pictures/Run62966_MPV_Vs_PathLengthTID.png",false);
        delete [] Histos_Name; Histos_Name=NULL;
        DrawTH2DZ(Run62966, "Charge_Vs_PathTID" ,"", "", "PathLength", "Charge", 0,0,0,0,save,"Pictures/Run62966_Charge_Vs_PathLengthTID.png", false);


        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_PathTOB";                  legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "Pathlength", "MPV", 0, 0, 0, 400, save, "Pictures/Run62966_MPV_Vs_PathLengthTOB.png",false);
        delete [] Histos_Name; Histos_Name=NULL;
        DrawTH2DZ(Run62966, "Charge_Vs_PathTOB" ,"", "", "PathLength", "Charge", 0,0,0,0,save,"Pictures/Run62966_Charge_Vs_PathLengthTOB.png", false);



        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_PathTEC1";                  legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "Pathlength", "MPV", 0, 0, 0, 400, save, "Pictures/Run62966_MPV_Vs_PathLengthTEC320.png",false);
        delete [] Histos_Name; Histos_Name=NULL;
        DrawTH2DZ(Run62966, "Charge_Vs_PathTEC1" ,"", "", "PathLength", "Charge", 0,0,0,0,save,"Pictures/Run62966_Charge_Vs_PathTEC320.png", false);


        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_Vs_PathTEC2";                  legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "Pathlength", "MPV", 0, 0, 0, 400, save, "Pictures/Run62966_MPV_Vs_PathLengthTEC500.png",false);
        delete [] Histos_Name; Histos_Name=NULL;
        DrawTH2DZ(Run62966, "Charge_Vs_PathTEC2" ,"", "", "PathLength", "Charge", 0,0,0,0,save,"Pictures/Run62966_Charge_Vs_PathTEC500.png", false);




        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "FirstStrip";                      legend.push_back("");                       Files[0] = Run62966;
        DrawSuperposedHistos(Files, Histos_Name, legend, 1, "",  "FirstStrip", "u.a.", 0, 0, 0, 0, save, "Pictures/Run62966_FirstStrip.png",false);
        delete [] Histos_Name; Histos_Name=NULL;


        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "Entries_Vs_Error";                legend.push_back("");                       Files[0] = Run62966;
        DrawTH2D(Files, Histos_Name,legend, "", "#Entries in Charge Distribution", "Fit Error on the MPV", 0,350,0,0,save,"Pictures/Run62966_Entries_Vs_Error.png");
        delete [] Histos_Name; Histos_Name=NULL;




/*
        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_vs_10RplusEta";               legend.push_back("");                       Files[0] = Run62966;
        DrawTH2D(Files, Histos_Name,legend, "", "R #times 10 + #eta", "MPV", 235,242,290,325,save,"Pictures/Run62966_MPV_Vs_10RplusEta_240.png");
        delete [] Histos_Name; Histos_Name=NULL;

        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_vs_10RplusEta";               legend.push_back("");                       Files[0] = Run62966;
        DrawTH2D(Files, Histos_Name,legend, "", "R #times 10 + #eta", "MPV", 1096,1102,305,325,save,"Pictures/Run62966_MPV_Vs_10RplusEta_1100.png");
        delete [] Histos_Name; Histos_Name=NULL;

        char** Histos_Name = new char*[4];                  std::vector<char*> legend;legend.clear();   TFile** Files = new TFile*[4];
        Histos_Name[0] = "MPV_vs_10RplusEta";               legend.push_back("");                       Files[0] = Run62966;
        DrawTH2D(Files, Histos_Name,legend, "", "R #times 10 + #eta", "MPV", 582,590,260,320,save,"Pictures/Run62966_MPV_Vs_10RplusEta_585_Sat.png");
        delete [] Histos_Name; Histos_Name=NULL;
*/

}


void DrawSuperposedHistos(TFile** File, char** Histos_Name, std::vector<char*> legend, int Rebin, char* Title,  char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax, bool save, char* save_path, bool special)
{
   int Color[]    = {2,4,1,4,1,6,3,9,5};
   int Marker[]   = {21,22,23,20,20,3,2};
   int Style[]    = {1,5,7,9,10};

   int    N         = legend.size();
   TH1D** Histos    = new TH1D*[N];

   double HistoMax      = -1;
   int    HistoHeighest = -1;

   for(int i=0;i<N;i++){
        Histos[i] = File[i]->FindObjectAny(Histos_Name[i]);
//        Histos[i] = File[i]->Get(Histos_Name[i]);
        Histos[i] = Histos[i]->Rebin(Rebin);
        Histos[i]->SetTitle();
        Histos[i]->SetStats(kFALSE);
//        Histos[i]->SetStats(kTRUE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend);
        Histos[i]->GetYaxis()->SetTitle(Ylegend);
        Histos[i]->GetYaxis()->SetTitleOffset(1.20);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
//        if(i==0)Histos[i]->SetFillColor(Color[i]);
        Histos[i]->SetFillColor(0);
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(0.5);

//        Histos[i]->SetLineStyle(Style[i]);
        Histos[i]->SetLineColor(Color[i]);
        Histos[i]->SetLineWidth(2.0);

        if(special){
           Histos[i]->SetLineColor(1);
           Histos[i]->SetLineWidth(1.0);
        }

        if(Histos[i]->GetMaximum() >= HistoMax){
	   HistoMax      = Histos[i]->GetMaximum();
           HistoHeighest = i;
	}

   }

   Histos[HistoHeighest]->Draw("");
   for(int i=0;i<N;i++){
        Histos[i]->Draw("Same");
   }


   if(strcmp(legend[0],"")!=0){
      TLegend* leg;
//    leg = new TLegend(0.10,0.90,0.45,0.90 - N*0.1);
      leg = new TLegend(0.55,0.90,0.90,0.90 - N*0.08);
      leg->SetFillColor(0);
      if(strcmp(Title,"")!=0)leg->SetHeader(Title);

      for(int i=0;i<N;i++){
          if(strcmp(legend[i],"")==0)continue;
          leg->AddEntry(Histos[i], legend[i] ,"L");
      }
      leg->Draw();
   }


   if(save==1){
//        char path[255]; sprintf(path,"Pictures/PNG/%s.png",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/EPS/%s.eps",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/C/%s.C"  ,save_path);  c1->SaveAs(path);
          c1->SaveAs(save_path);
   }
}

/*
void DrawTH2D(TFile** File, char** Histos_Name, std::vector<char*> legend, char* Title, bool save, char* save_path)
{
   int Color[]    = {2,4,8,1,4,6,3,9,5};
   int Marker[]   = {21,22,23,20,20,3,2};
   int Style[]    = {1,5,7,9,10};

   int    N         = legend.size();
   TH2D** Histos    = new TH2D*[N];

   double HistoMax      = -1;
   int    HistoHeighest = -1;

   for(int i=0;i<N;i++){
        Histos[i] = File[i]->Get(Histos_Name[i]);
        Histos[i]->SetTitle();
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle("\eta");
        Histos[i]->GetYaxis()->SetTitle("MPV");
        Histos[i]->GetYaxis()->SetTitleOffset(1.20);
//        Histos[i]->SetAxisRange(0,500,"X");
        Histos[i]->SetAxisRange(150,600,"Y");
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(1.5);
   }

   TLegend* leg;
   leg = new TLegend(0.10,0.90,0.50,0.90-N*0.1);
   leg->SetFillColor(0);
   leg->SetHeader(Title);

   Histos[0]->Draw("");
   for(int i=0;i<N;i++){
        Histos[i]->Draw("Same");
   }

   for(int i=0;i<N;i++){
      leg->AddEntry(Histos[i], legend[i] ,"P");
   }
   leg->Draw();


   if(save==1){
//        char path[255]; sprintf(path,"Pictures/PNG/%s.png",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/EPS/%s.eps",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/C/%s.C"  ,save_path);  c1->SaveAs(path);
           c1->SaveAs(save_path);

   }
}
*/


void DrawTProfile(TFile** File, char** Histos_Name, std::vector<char*> legend, char* Title, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax,  bool save, char* save_path)
{
   int Color[]    = {2,4,8,1,4,6,3,9,5};
   int Marker[]   = {21,22,23,20,20,3,2};
   int Style[]    = {1,5,7,9,10};

   int    N         = legend.size();
   TProfile** Histos    = new TProfile*[N];

   double HistoMax      = -1;
   int    HistoHeighest = -1;

   for(int i=0;i<N;i++){
        Histos[i] = File[i]->Get(Histos_Name[i]);
        Histos[i]->SetTitle();
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend);
        Histos[i]->GetYaxis()->SetTitle(Ylegend);
        Histos[i]->GetYaxis()->SetTitleOffset(1.20);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(1.5);
   }

   TLegend* leg;
//   leg = new TLegend(0.10,0.90,0.50,0.90-N*0.1);
   leg = new TLegend(0.90,0.90,0.55,0.90-N*0.1);
   leg->SetFillColor(0);
   if(strcmp(Title,"")!=0)leg->SetHeader(Title);

   Histos[0]->Draw("");
   for(int i=0;i<N;i++){
        Histos[i]->Draw("Same");
   }

   for(int i=0;i<N;i++){
      leg->AddEntry(Histos[i], legend[i] ,"P");
   }
   leg->Draw();


   if(save==1){
//        char path[255]; sprintf(path,"Pictures/PNG/%s.png",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/EPS/%s.eps",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/C/%s.C"  ,save_path);  c1->SaveAs(path);
           c1->SaveAs(save_path);

   }
}


void DrawTH2D(TFile** File, char** Histos_Name, std::vector<char*> legend, char* Title, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax,  bool save, char* save_path)
{
   DrawTH2D(File, Histos_Name, legend, Title, Xlegend, Ylegend, xmin, xmax, ymin, ymax,  save, save_path, 0.3);
}


void DrawTH2D(TFile** File, char** Histos_Name, std::vector<char*> legend, char* Title, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax,  bool save, char* save_path, double MarkerSize)
{
   int Color[]    = {4,2,1,3};
//   int Color[]    = {8, 4,2,1,4,6,3,9,5};
   int Marker[]   = {20,22,21,23,20,3,2};
   int Style[]    = {1,5,7,9,10};

   int    N         = legend.size();
   TH2D** Histos    = new TH2D*[N];


   double HistoMax      = -1;
   int    HistoHeighest = -1;

   for(int i=0;i<N;i++){
        Histos[i] = File[i]->Get(Histos_Name[i]);
        Histos[i]->SetTitle();
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend);
        Histos[i]->GetYaxis()->SetTitle(Ylegend);
        Histos[i]->GetYaxis()->SetTitleOffset(1.20);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(MarkerSize);
//        Histos[i]->SetMarkerSize(1.3);

   }
   //     Histos[1]->SetMarkerSize(MarkerSize*1.5);

   Histos[0]->Draw("");
   for(int i=0;i<N;i++){
        Histos[i]->Draw("Same");
   }


   if(strcmp(legend[0],"")!=0){
      TLegend* leg;
//    leg = new TLegend(0.10,0.90,0.50,0.90-N*0.1);
      leg = new TLegend(0.90,0.90,0.55,0.90-N*0.05);
      leg->SetFillColor(0);
      if(strcmp(Title,"")!=0)leg->SetHeader(Title);

      for(int i=0;i<N;i++){
         TH2D* temp = Histos[i]->Clone();
         temp->SetMarkerSize(1.3);
         leg->AddEntry(temp, legend[i] ,"P");
//       Histos[i]->SetMarkerSize(MarkerSize);
      }
      leg->Draw();
   }


   if(save==1){
//        char path[255]; sprintf(path,"Pictures/PNG/%s.png",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/EPS/%s.eps",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/C/%s.C"  ,save_path);  c1->SaveAs(path);
           c1->SaveAs(save_path);

   }
   delete [] Histos;
}

void DrawTH2DZ(TFile* File, char* Histos_Name, char* legend, char* Title, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax,  bool save, char* save_path, bool Normalization)
{
   TH2D* Histos = File->Get(Histos_Name);
   Histos->SetTitle();
   Histos->SetStats(kFALSE);
   Histos->GetXaxis()->SetTitle(Xlegend);
   Histos->GetYaxis()->SetTitle(Ylegend);
   Histos->GetYaxis()->SetTitleOffset(1.20);
//   Histos->GetYaxis()->SetTitleOffset(1.0);
   if(xmin!=xmax)Histos->SetAxisRange(xmin,xmax,"X");
   if(ymin!=ymax)Histos->SetAxisRange(ymin,ymax,"Y");

   if(Normalization){
      for(int x=0;x<Histos->GetXaxis()->GetNbins();x++){     
         TH1D* tmp = Histos->ProjectionY("",x,x);
	 double Integral = tmp->Integral();
         if(Integral==0)continue;
         double Factor = 1/Integral;
         for(int y=0;y<Histos->GetYaxis()->GetNbins();y++){
            Histos->SetBinContent(x,y, Histos->GetBinContent(x,y)*Factor );
            Histos->SetBinError  (x,y, Histos->GetBinError  (x,y)*Factor );
         }
      }
   }




   Histos->Draw("COLZ");
   gPad->SetLogz(1);
/*   c1->Update();
   TPaletteAxis* palette = (TPaletteAxis*)Histos->GetListOfFunctions()->FindObject("palette");
   palette->SetLabelOffset(0.1);
   palette->SetTitleOffset(0.1);
   c1->Modified();
*/

  
   if(save==1){
//        char path[255]; sprintf(path,"Pictures/PNG/%s.png",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/EPS/%s.eps",save_path);  c1->SaveAs(path);
//        char path[255]; sprintf(path,"Pictures/C/%s.C"  ,save_path);  c1->SaveAs(path);
           c1->SaveAs(save_path);

   }
}


