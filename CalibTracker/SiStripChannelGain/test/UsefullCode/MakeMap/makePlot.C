
#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TPaveText.h"
#include "TPaveStats.h"
#include "tdrstyle.C"

int Color [] = {2,4,1,8,6,7,3,9,5};
int Marker[] = {21,22,23,20,29,3,2};
int Style [] = {1,2,5,7,9,10};

char buffer[2048];


char* FileName;

void Core(char* SavePath);


TObject* GetObjectFromPath(TDirectory* File, const char* Path);
void SaveCanvas(TCanvas* c, const char* path, const char* name, bool OnlyPPNG=false);
void DrawStatBox(TObject** Histos, std::vector<char*> legend, bool Mean               , double X=0.15, double Y=0.93, double W=0.15, double H=0.03);
void DrawLegend (TObject** Histos, std::vector<char*> legend, char* Title, char* Style, double X=0.79, double Y=0.93, double W=0.20, double H=0.05);
void DrawSuperposedHistos(TH1D** Histos, std::vector<char*> legend, char* Style,  char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax);
void DrawTH2D   (TH2D**    Histos, std::vector<char*> legend, char* Style, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax);

void makePlot()
{
   setTDRStyle();
   gStyle->SetPadLeftMargin  (0.18);
   gStyle->SetPadTopMargin   (0.05);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetCanvasColor(0);
   gStyle->SetBarOffset(0);

   TCanvas* c1;
   TObject** Histos = new TObject*[10];
   std::vector<char*> legend;

   FileName        = "ProbaMap.root";
   system("mkdir -p pictures");
   Core("pictures/");
}


void Core(char* SavePath)
{
   TFile* f1               = new TFile(FileName);   
   TH3F*  Charge_Vs_Path3D = (TH3F*)GetObjectFromPath(f1,"Charge_Vs_Path");
   for(int x=0;x<15;x++){
      char xProjName[255];
      if(x==0){
         sprintf(xProjName,"%s","inc");
         Charge_Vs_Path3D->GetXaxis()->SetRange(0,15);
      }else{
         sprintf(xProjName,"%02i",x);
         Charge_Vs_Path3D->GetXaxis()->SetRange(x,x);
      }
      printf("---------------\n%s------------\n",xProjName);
      string xProjNameStr(xProjName);


      TH2D*  Charge_Vs_Path2D = (TH2D*)Charge_Vs_Path3D->Project3D("zy");
      double binMinA = Charge_Vs_Path2D->GetXaxis()->GetBinLowEdge(4);
      double binMaxA = Charge_Vs_Path2D->GetXaxis()->GetBinUpEdge(6);

      TH1D*  Charge_Vs_PathA  = (TH1D*)Charge_Vs_Path2D->ProjectionY("projA",4,6);
      Charge_Vs_PathA->Rebin(2);
      char ALegend[1024];sprintf(ALegend,"[%5.2f,%5.2f]",binMinA,binMaxA);


      double binMinB = Charge_Vs_Path2D->GetXaxis()->GetBinLowEdge(7);
      double binMaxB = Charge_Vs_Path2D->GetXaxis()->GetBinUpEdge(9);
      TH1D*  Charge_Vs_PathB  = (TH1D*)Charge_Vs_Path2D->ProjectionY("projB",7,9);
      Charge_Vs_PathB->Rebin(2);
      char BLegend[1024];sprintf(BLegend,"[%5.2f,%5.2f]",binMinB,binMaxB);

      double binMinC = Charge_Vs_Path2D->GetXaxis()->GetBinLowEdge(10);
      double binMaxC = Charge_Vs_Path2D->GetXaxis()->GetBinUpEdge(12);
      TH1D*  Charge_Vs_PathC  = (TH1D*)Charge_Vs_Path2D->ProjectionY("projC",10,12);
      Charge_Vs_PathC->Rebin(2);
      char CLegend[1024];sprintf(CLegend,"[%5.2f,%5.2f]",binMinC,binMaxC);

      double binMinD = Charge_Vs_Path2D->GetXaxis()->GetBinLowEdge(13);
      double binMaxD = Charge_Vs_Path2D->GetXaxis()->GetBinUpEdge(15);
      TH1D*  Charge_Vs_PathD  = (TH1D*)Charge_Vs_Path2D->ProjectionY("projD",13,15);
      Charge_Vs_PathD->Rebin(2);
      char DLegend[1024];sprintf(DLegend,"[%5.2f,%5.2f]",binMinD,binMaxD);

      printf("%f to %f\n",binMinA,binMaxA);
      printf("%f to %f\n",binMinB,binMaxB);
      printf("%f to %f\n",binMinC,binMaxC);
      printf("%f to %f\n",binMinD,binMaxD);


      TCanvas* c0;
      TObject** Histos = new TObject*[10]; 
      std::vector<char*> legend;

      c0  = new TCanvas("c0", "c0", 600,600);
      Charge_Vs_Path2D->SetTitle("");
      Charge_Vs_Path2D->SetStats(kFALSE);
      Charge_Vs_Path2D->GetXaxis()->SetTitle("pathlength (mm)");
      Charge_Vs_Path2D->GetYaxis()->SetTitle("#Delta E/#Delta x (ADC/mm)");
      Charge_Vs_Path2D->GetYaxis()->SetTitleOffset(1.80);
      Charge_Vs_Path2D->Draw("COLZ");

      c0->SetLogz(true);
      SaveCanvas(c0,SavePath,(xProjNameStr+"_TH2").c_str());
      delete c0;


      //Compute Probability Map.
      TH2D* Prob_ChargePath  = new TH2D ("Prob_ChargePath"     , "Prob_ChargePath" , Charge_Vs_Path2D->GetXaxis()->GetNbins(), Charge_Vs_Path2D->GetXaxis()->GetXmin(), Charge_Vs_Path2D->GetXaxis()->GetXmax(), Charge_Vs_Path2D->GetYaxis()->GetNbins(), Charge_Vs_Path2D->GetYaxis()->GetXmin(), Charge_Vs_Path2D->GetYaxis()->GetXmax());
      for(int j=0;j<=Prob_ChargePath->GetXaxis()->GetNbins()+1;j++){
	 double Ni = 0;
	 for(int k=0;k<=Prob_ChargePath->GetYaxis()->GetNbins()+1;k++){ Ni+=Charge_Vs_Path2D->GetBinContent(j,k);} 

	 for(int k=0;k<=Prob_ChargePath->GetYaxis()->GetNbins()+1;k++){
	    double tmp = 1E-10;
	    for(int l=0;l<=k;l++){ tmp+=Charge_Vs_Path2D->GetBinContent(j,l);}

	    if(Ni>0){
	       Prob_ChargePath->SetBinContent (j, k, tmp/Ni);
	    }else{
	       Prob_ChargePath->SetBinContent (j, k, 0);
	    }
	 }
      }

      c0  = new TCanvas("c0", "c0", 600,600);
      Prob_ChargePath->SetTitle("Probability MIP(#DeltaE/#DeltaX) < Obs (#DeltaE/#DeltaX)");
      Prob_ChargePath->SetStats(kFALSE);
      Prob_ChargePath->GetXaxis()->SetTitle("pathlength (mm)");
      Prob_ChargePath->GetYaxis()->SetTitle("Observed #DeltaE/#DeltaX (ADC/mm)");
      Prob_ChargePath->GetYaxis()->SetTitleOffset(1.80);
      Prob_ChargePath->GetXaxis()->SetRangeUser(0.28,1.2);
//      Prob_ChargePath->GetYaxis()->SetRangeUser(0,1000);
      Prob_ChargePath->Draw("COLZ");

      //c0->SetLogz(true);
      SaveCanvas(c0,SavePath,(xProjNameStr+"_TH2Proba").c_str());
      delete c0;

      c0 = new TCanvas("c1","c1,",600,600);          legend.clear();
      Histos[0] = Charge_Vs_PathA;                   legend.push_back(ALegend);
      Histos[1] = Charge_Vs_PathB;                   legend.push_back(BLegend);
      Histos[2] = Charge_Vs_PathC;                   legend.push_back(CLegend);
      Histos[3] = Charge_Vs_PathD;                   legend.push_back(DLegend);
      if(((TH1*)Histos[0])->Integral()>=1)((TH1*)Histos[0])->Scale(1/((TH1*)Histos[0])->Integral());
      if(((TH1*)Histos[1])->Integral()>=1)((TH1*)Histos[1])->Scale(1/((TH1*)Histos[1])->Integral());
      if(((TH1*)Histos[2])->Integral()>=1)((TH1*)Histos[2])->Scale(1/((TH1*)Histos[2])->Integral());
      if(((TH1*)Histos[3])->Integral()>=1)((TH1*)Histos[3])->Scale(1/((TH1*)Histos[3])->Integral());
   //   DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Normalized Cluster Charge (ADC/mm)", "u.a.", 0,1200, 0,0);
      DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Normalized Cluster Charge (ADC/mm)", "u.a.", 0,600, 0,0);
      DrawLegend(Histos,legend,"PathLength (mm):","L");
      c0->SetGridx(true);
      Charge_Vs_PathA->GetXaxis()->SetNdivisions(520);
      SaveCanvas(c0,SavePath,(xProjNameStr+"_TH1Linear").c_str());
      delete c0;


      c0 = new TCanvas("c1","c1,",600,600);          legend.clear();
      Histos[0] = Charge_Vs_PathA;                   legend.push_back(ALegend);
      Histos[1] = Charge_Vs_PathB;                   legend.push_back(BLegend);
      Histos[2] = Charge_Vs_PathC;                   legend.push_back(CLegend);
      Histos[3] = Charge_Vs_PathD;                   legend.push_back(DLegend);
      if(((TH1*)Histos[0])->Integral()>=1)((TH1*)Histos[0])->Scale(1/((TH1*)Histos[0])->Integral());
      if(((TH1*)Histos[1])->Integral()>=1)((TH1*)Histos[1])->Scale(1/((TH1*)Histos[1])->Integral());
      if(((TH1*)Histos[2])->Integral()>=1)((TH1*)Histos[2])->Scale(1/((TH1*)Histos[2])->Integral());
      if(((TH1*)Histos[3])->Integral()>=1)((TH1*)Histos[3])->Scale(1/((TH1*)Histos[3])->Integral());
      DrawSuperposedHistos((TH1D**)Histos, legend, "",  "Normalized Cluster Charge (ADC/mm)", "u.a.", 0,3000, 0,0);
//      DrawLegend(Histos,legend,"PathLength (mm):","L");
      c0->SetLogy(true);
      SaveCanvas(c0,SavePath,(xProjNameStr+"_TH1").c_str());
      delete c0;




      delete Charge_Vs_Path2D;
      delete Charge_Vs_PathA;
      delete Charge_Vs_PathB;
      delete Charge_Vs_PathC;
      delete Charge_Vs_PathD;
      delete Prob_ChargePath;

   }
}




TObject* GetObjectFromPath(TDirectory* File, const char* Path)
{
   string str(Path);
   size_t pos = str.find("/");

   if(pos < 256){
      string firstPart = str.substr(0,pos);
      string endPart   = str.substr(pos+1,str.length());
      TDirectory* TMP = (TDirectory*)File->Get(firstPart.c_str());
      if(TMP!=NULL)return GetObjectFromPath(TMP,endPart.c_str());

      printf("BUG\n");
      return NULL;
   }else{
      return File->Get(Path);
   }

}



void SaveCanvas(TCanvas* c, const char* path, const char* name, bool OnlyPPNG){
   char buff[1024];
   sprintf(buff,"%s/%s.png",path,name);  c->SaveAs(buff);   if(OnlyPPNG)return;
   sprintf(buff,"%s/%s.eps",path,name);  c->SaveAs(buff);
   sprintf(buff,"%s/%s.C"  ,path,name);  c->SaveAs(buff);
}



void DrawLegend(TObject** Histos, std::vector<char*> legend, char* Title, char* Style, double X, double Y, double W, double H)
{
   int    N             = legend.size();

   if(strcmp(legend[0],"")!=0){
      TLegend* leg;
      leg = new TLegend(X,Y,X-W,Y - N*H);
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      //leg->SetTextAlign(32);
      if(strcmp(Title,"")!=0)leg->SetHeader(Title);

      if(strcmp(Style,"DataMC")==0){
         for(int i=0;i<N;i++){
            TH2D* temp = (TH2D*)Histos[i];//->Clone();
            temp->SetMarkerSize(1.3);
            if(i==0){
               leg->AddEntry(temp, legend[i] ,"P");
            }else{
               leg->AddEntry(temp, legend[i] ,"L");
            }
         }
      }else{
         for(int i=0;i<N;i++){
            TH2D* temp = (TH2D*)Histos[i];//->Clone();
            temp->SetMarkerSize(1.3);
            leg->AddEntry(temp, legend[i] ,Style);
         }
      }
      leg->Draw();
   }
}


void DrawStatBox(TObject** Histos, std::vector<char*> legend, bool Mean, double X, double Y, double W, double H)
{
   int    N             = legend.size();
   char   buffer[255];

   if(Mean)H*=3;
   for(int i=0;i<N;i++){
           TPaveText* stat = new TPaveText(X,Y-(i*H), X+W, Y-(i+1)*H, "NDC");
           TH1* Histo = (TH1*)Histos[i];
           sprintf(buffer,"Entries : %i\n",(int)Histo->GetEntries());
           stat->AddText(buffer);

           if(Mean){
           sprintf(buffer,"Mean    : %6.2f\n",Histo->GetMean());
           stat->AddText(buffer);

           sprintf(buffer,"RMS     : %6.2f\n",Histo->GetRMS());
           stat->AddText(buffer);
           }

           stat->SetFillColor(0);
           stat->SetLineColor(Color[i]);
           stat->SetTextColor(Color[i]);
           stat->SetBorderSize(0);
           stat->SetMargin(0.05);
           stat->SetTextAlign(12);
           stat->Draw();
   }
}



void DrawTH2D(TH2D** Histos, std::vector<char*> legend, char* Style, char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax)
{
   int    N             = legend.size();

   for(int i=0;i<N;i++){
        if(!Histos[i])continue;
        Histos[i]->SetTitle("");
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend);
        Histos[i]->GetYaxis()->SetTitle(Ylegend);
        Histos[i]->GetYaxis()->SetTitleOffset(1.60);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(0.3);
   }

   char Buffer[256];
   Histos[0]->Draw(Style);
   for(int i=1;i<N;i++){
        sprintf(Buffer,"%s same",Style);
        Histos[i]->Draw(Buffer);
   }
}

void DrawSuperposedHistos(TH1D** Histos, std::vector<char*> legend, char* Style,  char* Xlegend, char* Ylegend, double xmin, double xmax, double ymin, double ymax)
{
   int    N             = legend.size();

   double HistoMax      = -1;
   int    HistoHeighest = -1;

   for(int i=0;i<N;i++){
        if(!Histos[i])continue;
        Histos[i]->SetTitle("");
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend);
        Histos[i]->GetYaxis()->SetTitle(Ylegend);
        Histos[i]->GetXaxis()->SetTitleOffset(1.20);
        Histos[i]->GetYaxis()->SetTitleOffset(1.70);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetFillColor(0);
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(0.5);
        Histos[i]->SetLineColor(Color[i]);
        Histos[i]->SetLineWidth(2);
       if(strcmp(Style,"DataMC")==0 && i==0){
           Histos[i]->SetFillColor(0);
           Histos[i]->SetMarkerStyle(20);
           Histos[i]->SetMarkerColor(1);
           Histos[i]->SetMarkerSize(1);
           Histos[i]->SetLineColor(1);
           Histos[i]->SetLineWidth(2);
       } 

        if(Histos[i]->GetMaximum() >= HistoMax){
           HistoMax      = Histos[i]->GetMaximum();
           HistoHeighest = i;
        }

   }

   char Buffer[256];
   if(strcmp(Style,"DataMC")==0){
      if(HistoHeighest==0){
         Histos[HistoHeighest]->Draw("E1");
      }else{
         Histos[HistoHeighest]->Draw("HIST");
      }
      for(int i=0;i<N;i++){        
           if(i==0){
              Histos[i]->Draw("same E1");
           }else{
              Histos[i]->Draw("same");
           }
      }
   }else{
      Histos[HistoHeighest]->Draw(Style);
      for(int i=0;i<N;i++){        
           if(strcmp(Style,"")!=0){
              sprintf(Buffer,"same %s",Style);
           }else{
              sprintf(Buffer,"same");
           }
           Histos[i]->Draw(Buffer);
      }
   }
}
