

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TPaveText.h"
#include "PlotFunction.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include<vector>
#include<tdrstyle.C>

std::map<unsigned int, double> RunToIntLumi;

bool LoadLumiToRun();
double getLumiFromRun(unsigned int run);

struct stLayerData{
   std::map<unsigned int, double      > LayerGain;
   std::map<unsigned int, double      > LayerGainErr;
   std::map<unsigned int, unsigned int> LayerN;
   std::map<unsigned int, string      > LayerName;
};

void PlotMacro_Core(string input, string moduleName, string output);
TF1*  getLandau(TH1* InputHisto, double* FitResults, double LowRange=50, double HighRange=5400);
TH1D* ChargeToMPV(TH2* InputHisto, string Name, bool DivideByX);

void GetAverageGain(string input, string moduleName, stLayerData& layerData );


int DataType = 2;

void PlotMacro(){
   gROOT->Reset();
   setTDRStyle();
   gStyle->SetPadTopMargin   (0.05);
   gStyle->SetPadBottomMargin(0.10);
   gStyle->SetPadRightMargin (0.18);
   gStyle->SetPadLeftMargin  (0.13);
   gStyle->SetTitleSize(0.04, "XYZ");
   gStyle->SetTitleXOffset(1.1);
   gStyle->SetTitleYOffset(1.35);
   gStyle->SetPalette(1);
   gStyle->SetCanvasColor(0);
   gStyle->SetBarOffset(0);

   std::vector<std::pair<unsigned int, unsigned int> > runRanges;
//   //RUN1
//   runRanges.push_back(std::make_pair(190645,191271));
//   runRanges.push_back(std::make_pair(193093,194108));
//   runRanges.push_back(std::make_pair(194115,194428));
//   runRanges.push_back(std::make_pair(194429,194790));
//   runRanges.push_back(std::make_pair(194825,195530));
//   runRanges.push_back(std::make_pair(195540,195868));
//   runRanges.push_back(std::make_pair(195915,199021));
//   runRanges.push_back(std::make_pair(199318,199868));
//   runRanges.push_back(std::make_pair(200042,200601));
//   runRanges.push_back(std::make_pair(200961,201229));
//   runRanges.push_back(std::make_pair(201278,202084));
//   runRanges.push_back(std::make_pair(202093,203742));
//   runRanges.push_back(std::make_pair(203777,204576));
//   runRanges.push_back(std::make_pair(204577,205238));
//   runRanges.push_back(std::make_pair(205303,205666));
//   runRanges.push_back(std::make_pair(205667,206199));
//   runRanges.push_back(std::make_pair(206257,206940));
//   runRanges.push_back(std::make_pair(207214,207883));

//   //RUN2
   runRanges.push_back(std::make_pair(247252,247988));
   runRanges.push_back(std::make_pair(247989,247990));
   runRanges.push_back(std::make_pair(247992,247992));
   runRanges.push_back(std::make_pair(248003,248025));
   runRanges.push_back(std::make_pair(248028,250927));
   runRanges.push_back(std::make_pair(251168,251168));
   runRanges.push_back(std::make_pair(251244,251244));
   runRanges.push_back(std::make_pair(251251,251251));
   runRanges.push_back(std::make_pair(251252,251252));
   runRanges.push_back(std::make_pair(251328,251612));
   runRanges.push_back(std::make_pair(251638,251721));

   std::vector<stLayerData> resultVec;

   system("mkdir Pictures");
   for(unsigned int i=0;i<runRanges.size();i++){
      stLayerData result;

      char filePath[256]; sprintf(filePath, "../../Data_Run_%06i_to_%06i_PCL/Gains_Tree.root", runRanges[i].first, runRanges[i].second);
      GetAverageGain(filePath, "SiStripCalib/", result);
      resultVec.push_back(result);
   }  

   LoadLumiToRun();

   std::map<string, TGraph*> graphMap;
   printf("%10s --> ", "Run Min");for(unsigned int i=0;i<runRanges.size();i++){printf("%06i " , runRanges[i].first );}printf("\n");
   printf("%10s --> ", "Run Max");for(unsigned int i=0;i<runRanges.size();i++){printf("%06i " , runRanges[i].second);}printf("\n");
   printf("%10s --> ", "LumiMin");for(unsigned int i=0;i<runRanges.size();i++){printf("%6.2f ", getLumiFromRun(runRanges[i].first) );}printf("\n");
   printf("%10s --> ", "LumiMax");for(unsigned int i=0;i<runRanges.size();i++){printf("%6.2f ", getLumiFromRun(runRanges[i].second) );}printf("\n");
   printf("%10s     ", "-------");for(unsigned int i=0;i<runRanges.size();i++){printf("%06s-" , "------"           );}printf("\n");
   for(std::map<unsigned int, string>::iterator it=resultVec[0].LayerName.begin(); it!=resultVec[0].LayerName.end();it++){
      TGraphErrors* graph = new TGraphErrors(runRanges.size());
 
      printf("%10s --> ", it->second.c_str());
      for(unsigned int i=0;i<runRanges.size();i++){
         graph->SetPoint     (i, getLumiFromRun(runRanges[i].first) + (getLumiFromRun(runRanges[i].second)-getLumiFromRun(runRanges[i].first))/2.0 , resultVec[i].LayerGain[it->first]);
         graph->SetPointError(i,                                      0.0, 0.0);//resultVec[i].LayerGainErr[it->first] / resultVec[0].LayerGain[it->first]);

//         graph->SetPoint     (i, getLumiFromRun(runRanges[i].first) + (getLumiFromRun(runRanges[i].second)-getLumiFromRun(runRanges[i].first))/2.0 , resultVec[i].LayerGain[it->first] / resultVec[0].LayerGain[it->first]);
//         graph->SetPointError(i,                                      (getLumiFromRun(runRanges[i].second)-getLumiFromRun(runRanges[i].first))/2.0 , 0.0);//resultVec[i].LayerGainErr[it->first] / resultVec[0].LayerGain[it->first]);
         printf("%6.2f ", resultVec[i].LayerGain[it->first]);
//         printf("%6.2f+-%5.2f ", resultVec[i].LayerGain[it->first], resultVec[i].LayerGainErr[it->first]);
      }printf("\n");
      graphMap[it->second.c_str()] = graph;
   }


   TCanvas* c1 = new TCanvas("c1","c1,",900,600);

   TH1D* frame = new TH1D("frame", "frame", 1,getLumiFromRun(runRanges[0].first)-20,getLumiFromRun(runRanges[runRanges.size()-1].second)+20);
   frame->GetXaxis()->SetNdivisions(505);
   frame->SetTitle("");
   frame->SetStats(kFALSE);
//   frame->GetXaxis()->SetTitle("Integrated Lumi (fb^{-1})");
     frame->GetXaxis()->SetTitle("Run Number");
//   frame->GetYaxis()->SetTitle("Average Gain");
//   frame->SetMaximum(1.25);
//   frame->SetMinimum(0.80);
   frame->GetYaxis()->SetTitle("Average Gain");
   frame->SetMaximum(1.10);
   frame->SetMinimum(0.85);
   frame->GetYaxis()->SetTitleOffset(1.50);

   TLegend* leg = new TLegend(0.15,0.93,0.80, 0.75);
   leg->SetFillStyle(0);
   leg->SetBorderSize(0);
   leg->SetTextFont(43);
   leg->SetTextSize(20);
   leg->SetNColumns(4);
   int colorTIBTOB[]={kBlue, kBlue+1, kBlue+2, kBlue+3,  kRed-10, kRed-8, kRed-6, kRed-4, kRed-2, kRed};
   int colorTIDTEC[]={kRed-10, kRed-9, kRed-8, kRed-7, kRed-6, kRed-5, kRed-4, kRed-3, kRed-2, kBlue, kBlue+1, kBlue+2};
   int colorTIDTECr[]={kRed-10, kRed-9, kRed-8, kRed-7, kRed-6, kRed-4, kRed-2, kBlue, kBlue+1, kBlue+2};
   int L;


   L=0;
   frame->Draw("AXIS");
   leg->Clear();
   for(std::map<string, TGraph*>::iterator it=graphMap.begin(); it!=graphMap.end();it++){
      if(it->first.find("lTIB")==string::npos && it->first.find("lTOB")==string::npos)continue;
      it->second->SetLineColor(colorTIBTOB[L]);
      it->second->SetLineWidth(2);
      it->second->Draw("same L*");
      leg->AddEntry(it->second, it->first.c_str() ,"L");
      L++;
   }
   leg->Draw();
   c1->SaveAs("Pictures/TIBTOB.png");
   c1->SaveAs("Pictures/TIBTOB.C");


   L=0;
   frame->Draw("AXIS");
   leg->Clear();
   for(std::map<string, TGraph*>::iterator it=graphMap.begin(); it!=graphMap.end();it++){
      if(it->first.find("wTID+")==string::npos && it->first.find("wTEC+")==string::npos)continue;
      it->second->SetLineColor(colorTIDTEC[L]);
      it->second->SetLineWidth(2);
      it->second->Draw("same L*");
      leg->AddEntry(it->second, it->first.c_str() ,"L");
      L++;
   }
   leg->Draw();
   c1->SaveAs("Pictures/WheelTIDTECp.png");
   c1->SaveAs("Pictures/WheelTIDTECp.C");

   L=0;
   frame->Draw("AXIS");
   leg->Clear();
   for(std::map<string, TGraph*>::iterator it=graphMap.begin(); it!=graphMap.end();it++){
      if(it->first.find("wTID-")==string::npos && it->first.find("wTEC-")==string::npos)continue;
      it->second->SetLineColor(colorTIDTEC[L]);
      it->second->SetLineWidth(2);
      it->second->Draw("same L*");
      leg->AddEntry(it->second, it->first.c_str() ,"L");
      L++;
   }
   leg->Draw();
   c1->SaveAs("Pictures/WheelTIDTECm.png");
   c1->SaveAs("Pictures/WheelTIDTECm.C");


   L=0;
   frame->Draw("AXIS");
   leg->Clear();
   for(std::map<string, TGraph*>::iterator it=graphMap.begin(); it!=graphMap.end();it++){
      if(it->first.find("rTID+")==string::npos && it->first.find("rTEC+")==string::npos)continue;
      it->second->SetLineColor(colorTIDTECr[L]);
      it->second->SetLineWidth(2);
      it->second->Draw("same L*");
      leg->AddEntry(it->second, it->first.c_str() ,"L");
      L++;
   }
   leg->Draw();
   c1->SaveAs("Pictures/RingTIDTECp.png");
   c1->SaveAs("Pictures/RingTIDTECp.C");
 
   L=0;
   frame->Draw("AXIS"); 
   leg->Clear(); 
   for(std::map<string, TGraph*>::iterator it=graphMap.begin(); it!=graphMap.end();it++){
      if(it->first.find("rTID-")==string::npos && it->first.find("rTEC-")==string::npos)continue;
      it->second->SetLineColor(colorTIDTECr[L]);
      it->second->SetLineWidth(2);
      it->second->Draw("same L*");
      leg->AddEntry(it->second, it->first.c_str() ,"L");
      L++;
   }
   leg->Draw();
   c1->SaveAs("Pictures/RingTIDTECm.png");
   c1->SaveAs("Pictures/RingTIDTECm.C");



   TFile* f1     = new TFile("clusternoise_ontrack_tob_l1.root");
   if(!f1 || f1->IsZombie() || !f1->IsOpen() || f1->TestBit(TFile::kRecovered)){
      printf("clusternoise file not found\nStop the execution of the code here\n");
      return;
   }
   TH1D* noiseH     = (TH1D*)GetObjectFromPath(f1,"clusternoise_ontrack_tob_l1");
   TH1D* noiseHnew  = (TH1D*)noiseH->Clone("noiseHnew");

   TGraph* noiseG    = new TGraph(noiseH->GetXaxis()->GetNbins()-1);
   TGraph* noiseGnew = new TGraph(noiseH->GetXaxis()->GetNbins()-1);

   unsigned int CurrRunRange =0;
   for(int b=1;b<noiseH->GetXaxis()->GetNbins();b++){
      unsigned int run; sscanf( noiseH->GetXaxis()->GetBinLabel(b), "%d", &run);
      double noise = noiseH->GetBinContent(b);     

//      //find current RunRange
//      for(unsigned int i=0;i<runRanges.size()-1;i++){
//         if(runRanges[i].first<run)CurrRunRange = i;
//      }

     if(run<206880)CurrRunRange=15;
     if(run<206037)CurrRunRange=13;
     if(run<202914)CurrRunRange=10;
     if(run<201820)CurrRunRange=7;
     if(run<199878)CurrRunRange=5;  //or 6?
     if(run<198301)CurrRunRange=3;  //or 4?
     if(run<194552)CurrRunRange=1;  //or 1?
     if(run<192701)CurrRunRange=0;


   //Mapping
//   unsigned int runs[] = {192701, 194552, 198301, 199878, 201820, 202914, 206037, 206880};
//   runRanges.push_back(std::make_pair(190645,191271)); 0
//   runRanges.push_back(std::make_pair(193093,194108)); 1
//   runRanges.push_back(std::make_pair(194115,194428)); 2
//   runRanges.push_back(std::make_pair(194429,194790)); 3
//   runRanges.push_back(std::make_pair(194825,195530)); 4
//   runRanges.push_back(std::make_pair(195540,195868)); 5
//   runRanges.push_back(std::make_pair(195915,199021)); 6
//   runRanges.push_back(std::make_pair(199318,199868)); 7
//   runRanges.push_back(std::make_pair(200042,200601)); 8
//   runRanges.push_back(std::make_pair(200961,201229)); 9
//   runRanges.push_back(std::make_pair(201278,202084)); 10
//   runRanges.push_back(std::make_pair(202093,203742)); 11
//   runRanges.push_back(std::make_pair(203777,204576)); 12
//   runRanges.push_back(std::make_pair(204577,205238)); 13
//   runRanges.push_back(std::make_pair(205303,205666)); 14
//   runRanges.push_back(std::make_pair(205667,206199)); 15
//   runRanges.push_back(std::make_pair(206257,206940)); 16
//   runRanges.push_back(std::make_pair(207214,207883)); 17




     stLayerData gainInfo = resultVec[CurrRunRange];
     double lTOB1Gain = gainInfo.LayerGain[5001];
     noiseHnew->SetBinContent(b,noise*lTOB1Gain);

     noiseG   ->SetPoint(b-1, run, noise);
     noiseGnew->SetPoint(b-1, run, noise*lTOB1Gain);

     //printf("%6i : %6.3f --> %6.3f\n",run, noise, noise*lTOB1Gain);
   }


   frame = new TH1D("frame", "frame", 1,190645, 208686);
   frame->GetXaxis()->SetNdivisions(510);
   frame->SetTitle("");
   frame->SetStats(kFALSE);
   frame->GetXaxis()->SetTitle("Run");
//   frame->GetYaxis()->SetTitle("Average Gain"); 
//   frame->SetMaximum(1.25);
//   frame->SetMinimum(0.80);
   frame->GetYaxis()->SetTitle("TOB1 Noise (on track)");
   frame->SetMaximum(7);
   frame->SetMinimum(6);
   frame->GetYaxis()->SetTitleOffset(1.50);
   frame->Draw("AXIS");

//   noiseG->Draw("A");
   noiseG->Draw("P");
   noiseGnew->SetMarkerColor(4);
   noiseGnew->Draw("P");
//   noiseHnew->SetLineColor(4);
//   noiseHnew->Draw("HIST same");



   unsigned int runs[] = {192701, 194552, 198301, 199878, 201820, 202914, 206037, 206880};
   for(unsigned int i=0;i<sizeof(runs)/sizeof(unsigned int);i++){
      //find the closest processed run
//      int closestRun = 0;
//      for(std::map<unsigned int, double>::iterator it = RunToIntLumi.begin(); it!=RunToIntLumi.end();it++){
//         if(it->first>runs[i] && abs(it->first-runs[i])<abs(closestRun-runs[i]))closestRun = it->first;
//      }
//      printf("Draw line for run %i at %f\n",closestRun, RunToIntLumi[closestRun]);

      TLine* line = NULL;
//      line = new TLine( RunToIntLumi[closestRun], ymin, RunToIntLumi[closestRun], ymax);
      line = new TLine( runs[i], 6, runs[i], 7);
      line->SetLineColor(1);
      line->SetLineWidth(2);
      line->SetLineStyle(2);
      line->Draw("same");
   }



   c1->SaveAs("Pictures/noise_tob_l1.png");   



}



void GetAverageGain(string input, string moduleName, stLayerData& layerData )
{
   FILE* pFile;
   TCanvas* c1;
   TObject** Histos = new TObject*[10];                
   std::vector<string> legend;

   unsigned int  tree_Index;
   unsigned int  tree_DetId;
   unsigned char tree_APVId;
   unsigned char tree_SubDet;
   float         tree_x;
   float         tree_y;
   float         tree_z;
   float         tree_Eta;
   float         tree_R;
   float         tree_Phi;
   float         tree_Thickness;
   float         tree_FitMPV;
   float         tree_FitMPVErr;
   float         tree_FitWidth;
   float         tree_FitWidthErr;
   float         tree_FitChi2NDF;
   double        tree_Gain;
   double        tree_PrevGain;
   double        tree_NEntries;
   bool          tree_isMasked;

   TFile* f1     = new TFile(input.c_str());
   TTree *t1     = (TTree*)GetObjectFromPath(f1,moduleName+"APVGain");

   t1->SetBranchAddress("Index"             ,&tree_Index      );
   t1->SetBranchAddress("DetId"             ,&tree_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree_APVId      );
   t1->SetBranchAddress("SubDet"            ,&tree_SubDet     );
   t1->SetBranchAddress("x"                 ,&tree_x          );
   t1->SetBranchAddress("y"                 ,&tree_y          );
   t1->SetBranchAddress("z"                 ,&tree_z          );
   t1->SetBranchAddress("Eta"               ,&tree_Eta        );
   t1->SetBranchAddress("R"                 ,&tree_R          );
   t1->SetBranchAddress("Phi"               ,&tree_Phi        );
   t1->SetBranchAddress("Thickness"         ,&tree_Thickness  );
   t1->SetBranchAddress("FitMPV"            ,&tree_FitMPV     );
   t1->SetBranchAddress("FitMPVErr"         ,&tree_FitMPVErr  );
   t1->SetBranchAddress("FitWidth"          ,&tree_FitWidth   );
   t1->SetBranchAddress("FitWidthErr"       ,&tree_FitWidthErr);
   t1->SetBranchAddress("FitChi2NDF"        ,&tree_FitChi2NDF );
   t1->SetBranchAddress("Gain"              ,&tree_Gain       );
   t1->SetBranchAddress("PrevGain"          ,&tree_PrevGain   );
   t1->SetBranchAddress("NEntries"          ,&tree_NEntries   );
   t1->SetBranchAddress("isMasked"          ,&tree_isMasked   );

   int TreeStep = t1->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      t1->GetEntry(ientry);
      SiStripDetId SSdetId(tree_DetId);

       char LayerName[255];

      int LayerID=tree_SubDet*1000;      
      switch(tree_SubDet){         
         case 3:{
            TIBDetId tibid = TIBDetId(tree_DetId);
            LayerID += tibid.layer();
            sprintf(LayerName,"lTIB%i",tibid.layer());
         }break;
         case 4:{
            TIDDetId tidid = TIDDetId(tree_DetId);
            LayerID += tidid.side()*100;
            LayerID += tidid.ring();
            sprintf(LayerName,"rTID%c%i",tidid.side()==1?'-':'+', tidid.ring());
         }break;
         case 5:{
            TOBDetId tobid = TOBDetId(tree_DetId);
            LayerID += tobid.layer();
            sprintf(LayerName,"lTOB%i",tobid.layer());
         }break;
         case 6:{
            TECDetId tecid = TECDetId(tree_DetId);
            LayerID += tecid.side()*100;
            LayerID += tecid.ring();
            sprintf(LayerName,"rTEC%c%i",tecid.side()==1?'-':'+', tecid.ring());
         }break;
         default:
         break;
      }
      layerData.LayerGain[LayerID] += tree_Gain;
      layerData.LayerGainErr[LayerID] += tree_Gain*tree_Gain;
      layerData.LayerN   [LayerID] += 1;
      layerData.LayerName[LayerID]  = LayerName;


      LayerID=tree_SubDet*1000;
      switch(tree_SubDet){
         case 4:{
            TIDDetId tidid = TIDDetId(tree_DetId);
            LayerID += (2+tidid.side())*100;
            LayerID += tidid.wheel();
            sprintf(LayerName,"wTID%c%i",tidid.side()==1?'-':'+', tidid.wheel());
         }break;
         case 6:{
            TECDetId tecid = TECDetId(tree_DetId);
            LayerID += (2+tecid.side())*100;
            LayerID += tecid.wheel();
            sprintf(LayerName,"wTEC%c%i",tecid.side()==1?'-':'+', tecid.wheel());
         }break;
         default:
         break;
      }
      if(LayerID!=tree_SubDet*1000){
         layerData.LayerGain[LayerID] += tree_Gain;
         layerData.LayerGainErr[LayerID] += tree_Gain*tree_Gain;
         layerData.LayerN   [LayerID] += 1;
         layerData.LayerName[LayerID]  = LayerName;
      }
   }

   for(std::map<unsigned int, double>::iterator it=layerData.LayerGain.begin(); it!=layerData.LayerGain.end();it++){
      layerData.LayerGainErr[it->first] = sqrt((layerData.LayerGainErr[it->first] - (pow(layerData.LayerGain[it->first],2) / layerData.LayerN[it->first])) / (layerData.LayerN[it->first] - 1) );
      layerData.LayerGain[it->first] /= layerData.LayerN[it->first];
   }

}




void PlotMacro_Core(string input, string moduleName, string output)
{
   FILE* pFile;
   TCanvas* c1;
   TObject** Histos = new TObject*[10];                
   std::vector<string> legend;

   unsigned int  tree_Index;
   unsigned int  tree_DetId;
   unsigned char tree_APVId;
   unsigned char tree_SubDet;
   float         tree_x;
   float         tree_y;
   float         tree_z;
   float         tree_Eta;
   float         tree_R;
   float         tree_Phi;
   float         tree_Thickness;
   float         tree_FitMPV;
   float         tree_FitMPVErr;
   float         tree_FitWidth;
   float         tree_FitWidthErr;
   float         tree_FitChi2NDF;
   double        tree_Gain;
   double        tree_PrevGain;
   double        tree_NEntries;
   bool          tree_isMasked;

   TFile* f1     = new TFile(input.c_str());
   TTree *t1     = (TTree*)GetObjectFromPath(f1,moduleName+"APVGain");

   t1->SetBranchAddress("Index"             ,&tree_Index      );
   t1->SetBranchAddress("DetId"             ,&tree_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree_APVId      );
   t1->SetBranchAddress("SubDet"            ,&tree_SubDet     );
   t1->SetBranchAddress("x"                 ,&tree_x          );
   t1->SetBranchAddress("y"                 ,&tree_y          );
   t1->SetBranchAddress("z"                 ,&tree_z          );
   t1->SetBranchAddress("Eta"               ,&tree_Eta        );
   t1->SetBranchAddress("R"                 ,&tree_R          );
   t1->SetBranchAddress("Phi"               ,&tree_Phi        );
   t1->SetBranchAddress("Thickness"         ,&tree_Thickness  );
   t1->SetBranchAddress("FitMPV"            ,&tree_FitMPV     );
   t1->SetBranchAddress("FitMPVErr"         ,&tree_FitMPVErr  );
   t1->SetBranchAddress("FitWidth"          ,&tree_FitWidth   );
   t1->SetBranchAddress("FitWidthErr"       ,&tree_FitWidthErr);
   t1->SetBranchAddress("FitChi2NDF"        ,&tree_FitChi2NDF );
   t1->SetBranchAddress("Gain"              ,&tree_Gain       );
   t1->SetBranchAddress("PrevGain"          ,&tree_PrevGain   );
   t1->SetBranchAddress("NEntries"          ,&tree_NEntries   );
   t1->SetBranchAddress("isMasked"          ,&tree_isMasked   );


   TH2D* ChargeDistrib  = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_Index");
   TH2D* ChargeDistribA = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_Index_Absolute");

   TH2D* Charge_Vs_PathlengthTIB   = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTIB");
   TH2D* Charge_Vs_PathlengthTOB   = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTOB");
   TH2D* Charge_Vs_PathlengthTIDP  = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTIDP");
   TH2D* Charge_Vs_PathlengthTIDM  = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTIDM");
   TH2D* Charge_Vs_PathlengthTID   = (TH2D*)Charge_Vs_PathlengthTIDP->Clone("Charge_Vs_PathlengthTID");
         Charge_Vs_PathlengthTID      ->Add(Charge_Vs_PathlengthTIDM);
   TH2D* Charge_Vs_PathlengthTECP1 = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTECP1");
   TH2D* Charge_Vs_PathlengthTECP2 = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTECP2");
   TH2D* Charge_Vs_PathlengthTECM1 = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTECM1");
   TH2D* Charge_Vs_PathlengthTECM2 = (TH2D*)GetObjectFromPath(f1,moduleName+"Charge_Vs_PathlengthTECM2");
   TH2D* Charge_Vs_PathlengthTECP  = (TH2D*)Charge_Vs_PathlengthTECP1->Clone("Charge_Vs_PathlengthTECP");
         Charge_Vs_PathlengthTECP     ->Add(Charge_Vs_PathlengthTECP2);
   TH2D* Charge_Vs_PathlengthTECM  = (TH2D*)Charge_Vs_PathlengthTECM1->Clone("Charge_Vs_PathlengthTECM");
         Charge_Vs_PathlengthTECM     ->Add(Charge_Vs_PathlengthTECM2);
   TH2D* Charge_Vs_PathlengthTEC1  = (TH2D*)Charge_Vs_PathlengthTECP1->Clone("Charge_Vs_PathlengthTEC1");
         Charge_Vs_PathlengthTEC1     ->Add(Charge_Vs_PathlengthTECM1);
   TH2D* Charge_Vs_PathlengthTEC2  = (TH2D*)Charge_Vs_PathlengthTECP2->Clone("Charge_Vs_PathlengthTEC2");
         Charge_Vs_PathlengthTEC2     ->Add(Charge_Vs_PathlengthTECM2); 
   TH2D* Charge_Vs_PathlengthTEC   = (TH2D*)Charge_Vs_PathlengthTECP ->Clone("Charge_Vs_PathlengthTEC");
         Charge_Vs_PathlengthTEC      ->Add(Charge_Vs_PathlengthTECM );

   TH2D* Charge_Vs_PathlengthThin  = (TH2D*)Charge_Vs_PathlengthTEC1->Clone("Charge_Vs_PathlengthThin");
         Charge_Vs_PathlengthThin     ->Add(Charge_Vs_PathlengthTIB );
         Charge_Vs_PathlengthThin     ->Add(Charge_Vs_PathlengthTID );
   TH2D* Charge_Vs_PathlengthThick = (TH2D*)Charge_Vs_PathlengthTEC2->Clone("Charge_Vs_PathlengthThin");
         Charge_Vs_PathlengthThick    ->Add(Charge_Vs_PathlengthTOB );

   TH1D* MPV_Vs_PathlengthTIB      = ChargeToMPV(Charge_Vs_PathlengthTIB  ,"MPV_Vs_PathlengthTIB"  , true);
   TH1D* MPV_Vs_PathlengthTID      = ChargeToMPV(Charge_Vs_PathlengthTID  ,"MPV_Vs_PathlengthTID"  , true);
// TH1D* MPV_Vs_PathlengthTIDP     = ChargeToMPV(Charge_Vs_PathlengthTIDP ,"MPV_Vs_PathlengthTIDP" , true);
// TH1D* MPV_Vs_PathlengthTIDM     = ChargeToMPV(Charge_Vs_PathlengthTIDM ,"MPV_Vs_PathlengthTIDM" , true);



   TH1D* MPV_Vs_PathlengthTOB      = ChargeToMPV(Charge_Vs_PathlengthTOB  ,"MPV_Vs_PathlengthTOB"  , true);
// TH1D* MPV_Vs_PathlengthTEC      = ChargeToMPV(Charge_Vs_PathlengthTEC  ,"MPV_Vs_PathlengthTEC"  , true);
// TH1D* MPV_Vs_PathlengthTECP     = ChargeToMPV(Charge_Vs_PathlengthTECP ,"MPV_Vs_PathlengthTECP" , true);
// TH1D* MPV_Vs_PathlengthTECM     = ChargeToMPV(Charge_Vs_PathlengthTECM ,"MPV_Vs_PathlengthTECM" , true);
   TH1D* MPV_Vs_PathlengthTEC1     = ChargeToMPV(Charge_Vs_PathlengthTEC1 ,"MPV_Vs_PathlengthTEC1" , true);
   TH1D* MPV_Vs_PathlengthTEC2     = ChargeToMPV(Charge_Vs_PathlengthTEC2 ,"MPV_Vs_PathlengthTEC2" , true);


// TH1D* MPV_Vs_PathlengthTECP1    = ChargeToMPV(Charge_Vs_PathlengthTECP1,"MPV_Vs_PathlengthTECP1", true);
// TH1D* MPV_Vs_PathlengthTECP2    = ChargeToMPV(Charge_Vs_PathlengthTECP2,"MPV_Vs_PathlengthTECP2", true);
// TH1D* MPV_Vs_PathlengthTECM1    = ChargeToMPV(Charge_Vs_PathlengthTECM1,"MPV_Vs_PathlengthTECM1", true);
// TH1D* MPV_Vs_PathlengthTECM2    = ChargeToMPV(Charge_Vs_PathlengthTECM2,"MPV_Vs_PathlengthTECM2", true);
   TH1D* MPV_Vs_PathlengthThin     = ChargeToMPV(Charge_Vs_PathlengthThin ,"MPV_Vs_PathlengthThin" , true);
   TH1D* MPV_Vs_PathlengthThick    = ChargeToMPV(Charge_Vs_PathlengthThick,"MPV_Vs_PathlengthThick", true);



   TH2D* MPV_Vs_EtaTIB  = new TH2D("MPV_Vs_EtaTIB" ,"MPV_Vs_EtaTIB" , 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTID  = new TH2D("MPV_Vs_EtaTID" ,"MPV_Vs_EtaTID" , 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTOB  = new TH2D("MPV_Vs_EtaTOB" ,"MPV_Vs_EtaTOB" , 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTEC  = new TH2D("MPV_Vs_EtaTEC" ,"MPV_Vs_EtaTEC" , 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTEC1 = new TH2D("MPV_Vs_EtaTEC1","MPV_Vs_EtaTEC1", 50, -3.0, 3.0, 300, 0, 600);
   TH2D* MPV_Vs_EtaTEC2 = new TH2D("MPV_Vs_EtaTEC2","MPV_Vs_EtaTEC2", 50, -3.0, 3.0, 300, 0, 600);

   TH2D* MPV_Vs_PhiTIB  = new TH2D("MPV_Vs_PhiTIB" ,"MPV_Vs_PhiTIB" , 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTID  = new TH2D("MPV_Vs_PhiTID" ,"MPV_Vs_PhiTID" , 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTOB  = new TH2D("MPV_Vs_PhiTOB" ,"MPV_Vs_PhiTOB" , 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTEC  = new TH2D("MPV_Vs_PhiTEC" ,"MPV_Vs_PhiTEC" , 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTEC1 = new TH2D("MPV_Vs_PhiTEC1","MPV_Vs_PhiTEC1", 50, -3.4, 3.4, 300, 0, 600);
   TH2D* MPV_Vs_PhiTEC2 = new TH2D("MPV_Vs_PhiTEC2","MPV_Vs_PhiTEC2", 50, -3.4, 3.4, 300, 0, 600);

   TH2D* NoMPV          = new TH2D("NoMPV"         ,"NoMPV"         ,350, -350, 350, 240, 0, 120);



   TH1D* Gains          = new TH1D("Gains"         ,"Gains"         ,                300, 0, 2);
   TH1D* MPVs           = new TH1D("MPVs"          ,"MPVs"          ,                300, 0, 600);
   TH1D* MPVs320        = new TH1D("MPVs320"       ,"MPVs320"       ,                300, 0, 600);
   TH1D* MPVs500        = new TH1D("MPVs500"       ,"MPVs500"       ,                300, 0, 600);
   TH1D* MPVsTIB        = new TH1D("MPVsTIB"       ,"MPVsTIB"       ,                300, 0, 600);
   TH1D* MPVsTID        = new TH1D("MPVsTID"       ,"MPVsTID"       ,                300, 0, 600);
   TH1D* MPVsTIDP       = new TH1D("MPVsTIDP"      ,"MPVsTIDP"      ,                300, 0, 600);
   TH1D* MPVsTIDM       = new TH1D("MPVsTIDM"      ,"MPVsTIDM"      ,                300, 0, 600);
   TH1D* MPVsTOB        = new TH1D("MPVsTOB"       ,"MPVsTOB"       ,                300, 0, 600);
   TH1D* MPVsTEC        = new TH1D("MPVsTEC"       ,"MPVsTEC"       ,                300, 0, 600);
   TH1D* MPVsTECP       = new TH1D("MPVsTECP"      ,"MPVsTECP"      ,                300, 0, 600);
   TH1D* MPVsTECM       = new TH1D("MPVsTECM"      ,"MPVsTECM"      ,                300, 0, 600);
   TH1D* MPVsTEC1       = new TH1D("MPVsTEC1"      ,"MPVsTEC1"      ,                300, 0, 600);
   TH1D* MPVsTEC2       = new TH1D("MPVsTEC2"      ,"MPVsTEC2"      ,                300, 0, 600);
   TH1D* MPVsTECP1      = new TH1D("MPVsTECP1"     ,"MPVsTECP1"     ,                300, 0, 600);
   TH1D* MPVsTECP2      = new TH1D("MPVsTECP2"     ,"MPVsTECP2"     ,                300, 0, 600);
   TH1D* MPVsTECM1      = new TH1D("MPVsTECM1"     ,"MPVsTECM1"     ,                300, 0, 600);
   TH1D* MPVsTECM2      = new TH1D("MPVsTECM2"     ,"MPVsTECM2"     ,                300, 0, 600);


   TH1D* MPVError       = new TH1D("MPVError"      ,"MPVError"      ,                150, 0, 150);
   TH2D* MPVErrorVsMPV  = new TH2D("MPVErrorVsMPV" ,"MPVErrorVsMPV" ,300,    0, 600, 150, 0, 150);
   TH2D* MPVErrorVsEta  = new TH2D("MPVErrorVsEta" ,"MPVErrorVsEta" , 50, -3.0, 3.0, 150, 0, 150); 
   TH2D* MPVErrorVsPhi  = new TH2D("MPVErrorVsPhi" ,"MPVErrorVsPhi" , 50, -3.4, 3.4, 150, 0, 150);             
   TH2D* MPVErrorVsN    = new TH2D("MPVErrorVsN"   ,"MPVErrorVsN"   ,500,    0,1000, 150, 0, 150);              




   TH1D* ChargeTIB      = new TH1D("ChargeTIB"     ,"ChargeTIB"     ,               1000, 0,2000);
   TH1D* ChargeTID      = new TH1D("ChargeTID"     ,"ChargeTID"     ,               1000, 0,2000);
   TH1D* ChargeTIDP     = new TH1D("ChargeTIDP"    ,"ChargeTIDP"    ,               1000, 0,2000);
   TH1D* ChargeTIDM     = new TH1D("ChargeTIDM"    ,"ChargeTIDM"    ,               1000, 0,2000);
   TH1D* ChargeTOB      = new TH1D("ChargeTOB"     ,"ChargeTOB"     ,               1000, 0,2000);
   TH1D* ChargeTEC      = new TH1D("ChargeTEC"     ,"ChargeTEC"     ,               1000, 0,2000);
   TH1D* ChargeTECP     = new TH1D("ChargeTECP"    ,"ChargeTECP"    ,               1000, 0,2000);
   TH1D* ChargeTECM     = new TH1D("ChargeTECM"    ,"ChargeTECM"    ,               1000, 0,2000);
   TH1D* ChargeTEC1     = new TH1D("ChargeTEC1"    ,"ChargeTEC1"    ,               1000, 0,2000);
   TH1D* ChargeTEC2     = new TH1D("ChargeTEC2"    ,"ChargeTEC2"    ,               1000, 0,2000);
   TH1D* ChargeTECP1    = new TH1D("ChargeTECP1"   ,"ChargeTECP1"   ,               1000, 0,2000);
   TH1D* ChargeTECP2    = new TH1D("ChargeTECP2"   ,"ChargeTECP2"   ,               1000, 0,2000);
   TH1D* ChargeTECM1    = new TH1D("ChargeTECM1"   ,"ChargeTECM1"   ,               1000, 0,2000);
   TH1D* ChargeTECM2    = new TH1D("ChargeTECM2"   ,"ChargeTECM2"   ,               1000, 0,2000);

   TH1D* ChargeAbsTIB   = new TH1D("ChargeAbsTIB"  ,"ChargeAbsTIB"  ,                500, 0,2000);
   TH1D* ChargeAbsTID   = new TH1D("ChargeAbsTID"  ,"ChargeAbsTID"  ,                500, 0,2000);
   TH1D* ChargeAbsTIDP  = new TH1D("ChargeAbsTIDP" ,"ChargeAbsTIDP" ,                500, 0,2000);
   TH1D* ChargeAbsTIDM  = new TH1D("ChargeAbsTIDM" ,"ChargeAbsTIDM" ,                500, 0,2000);
   TH1D* ChargeAbsTOB   = new TH1D("ChargeAbsTOB"  ,"ChargeAbsTOB"  ,                500, 0,2000);
   TH1D* ChargeAbsTEC   = new TH1D("ChargeAbsTEC"  ,"ChargeAbsTEC"  ,                500, 0,2000);
   TH1D* ChargeAbsTECP  = new TH1D("ChargeAbsTECP" ,"ChargeAbsTECP" ,                500, 0,2000);
   TH1D* ChargeAbsTECM  = new TH1D("ChargeAbsTECM" ,"ChargeAbsTECM" ,                500, 0,2000);
   TH1D* ChargeAbsTEC1  = new TH1D("ChargeAbsTEC1" ,"ChargeAbsTEC1" ,                500, 0,2000);
   TH1D* ChargeAbsTEC2  = new TH1D("ChargeAbsTEC2" ,"ChargeAbsTEC2" ,                500, 0,2000);
   TH1D* ChargeAbsTECP1 = new TH1D("ChargeAbsTECP1","ChargeAbsTECP1",                500, 0,2000);
   TH1D* ChargeAbsTECP2 = new TH1D("ChargeAbsTECP2","ChargeAbsTECP2",                500, 0,2000);
   TH1D* ChargeAbsTECM1 = new TH1D("ChargeAbsTECM1","ChargeAbsTECM1",                500, 0,2000);
   TH1D* ChargeAbsTECM2 = new TH1D("ChargeAbsTECM2","ChargeAbsTECM2",                500, 0,2000);

   TH1D* DiffWRTPrevGainTIB      = new TH1D("DiffWRTPrevGainTIB"     ,"DiffWRTPrevGainTIB"     ,               250, 0,2);
   TH1D* DiffWRTPrevGainTID      = new TH1D("DiffWRTPrevGainTID"     ,"DiffWRTPrevGainTID"     ,               250, 0,2);
   TH1D* DiffWRTPrevGainTOB      = new TH1D("DiffWRTPrevGainTOB"     ,"DiffWRTPrevGainTOB"     ,               250, 0,2);
   TH1D* DiffWRTPrevGainTEC      = new TH1D("DiffWRTPrevGainTEC"     ,"DiffWRTPrevGainTEC"     ,               250, 0,2);

   TH2D* GainVsPrevGainTIB      = new TH2D("GainVsPrevGainTIB"     ,"GainVsPrevGainTIB"     ,               100, 0,2, 100, 0,2);
   TH2D* GainVsPrevGainTID      = new TH2D("GainVsPrevGainTID"     ,"GainVsPrevGainTID"     ,               100, 0,2, 100, 0,2);
   TH2D* GainVsPrevGainTOB      = new TH2D("GainVsPrevGainTOB"     ,"GainVsPrevGainTOB"     ,               100, 0,2, 100, 0,2);
   TH2D* GainVsPrevGainTEC      = new TH2D("GainVsPrevGainTEC"     ,"GainVsPrevGainTEC"     ,               100, 0,2, 100, 0,2);


   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Looping on the Tree          :");
   int TreeStep = t1->GetEntries()/50;if(TreeStep==0)TreeStep=1;
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      int bin = ChargeDistrib->GetXaxis()->FindBin(tree_Index);
      TH1D* Proj         = ChargeDistrib ->ProjectionY("proj" ,bin, bin);
      TH1D* ProjAbsolute = ChargeDistribA->ProjectionY("projA",bin, bin);

      if(tree_FitMPV<0                        ) NoMPV         ->Fill(tree_z ,tree_R);
      if(tree_FitMPV>=0){

      if(tree_SubDet==3                       ) MPV_Vs_EtaTIB ->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==4                       ) MPV_Vs_EtaTID ->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==5                       ) MPV_Vs_EtaTOB ->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==6                       ) MPV_Vs_EtaTEC ->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04) MPV_Vs_EtaTEC1->Fill(tree_Eta,tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04) MPV_Vs_EtaTEC2->Fill(tree_Eta,tree_FitMPV);

      if(tree_SubDet==3                       ) MPV_Vs_PhiTIB ->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==4                       ) MPV_Vs_PhiTID ->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==5                       ) MPV_Vs_PhiTOB ->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==6                       ) MPV_Vs_PhiTEC ->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04) MPV_Vs_PhiTEC1->Fill(tree_Phi,tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04) MPV_Vs_PhiTEC2->Fill(tree_Phi,tree_FitMPV);

      if(tree_FitMPV>0                        ) Gains         ->Fill(         tree_Gain  );
                                                MPVs          ->Fill(         tree_FitMPV);
      if(                  tree_Thickness<0.04) MPVs320       ->Fill(         tree_FitMPV);
      if(                  tree_Thickness>0.04) MPVs500       ->Fill(         tree_FitMPV);
      if(tree_SubDet==3                       ) MPVsTIB       ->Fill(         tree_FitMPV);
      if(tree_SubDet==4                       ) MPVsTID       ->Fill(         tree_FitMPV);
      if(tree_SubDet==4 && tree_Eta<0         ) MPVsTIDM      ->Fill(         tree_FitMPV);
      if(tree_SubDet==4 && tree_Eta>0         ) MPVsTIDP      ->Fill(         tree_FitMPV);
      if(tree_SubDet==5                       ) MPVsTOB       ->Fill(         tree_FitMPV);
      if(tree_SubDet==6                       ) MPVsTEC       ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04) MPVsTEC1      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04) MPVsTEC2      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Eta<0         ) MPVsTECP      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Eta>0         ) MPVsTECM      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04 && tree_Eta>0) MPVsTECP1      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04 && tree_Eta>0) MPVsTECP2      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness<0.04 && tree_Eta<0) MPVsTECM1      ->Fill(         tree_FitMPV);
      if(tree_SubDet==6 && tree_Thickness>0.04 && tree_Eta<0) MPVsTECM2      ->Fill(         tree_FitMPV);

                                                MPVError      ->Fill(         tree_FitMPVErr);    
                                                MPVErrorVsMPV ->Fill(tree_FitMPV,tree_FitMPVErr);
                                                MPVErrorVsEta ->Fill(tree_Eta,tree_FitMPVErr);
                                                MPVErrorVsPhi ->Fill(tree_Phi,tree_FitMPVErr);
                                                MPVErrorVsN   ->Fill(tree_NEntries,tree_FitMPVErr);
      }


      if(tree_SubDet==3                       ) ChargeTIB  ->Add(Proj,1);
      if(tree_SubDet==4                       ) ChargeTID  ->Add(Proj,1);
      if(tree_SubDet==4 && tree_Eta<0         ) ChargeTIDM ->Add(Proj,1);
      if(tree_SubDet==4 && tree_Eta>0         ) ChargeTIDP ->Add(Proj,1);
      if(tree_SubDet==5                       ) ChargeTOB  ->Add(Proj,1);
      if(tree_SubDet==6                       ) ChargeTEC  ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Thickness<0.04) ChargeTEC1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Thickness>0.04) ChargeTEC2 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0         ) ChargeTECP ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0         ) ChargeTECM ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness<0.04) ChargeTECM1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness>0.04) ChargeTECM2 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness<0.04) ChargeTECP1 ->Add(Proj,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness>0.04) ChargeTECP2 ->Add(Proj,1);


      if(tree_SubDet==3                       ) ChargeAbsTIB  ->Add(ProjAbsolute,1);
      if(tree_SubDet==4                       ) ChargeAbsTID  ->Add(ProjAbsolute,1);
      if(tree_SubDet==4 && tree_Eta<0         ) ChargeAbsTIDM ->Add(ProjAbsolute,1);
      if(tree_SubDet==4 && tree_Eta>0         ) ChargeAbsTIDP ->Add(ProjAbsolute,1);
      if(tree_SubDet==5                       ) ChargeAbsTOB  ->Add(ProjAbsolute,1);
      if(tree_SubDet==6                       ) ChargeAbsTEC  ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Thickness<0.04) ChargeAbsTEC1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Thickness>0.04) ChargeAbsTEC2 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0         ) ChargeAbsTECP ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0         ) ChargeAbsTECM ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness<0.04) ChargeAbsTECM1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta<0 && tree_Thickness>0.04) ChargeAbsTECM2 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness<0.04) ChargeAbsTECP1 ->Add(ProjAbsolute,1);
      if(tree_SubDet==6 && tree_Eta>0 && tree_Thickness>0.04) ChargeAbsTECP2 ->Add(ProjAbsolute,1);

      if(tree_SubDet==3                       ) DiffWRTPrevGainTIB  ->Fill(tree_Gain/tree_PrevGain);
      if(tree_SubDet==4                       ) DiffWRTPrevGainTID  ->Fill(tree_Gain/tree_PrevGain);
      if(tree_SubDet==5                       ) DiffWRTPrevGainTOB  ->Fill(tree_Gain/tree_PrevGain);
      if(tree_SubDet==6                       ) DiffWRTPrevGainTEC  ->Fill(tree_Gain/tree_PrevGain);

      if(tree_SubDet==3                       ) GainVsPrevGainTIB  ->Fill(tree_PrevGain,tree_Gain);
      if(tree_SubDet==4                       ) GainVsPrevGainTID  ->Fill(tree_PrevGain,tree_Gain);
      if(tree_SubDet==5                       ) GainVsPrevGainTOB  ->Fill(tree_PrevGain,tree_Gain);
      if(tree_SubDet==6                       ) GainVsPrevGainTEC  ->Fill(tree_PrevGain,tree_Gain);

      delete Proj;
      delete ProjAbsolute;
   }printf("\n");



   // ######################################################### PRINT OUT APV INFOS #################################
   unsigned int CountAPV_Total    = 0;
   unsigned int CountAPV_NoEntry  = 0;
   unsigned int CountAPV_NoEntryU = 0;
   unsigned int CountAPV_NoGain   = 0;
   unsigned int CountAPV_NoGainU  = 0;
   unsigned int CountAPV_LowGain  = 0;
   unsigned int CountAPV_DiffGain = 0;

   pFile = fopen((output + "_LowResponseModule.txt").c_str(),"w");
   fprintf(pFile,"\n\nALL APVs WITH NO ENTRIES (NO RECO CLUSTER ON IT)\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {      
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      CountAPV_Total++;
      if(tree_NEntries==0){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); CountAPV_NoEntry++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_NoEntry,CountAPV_Total,(100.0*CountAPV_NoEntry)/CountAPV_Total);

   fprintf(pFile,"\n\nUNMASKED APVs WITH NO ENTRIES (NO RECO CLUSTER ON IT)\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      if(tree_NEntries==0 && !tree_isMasked){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); CountAPV_NoEntryU++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_NoEntryU,CountAPV_Total,(100.0*CountAPV_NoEntryU)/CountAPV_Total);

   fprintf(pFile,"\n\nALL APVs WITH NO GAIN COMPUTED\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      if(tree_FitMPV<0){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); CountAPV_NoGain++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_NoGain,CountAPV_Total,(100.0*CountAPV_NoGain)/CountAPV_Total);

   fprintf(pFile,"\n\nUNMASKED APVs WITH NO GAIN COMPUTED\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      if(tree_FitMPV<0 && !tree_isMasked){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); CountAPV_NoGainU++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_NoGainU,CountAPV_Total,(100.0*CountAPV_NoGainU)/CountAPV_Total);

   fprintf(pFile,"\n\nUNMASKED APVs WITH LOW RESPONSE\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      if(tree_FitMPV>0 && tree_FitMPV<220 && !tree_isMasked){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); CountAPV_LowGain++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_LowGain,CountAPV_Total,(100.0*CountAPV_LowGain)/CountAPV_Total);

   fprintf(pFile,"\n\nUNMASKED APVs WITH SIGNIFICANT CHANGE OF GAIN VALUE\n--------------------------------------------\n");
   printf("Looping on the Tree          :");
   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
      t1->GetEntry(ientry);
      if(tree_FitMPV>0 && !tree_isMasked && (tree_Gain/tree_PrevGain<0.7 || tree_Gain/tree_PrevGain>1.3)){fprintf(pFile,"%i-%i, ",tree_DetId,tree_APVId); CountAPV_DiffGain++;}
   }printf("\n");
   fprintf(pFile,"\n--> %i / %i = %f%% APV Concerned\n",CountAPV_DiffGain,CountAPV_Total,(100.0*CountAPV_DiffGain)/CountAPV_Total);
   fclose(pFile);



   // ######################################################### PRINT DISTRIBUTION INFO #################################
   pFile = fopen((output + "_SubDetector_MPV.txt").c_str(),"w");

   double Results[5]; TF1* landau;
   landau = getLandau(ChargeTIB, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTIB->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   SaveCanvas(c1,output,"SubDetChargeTIB");
   fprintf(pFile,"TIB   MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTIDM, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTIDM->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   SaveCanvas(c1,output,"SubDetChargeTIDM");
   fprintf(pFile, "TIDM  MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTIDP, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTIDP->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   SaveCanvas(c1,output,"SubDetChargeTIDP");
   fprintf(pFile, "TIDP  MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTOB, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTOB->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   SaveCanvas(c1,output,"SubDetChargeTOB");
   fprintf(pFile, "TOB   MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTECP1, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTECP1->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   SaveCanvas(c1,output,"SubDetChargeTECP1");
   fprintf(pFile, "TECP1 MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTECP2, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTECP2->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   SaveCanvas(c1,output,"SubDetChargeTECP2");
   fprintf(pFile, "TECP2 MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTECM1, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTECM1->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   SaveCanvas(c1,output,"SubDetChargeTECM1");
   fprintf(pFile, "TECM1 MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   landau = getLandau(ChargeTECM2, Results, 0, 5400);
   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   ChargeTECM2->Draw();
   landau->SetLineWidth(3);
   landau->Draw("same");
   SaveCanvas(c1,output,"SubDetChargeTECM2");
   fprintf(pFile, "TECM2 MPV=%7.2f +- %7.2f  Chi2=%7.2f\n",Results[0],Results[1],Results[4]);

   fclose(pFile);
   // ######################################################### MAKE PLOTS #################################


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = MPV_Vs_EtaTEC;                     legend.push_back("TEC");
   Histos[1] = MPV_Vs_EtaTIB;                     legend.push_back("TIB");
   Histos[2] = MPV_Vs_EtaTID;                     legend.push_back("TID");
   Histos[3] = MPV_Vs_EtaTOB;                     legend.push_back("TOB");
   DrawTH2D((TH2D**)Histos,legend, "", "module #eta", "MPV [ADC/mm]", -3.0,3.0, 0,600);
   DrawLegend (Histos,legend,"","P");
   DrawStatBox(Histos,legend,false);
   SaveCanvas(c1,output,"MPV_Vs_EtaSubDet");
   delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PhiTEC;                     legend.push_back("TEC");
    Histos[1] = MPV_Vs_PhiTIB;                     legend.push_back("TIB");
    Histos[2] = MPV_Vs_PhiTID;                     legend.push_back("TID");
    Histos[3] = MPV_Vs_PhiTOB;                     legend.push_back("TOB");
    DrawTH2D((TH2D**)Histos,legend, "", "module #phi", "MPV [ADC/mm]", -3.4,3.4, 0,600);
    DrawLegend(Histos,legend,"","P");
    DrawStatBox(Histos,legend,false);
    SaveCanvas(c1,output,"MPV_Vs_PhiSubDet");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = NoMPV;                             legend.push_back("NoMPV");
    DrawTH2D((TH2D**)Histos,legend, "", "z (cm)", "R (cms)", 0,0, 0,0);
    SaveCanvas(c1,output,"NoMPV", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    ChargeDistrib->GetXaxis()->SetNdivisions(5+500);
    Histos[0] = ChargeDistrib;                     legend.push_back("Charge Vs Index");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "APV Index", "Charge [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,output,"Charge2D", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = Gains;                             legend.push_back("");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Gain", "Number of APVs", 0.5,1.5, 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true);
    SaveCanvas(c1,output,"Gains");

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVs320;                           legend.push_back("320 #mum");
    Histos[1] = MPVs500;                           legend.push_back("500 #mum");
    Histos[2] = MPVs;                              legend.push_back("320 + 500 #mum");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "MPV [ADC/mm]", "Number of APVs", 100,550, 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(DataType);
    SaveCanvas(c1,output,"MPVs");
    c1->SetLogy(true);
    SaveCanvas(c1,output,"MPVsLog");
    DrawStatBox(Histos,legend,true);
    c1->SetLogy(false);
    SaveCanvas(c1,output,"MPVsAndStat");
    c1->SetLogy(true);
    SaveCanvas(c1,output,"MPVsLogAndStat");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    MPVsTOB->GetXaxis()->SetNdivisions(5+500);
    Histos[0] = MPVsTIB;                           legend.push_back("TIB (320 #mum)");
    Histos[1] = MPVsTID;                           legend.push_back("TID (320 #mum)");
    Histos[2] = MPVsTOB;                           legend.push_back("TOB (500 #mum)");
    Histos[3] = MPVsTEC1;                          legend.push_back("TEC (320 #mum)");
    Histos[4] = MPVsTEC2;                          legend.push_back("TEC (500 #mum)");
//    Histos[5] = MPVs;                              legend.push_back("All");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "MPV [ADC/mm]", "Number of APVs", 100,550, 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(DataType);
    SaveCanvas(c1,output,"MPVsSubDet");
    c1->SetLogy(true);
    SaveCanvas(c1,output,"MPVsSubDetLog");
    DrawStatBox(Histos,legend,true);
    SaveCanvas(c1,output,"MPVsSubDetLogAndStat");
    c1->SetLogy(false);
    SaveCanvas(c1,output,"MPVsSubDetAndStat");
    delete c1;



    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    MPVsTOB->GetXaxis()->SetNdivisions(5+500);
    Histos[0] = MPVsTECP1;                          legend.push_back("TEC+ (320 #mum)");
    Histos[1] = MPVsTECP2;                          legend.push_back("TEC+ (500 #mum)");
    Histos[2] = MPVsTECM1;                          legend.push_back("TEC- (320 #mum)");
    Histos[3] = MPVsTECM2;                          legend.push_back("TEC- (500 #mum)");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "MPV [ADC/mm]", "Number of APVs", 100,550, 0,0);
    DrawLegend(Histos,legend,"#sqrt{s}=7TeV","L");
    DrawStatBox(Histos,legend,true);
    SaveCanvas(c1,output,"MPVsTEC");
    c1->SetLogy(true);
    SaveCanvas(c1,output,"MPVsTECLog");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVError;                          legend.push_back("MPV Error");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Error on MPV [ADC/mm]", "Number of APVs", 0,500, 0,0);
    DrawStatBox(Histos,legend,true);
    c1->SetLogy(true);
    SaveCanvas(c1,output,"Error");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsMPV;                     legend.push_back("Error Vs MPV");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "MPV [ADC/mm]", "Error on MPV [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,output,"Error_Vs_MPV", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsEta;                     legend.push_back("Error Vs Eta");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "module #eta", "Error on MPV [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,output,"Error_Vs_Eta", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsPhi;                     legend.push_back("Error Vs Phi");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "module #phi", "Error on MPV [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,output,"Error_Vs_Phi", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPVErrorVsN;                       legend.push_back("Error Vs Entries");
    DrawTH2D((TH2D**)Histos,legend, "COLZ", "Number of Entries", "Error on MPV [ADC/mm]", 0,0, 0,0);
    c1->SetLogz(true);
    SaveCanvas(c1,output,"Error_Vs_N", true);
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTEC;                         legend.push_back("TEC");
    Histos[1] = ChargeTIB;                         legend.push_back("TIB");
    Histos[2] = ChargeTID;                         legend.push_back("TID");
    Histos[3] = ChargeTOB;                         legend.push_back("TOB");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawPreliminary(DataType);
    TLine* l0 = new TLine(300, 0,300,((TH1*)Histos[0])->GetMaximum()); l0->SetLineWidth(3); l0->SetLineStyle(2); l0->Draw("same");
    SaveCanvas(c1,output,"Charge");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeAndStat");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTECP;                        legend.push_back("TEC+");
    Histos[1] = ChargeTECM;                        legend.push_back("TEC-");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeTECSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTEC1;                        legend.push_back("TEC Thin");
    Histos[1] = ChargeTEC2;                        legend.push_back("TEC Thick");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeTECThickness");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeTIDP;                        legend.push_back("TID+");
    Histos[1] = ChargeTIDM;                        legend.push_back("TID-");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC/mm]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeTIDSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTEC;                      legend.push_back("TEC");
    Histos[1] = ChargeAbsTIB;                      legend.push_back("TIB");
    Histos[2] = ChargeAbsTID;                      legend.push_back("TID");
    Histos[3] = ChargeAbsTOB;                      legend.push_back("TOB");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeAbs");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTECP;                     legend.push_back("TEC+");
    Histos[1] = ChargeAbsTECM;                     legend.push_back("TEC-");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeAbsTECSide");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTEC1;                     legend.push_back("TEC Thin");
    Histos[1] = ChargeAbsTEC2;                     legend.push_back("TEC Thick");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeAbsTECThickness");
    delete c1;

    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = ChargeAbsTIDP;                     legend.push_back("TID+");
    Histos[1] = ChargeAbsTIDM;                     legend.push_back("TID-");
    DrawSuperposedHistos((TH1**)Histos, legend, "",  "Charge [ADC]", "Number of Clusters", 0,800 , 0,0);
    DrawLegend(Histos,legend,"","L");
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"ChargeAbsTIDSide");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PathlengthThin;             legend.push_back("320 #mum");
    Histos[1] = MPV_Vs_PathlengthThick;            legend.push_back("500 #mum");
    DrawSuperposedHistos((TH1**)Histos, legend, "HIST",  "pathlength [mm]", "MPV [ADC/mm]", 0,0 , 230,380);
    DrawLegend(Histos,legend,"","L");
    SaveCanvas(c1,output,"MPV_Vs_Path");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = MPV_Vs_PathlengthTIB;              legend.push_back("TIB (320 #mum)");
    Histos[1] = MPV_Vs_PathlengthTID;              legend.push_back("TID (320 #mum)");
    Histos[2] = MPV_Vs_PathlengthTOB;              legend.push_back("TOB (500 #mum)");
    Histos[3] = MPV_Vs_PathlengthTEC1;             legend.push_back("TEC (320 #mum)");
    Histos[4] = MPV_Vs_PathlengthTEC2;             legend.push_back("TEC (500 #mum)");
    DrawSuperposedHistos((TH1**)Histos, legend, "HIST",  "pathlength [mm]", "MPV [ADC/mm]", 0,0 , 230,380);
    DrawLegend(Histos,legend,"","L");
    SaveCanvas(c1,output,"MPV_Vs_PathSubDet");
    delete c1;


    c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
    Histos[0] = DiffWRTPrevGainTIB;                legend.push_back("TIB");
    Histos[1] = DiffWRTPrevGainTID;                legend.push_back("TID");
    Histos[2] = DiffWRTPrevGainTOB;                legend.push_back("TOB");
    Histos[3] = DiffWRTPrevGainTEC;                legend.push_back("TEC");
    DrawSuperposedHistos((TH1**)Histos, legend, "HIST",  "New Gain / Previous Gain", "Number of APV", 0.0,2.0 ,0,0);
    DrawLegend(Histos,legend,"","L");
    c1->SetLogy(true);
    DrawStatBox(Histos,legend,true, 0.6, 0.7);
    SaveCanvas(c1,output,"GainDividedPrevGain");
    delete c1;


   c1 = new TCanvas("c1","c1,",600,600);          legend.clear();
   Histos[0] = GainVsPrevGainTEC;                 legend.push_back("TEC");
   Histos[1] = GainVsPrevGainTIB;                 legend.push_back("TIB");
   Histos[2] = GainVsPrevGainTID;                 legend.push_back("TID");
   Histos[3] = GainVsPrevGainTOB;                 legend.push_back("TOB");
   DrawTH2D((TH2D**)Histos,legend, "", "Previous Gain", "New Gain", 0.5,1.8, 0.5,1.8);
   TLine diagonal(0.5,0.5,1.8,1.8);
   diagonal.SetLineWidth(3);
   diagonal.SetLineStyle(2);
   diagonal.Draw("same");
   DrawLegend (Histos,legend,"","P");
   DrawStatBox(Histos,legend,false);
   SaveCanvas(c1,output,"GainVsPrevGain");
   delete c1;


}

TF1* getLandau(TH1* InputHisto, double* FitResults, double LowRange, double HighRange)
{
   FitResults[0]         = -0.5;  //MPV
   FitResults[1]         =  0;    //MPV error
   FitResults[2]         = -0.5;  //Width
   FitResults[3]         =  0;    //Width error
   FitResults[4]         = -0.5;  //Fit Chi2/NDF

   // perform fit with standard landau
   TF1* MyLandau = new TF1("MyLandau","landau",LowRange, HighRange);
   MyLandau->SetParameter(1,300);
   InputHisto->Fit("MyLandau","0QR WW");

   // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
   FitResults[0]         = MyLandau->GetParameter(1);  //MPV
   FitResults[1]         = MyLandau->GetParError(1);   //MPV error
   FitResults[2]         = MyLandau->GetParameter(2);  //Width
   FitResults[3]         = MyLandau->GetParError(2);   //Width error
   FitResults[4]         = MyLandau->GetChisquare() / MyLandau->GetNDF();  //Fit Chi2/NDF

   return MyLandau;
}

TH1D* ChargeToMPV(TH2* InputHisto, string Name,  bool DivideByX)
{
   TH1D* toReturn = new TH1D(Name.c_str(),Name.c_str(),InputHisto->GetXaxis()->GetNbins(), InputHisto->GetXaxis()->GetXmin(), InputHisto->GetXaxis()->GetXmax() );
   double Results[5];

   for(int i=0;i<=InputHisto->GetXaxis()->GetNbins();i++){
      TH1D* proj   = InputHisto->ProjectionY("",i,i);
      if(proj->GetEntries()<50){delete proj;continue;}

      TF1*  landau = getLandau(proj,Results);

      if(DivideByX){
         toReturn->SetBinContent(i,Results[0] / InputHisto->GetXaxis()->GetBinCenter(i) );
         toReturn->SetBinError  (i,Results[1] / InputHisto->GetXaxis()->GetBinCenter(i) );
      }else{
         toReturn->SetBinContent(i,Results[0]);
         toReturn->SetBinError  (i,Results[1]);
      }
      delete landau;
     delete proj;
   }

   return toReturn;
}


bool LoadLumiToRun()
{
   float TotalIntLuminosity = 0;

   FILE* pFile = fopen("lumi.txt","r");
   if(!pFile){
      printf("Not Found: %s\n","out.txt");
      return false;
   }

   unsigned int Run; float IntLumi;
   unsigned int DeliveredLs; double DeliveredLumi;
   char Line[2048], Tmp1[2048], Tmp2[2048], Tmp3[2048];
   while ( ! feof (pFile) ){
     fscanf(pFile,"%s\n",Line);
     //printf("%s\n",Line);
     for(unsigned int i=0;Line[i]!='\0';i++){if(Line[i]==',')Line[i]=' ';} 
     sscanf(Line,"%d:%d %s %s %s %f\n",&Run,&Tmp1, Tmp1,Tmp2,Tmp3,&IntLumi);
     TotalIntLuminosity+= IntLumi/1000000000.0;
//     printf("%6i --> %f/fb  (%f)  (%s | %s | %s)\n",Run,IntLumi,TotalIntLuminosity, Tmp1,Tmp2,Tmp3);
     RunToIntLumi[Run] = TotalIntLuminosity;
   }
   fclose(pFile);
   return true;
}

double getLumiFromRun(unsigned int run){
   return run; //simply return the run number for now on
   int closestRun = 0;  int lastRun=0;
   for(std::map<unsigned int, double>::iterator it = RunToIntLumi.begin(); it!=RunToIntLumi.end();it++){
      if(it->first>run && abs(it->first-run)<abs(closestRun-run))closestRun = it->first;
      if(it->first>lastRun)lastRun=it->first;;
   }
   if(closestRun==0 && run>RunToIntLumi.begin()->first )closestRun=lastRun;
   return RunToIntLumi[closestRun];
}



