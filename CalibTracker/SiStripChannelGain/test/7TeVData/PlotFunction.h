
int Color [] = {4,2,1,3,9,6,7,8,5};
int Marker[] = {20,22,21,23,29,3,2};
int Style [] = {1,2,5,7,9,10};


TObject* GetObjectFromPath(TDirectory* File, string Path)
{
   size_t pos = Path.find("/");
   if(pos < 256){
      string firstPart = Path.substr(0,pos);
      string endPart   = Path.substr(pos+1,Path.length());
      TDirectory* TMP = (TDirectory*)File->Get(firstPart.c_str());
      if(TMP!=NULL)return GetObjectFromPath(TMP,endPart);

      printf("BUG: %s\n",Path.c_str());
      return NULL;
   }else{
      return File->Get(Path.c_str());
   }
   
}

void SaveCanvas(TCanvas* c, string path, string name, bool OnlyPPNG=false){
   string filepath;
   filepath = path + "_" + name + ".png"; c->SaveAs(filepath.c_str()); if(OnlyPPNG)return;
   filepath = path + "_" + name + ".eps"; c->SaveAs(filepath.c_str());
   filepath = path + "_" + name + ".C"  ; c->SaveAs(filepath.c_str());
}

void DrawPreliminary(int Type, double X=0.28, double Y=0.98, double W=0.85, double H=0.95){
   TPaveText* T = new TPaveText(X,Y,W,H, "NDC");
   T->SetFillColor(0);
   T->SetTextAlign(11);
   if(Type==0)T->AddText("CMS Preliminary 2010 :   #sqrt{s} = 7 TeV");
   if(Type==1)T->AddText("CMS Preliminary 2010 : MC with   #sqrt{s} = 7 TeV");
   if(Type==2)T->AddText("CMS Preliminary 2010 : Data with   #sqrt{s} = 7 TeV");
   T->Draw("same");
}

void DrawLegend (TObject** Histos, std::vector<string> legend, string Title, string Style, double X=0.79, double Y=0.93, double W=0.20, double H=0.05)
{
   int    N             = legend.size();
   
   if(legend[0]!=""){
      TLegend* leg;
      leg = new TLegend(X,Y,X-W,Y - N*H);
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      //leg->SetTextAlign(32);
      if(Title!="")leg->SetHeader(Title.c_str());

      if(Style=="DataMC"){
         for(int i=0;i<N;i++){
            TH2D* temp = (TH2D*)Histos[i]->Clone();
            temp->SetMarkerSize(1.3);
            if(i==0){
               leg->AddEntry(temp, legend[i].c_str() ,"P");
            }else{
               leg->AddEntry(temp, legend[i].c_str() ,"L");
            }
         }
      }else{
         for(int i=0;i<N;i++){
            TH2D* temp = (TH2D*)Histos[i]->Clone();
            temp->SetMarkerSize(1.3);
            leg->AddEntry(temp, legend[i].c_str() ,Style.c_str());
         }
      }
      leg->Draw();
   }
} 


void DrawStatBox(TObject** Histos, std::vector<string> legend, bool Mean, double X=0.15, double Y=0.93, double W=0.15, double H=0.03)
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



void DrawTH2D(TH2D** Histos, std::vector<string> legend, string Style, string Xlegend, string Ylegend, double xmin, double xmax, double ymin, double ymax)
{
   int    N             = legend.size();
   
   for(int i=0;i<N;i++){
        if(!Histos[i])continue;
        Histos[i]->SetTitle("");
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend.c_str());
        Histos[i]->GetYaxis()->SetTitle(Ylegend.c_str());
        Histos[i]->GetYaxis()->SetTitleOffset(1.60);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(0.3);
   }

   char Buffer[256];
   Histos[0]->Draw(Style.c_str());
   for(int i=1;i<N;i++){
        sprintf(Buffer,"%s same",Style.c_str());
        Histos[i]->Draw(Buffer);
   }
}


void DrawSuperposedHistos(TH1** Histos, std::vector<string> legend, string Style,  string Xlegend, string Ylegend, double xmin, double xmax, double ymin, double ymax, bool Normalize=false)
{
   int    N             = legend.size();

   double HistoMax      = -1;
   int    HistoHeighest = -1;

   for(int i=0;i<N;i++){
        if(!Histos[i])continue;
        if(Normalize && Histos[i]->Integral()!=0)Histos[i]->Scale(1.0/Histos[i]->Integral());
        Histos[i]->SetTitle("");
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend.c_str());
        Histos[i]->GetYaxis()->SetTitle(Ylegend.c_str());
        Histos[i]->GetYaxis()->SetTitleOffset(1.70);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetFillColor(0);
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(1.0);
        Histos[i]->SetLineColor(Color[i]);
        Histos[i]->SetLineWidth(2);
       if(Style=="DataMC" && i==0){
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
   if(Style=="DataMC"){
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
      Histos[HistoHeighest]->Draw(Style.c_str());
      for(int i=0;i<N;i++){
           if(Style!=""){
              sprintf(Buffer,"same %s",Style.c_str());
           }else{
              sprintf(Buffer,"same");
           }
           Histos[i]->Draw(Buffer);
      }
   }
}




