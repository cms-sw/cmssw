#include "TEveElement.h"
#include "TEveGeoNode.h"
#include "TEveManager.h"
#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"
#include "TEveBoxSet.h"
#include "TEveTrackPropagator.h"
#include "TGLViewer.h"
#include "TGLUtil.h"
#include <TH2.h>
#include <TBox.h>
#include <TLine.h>
#include <TLatex.h>
#include <TPaveText.h>
#include <TCanvas.h>

#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Fireworks/Core/interface/FWDetailView.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWDetailView.h"

using reco::Track;
using reco::TrackBase;
using reco::HitPattern;
using reco::TrackResiduals;

class FWTrackDetailView : public FWDetailView<reco::Track> {

public:
   FWTrackDetailView();
   virtual ~FWTrackDetailView();

   virtual TEveElement* build (const FWModelId &id,const reco::Track*);

protected:
   void getCenter( Double_t* vars )
   {
      vars[0] = rotationCenter()[0];
      vars[1] = rotationCenter()[1];
      vars[2] = rotationCenter()[2];
   }

private:
   FWTrackDetailView(const FWTrackDetailView&); // stop default
   const FWTrackDetailView& operator=(const FWTrackDetailView&); // stop default

   // ---------- member data --------------------------------
   void resetCenter() {
      rotationCenter()[0] = 0;
      rotationCenter()[1] = 0;
      rotationCenter()[2] = 0;
   }
};


FWTrackDetailView::FWTrackDetailView ()
{
   setUseGL(kFALSE);
}

FWTrackDetailView::~FWTrackDetailView ()
{
}

TEveElement* FWTrackDetailView::build (const FWModelId &id, const reco::Track* track)
{
   int nhits;
   float res[2][64];
   int hittype[64];
   int stereo[64];
   int substruct[64];
   int subsubstruct[64];
   int detector[64];

   const static char* det_tracker_str[6]={"PXB","PXF","TIB","TID","TOB","TEC"};
   HitPattern hitpat = track->hitPattern();
   TrackResiduals residuals = track->residuals();

   int numHits=hitpat.numberOfHits();
   for (int i = 0; i < numHits; ++i) {
      hittype[i] = 0x3 & hitpat.getHitPattern(i);
      stereo[i] = 0x1 & hitpat.getHitPattern(i) >> 2;
      subsubstruct[i] = 0xf & hitpat.getHitPattern(i) >> 3;
      substruct[i] = 0x7 & hitpat.getHitPattern(i) >> 7;
      detector[i] = 0x01 & hitpat.getHitPattern(i) >> 10;
      res[0][i] = residuals.residualX(i, hitpat);
      res[1][i] = residuals.residualY(i, hitpat);
   }
   nhits=numHits;


   int ndet=1;
   int det[64];
   det[0]=0;
   for(int j=0; j<nhits-1;ndet++) {
      //    printf("%i %i %s",ndet,j,det_str[j]);
      int k=j+1;
      for(; k<nhits ; k++) {
         if(substruct[j]==substruct[k]  && subsubstruct[j]==subsubstruct[k]) {
            //	printf("\t %i %s %i",k,det_str[k],layer[k]);
            //	printf(" same\n");
         }
         else break;
      }
      j=k;
      // printf("%i\n",k);
      det[ndet]=j;
   }
   det[ndet]=det[ndet-1]+1;
   //for(int l=0; l<ndet; l++)
   //   printf("%i %s %i\n",det[l],det_tracker_str[substruct[det[l]]-1],subsubstruct[det[l]]);


   float larray[9]={0.0,0.5,1.0,1.5,2.0,2.5,3.5,4.5,5.5};
   float larray2[8];
   for(int l=0; l<8; l++) {
      float diff2=(larray[l+1]-larray[l])/2;
      larray2[l]=larray[l]+diff2;
   }
   //   for(int i=0; i<8; i++)
   //   printf("(%.1f,%.1f),",larray[i],larray[i+1]);

   int resi[2][64];

   for(int l=0; l<nhits; l++) {
      for(int k=0; k<8; k++) {
         if(fabs(res[0][l])==larray2[k]) resi[0][l]=k;
         if(fabs(res[1][l])==larray2[k]) resi[1][l]=k;
      }
   }


   //______________________________________________________________________________
   // text canvas

   char mytext[256];
   int nvalid=0;
   int npix=0;
   int nstrip=0;
   for(int i=0; i<nhits; i++) {
      if(hittype[i]==0) nvalid++;
      if(substruct[i]<3) npix++;
      else nstrip++;
   }

   textCanvas()->cd();

   TLatex* latex = new TLatex(0.02, 0.970, "");
   latex->SetTextSize(0.07);
   float_t x0 = 0.02;
   float   y = 0.83;
   float fontsize = latex->GetTextSize()*0.6;
      
   // top text
   {
      sprintf(mytext,"valid/total Si hits: %i/%i",nvalid,nhits);
      latex->DrawLatex(x0, y, mytext);
      y -= fontsize;

      sprintf(mytext,"valid Si pixel (strip) hits: %i (%i)",npix,nstrip);
      latex->DrawLatex(x0, y, mytext);
      y -= fontsize;

      sprintf(mytext,"Number of layers hit: %i",ndet);
      latex->DrawLatex(x0, y, mytext);
   }
   // legend
   {
      textCanvas()->cd();

      Float_t x1 = 0.4;
      Float_t x2 = 0.7;
      Float_t boxH = 0.02;

      //_________________________
      y -= fontsize*2;
      sprintf(mytext,"r-phi hit");
      latex->DrawLatex(x0, y, mytext);

      TBox *b31 = new TBox(x1,y,x2,y+boxH);
      b31->SetFillColor(kGreen-9);
      b31->Draw();

      TBox *b3 = new TBox(x1,y,x2,y+boxH);
      b3->SetFillStyle(3006);
      b3->SetFillColor(1);
      b3->Draw();

      TBox *b6 = new TBox(x1,y,x2,y+boxH);
      b6->SetFillStyle(0);
      b6->SetFillColor(1);
      b6->SetLineWidth(2);
      b6->Draw();

      //--------------
      y = y - boxH*2;
      sprintf(mytext,"stereo hit");
      latex->DrawLatex(x0, y, mytext);

      TBox *b21 = new TBox(x1,y,x2,y+boxH);
      b21->SetFillColor(kCyan-9);
      b21->Draw();

      TBox *b2 = new TBox(x1,y,x2,y+boxH);
      b2->SetFillStyle(3004);
      b2->SetFillColor(1);
      b2->Draw();

      TBox *b5 = new TBox(x1,y,x2,y+boxH);
      b5->SetFillStyle(0);
      b5->SetFillColor(1);
      b5->SetLineWidth(2);
      b5->Draw();

      //--------------
      y = y -boxH*2;
      sprintf(mytext,"invalid hit");
      latex->DrawLatex(x0, y, mytext);

      TBox *b11 = new TBox(x1, y, x2, y+boxH);
      b11->SetFillColor(2);
      b11->Draw();

      TBox *b1 = new TBox(x1, y, x2, y+boxH);
      b1->SetFillColor(1);
      b1->SetFillStyle(3001);
      b1->Draw();


      TBox *b4 = new TBox(x1, y, x2, y+boxH);
      b4->SetFillStyle(0);
      b4->SetFillColor(1);
      b4->SetLineWidth(2);
      b4->Draw();
   }

   //______________________________________________________________________________
   // draw histogram
   TCanvas *canvas = FWTrackDetailView::viewCanvas();
   canvas->cd();
   canvas->SetFillColor(kOrange-8);

   TPad* pads[2];
   pads[0] = new TPad("pad1","pad1",0.05,0.05,0.475,0.95);
   canvas->cd();
   pads[0]->Draw();
   pads[1] = new TPad("pad2","pad2",0.535,0.05,0.95,0.95);
   canvas->cd();
   pads[1]->Draw();

   TH2F* h_res[2];
   h_res[0] = new TH2F("h_resx","h_resx",10,-5.5,5.5,ndet,0,ndet);
   h_res[1] = new TH2F("h_resy","h_resy",10,-5.5,5.5,ndet,0,ndet);
   char* res_str[2]={"x-axis residuals in Si detector local x-y coord.","y-axis residuals in Si detector local x-y coord."};

   char* xtitle[2] = {"resX","resY"};
   for(int i=0; i<2; i++) {
      pads[i]->cd();
      pads[i]->SetLeftMargin(0.15);
      pads[i]->SetFillColor(0);

      pads[i]->Range(-7.7,-1.5,6.966667,13.5);
      pads[i]->Modified();
      //    h_res[i]->SetTitle(res_str[i]);
      h_res[i]->SetTitle("");
      h_res[i]->SetTitleSize(0.04);
      h_res[i]->GetYaxis()->SetRangeUser(0,ndet+1);
      h_res[i]->GetYaxis()->SetLabelSize(0.06);
      h_res[i]->SetXTitle(xtitle[i]);
      h_res[i]->GetXaxis()->SetTickLength(0);
      h_res[i]->GetYaxis()->SetTickLength(0);
      h_res[i]->GetXaxis()->SetNdivisions(20);
      h_res[i]->Draw();
      pads[i]->SetGridy();
      TPaveText *pt = new TPaveText(0.05,0.915,0.95,0.995,"blNDC");
      pt->SetName("mytitle");
      pt->SetBorderSize(0);
      pt->SetFillColor(0);
      pt->AddText(res_str[i]);
      pt->Draw();
      canvas->cd();
   }

   TLine* lines[17];
   for(int l=0; l<17; l++) {
      int ix=l%9;
      int sign=1;
      sign = (l>8) ? -1 : 1;
      lines[l] = new TLine(sign*larray[ix],0,sign*larray[ix],ndet);
      if(l!=9)
         lines[l]->SetLineStyle(3);
      pads[0]->cd();
      lines[l]->Draw();
      pads[1]->cd();
      lines[l]->Draw();
   }

   float width=0.25;
   for(int h=0; h<2; h++) {
      pads[h]->cd();
      //      TPaveText *text_error = new TPaveText(5.6,0,6.0,ndet);
      //      text_error->Draw();

      float height1=0;
      for(int j=0; j<ndet; j++) {
         char det_str2[256];
         sprintf(det_str2,"%s/%i",det_tracker_str[substruct[det[j]]-1],subsubstruct[det[j]]);
         h_res[h]->GetYaxis()->SetBinLabel(j+1,det_str2);
         int diff=det[j+1]-det[j];
         int k=0;
         width=1.0/diff;
         // printf("%f\n",width);
         for(int l=det[j]; l<(det[j]+diff); l++) {
            //      g->SetPoint(l,resx[l],j+0.5);
            //	printf("%i, %f %f %f\n",l,resx[l],sign*larray[resxi[l]],sign*larray[resxi[l]+1]);
            int sign = (res[h][l]<0) ? -1 : 1;
            float left = (hittype[l]==0) ? sign*larray[resi[h][l]] : -5.5;
            float right = (hittype[l]==0) ? sign*larray[resi[h][l]+1] : 5.5;



            //	else width=0.25;
            float height=1.0;
            if(substruct[det[j]]<3) height=0.5;
            //	float height1=j+width*k;
            //	float height2=j+width*(k+1);
            //	float height1_tmp=j+width*k;
            //	float height2_tmp=j+width*(k+1);
            float height1_tmp=height1+width*k;
            float height2_tmp=height1+width*(k+1);
            TBox *tbox = new TBox(left,height1_tmp,right,height2_tmp);
            TBox *tbox3 = new TBox(left,height1_tmp,right,height2_tmp);

            int filltype = 3006;
            int color=kGreen-9;
            if(stereo[l]==1) {
               color=kCyan-9;
               filltype=3004;
            }
            if(hittype[l]!=0) {
               filltype=3001;
               color=kRed;
            }
            //	int color=1;
            //	if(hittype[l]!=0) color=2;
            //	tbox->SetLineColor(1);
            tbox->SetFillColor(1);
            tbox->SetFillStyle(filltype);
            tbox->SetLineColor(1);
            tbox3->SetFillColor(color);
            tbox3->Draw();
            tbox->Draw();

            TBox *tbox2 = new TBox(left,height1_tmp,right,height2_tmp);
            tbox2->SetFillColor(1);
            tbox2->SetFillStyle(0);
            tbox2->SetLineWidth(2);
            tbox2->Draw();
            k++;
         }
         //      if(substruct[det[j]]<3) height1+=0.5;
         //      else height1++;
         height1++;
      }
      //    printf("\n");
   }
   return 0;
}

REGISTER_FWDETAILVIEW(FWTrackDetailView);

