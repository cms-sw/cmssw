#include "HcalVisualSelector.h"
#include "MyHcalClasses.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TRandom.h"
#include "TStyle.h"

inline double sign(double x) { return (x < 0.0) ? -1.0 : 1.0; }

static const float BADVAL = -1e99;

HcalVisualSelector::HcalVisualSelector(Callbacks* cb,
				       int ieta_lo, int ieta_hi,
				       int iphi_lo, int iphi_hi)
{
  m_cb=cb;
  m_canvas=new TCanvas("HcalSelector");
  m_canvas->Divide(2,2);
  
  int ieta_bins=ieta_hi-ieta_lo+1;
  int iphi_bins=iphi_hi-iphi_lo+1;

  char name[10];
  char title[256];
  for(int i=0;i<4;i++) {
    sprintf(title,"iEta/iPhi Space Depth %d",i+1);
    sprintf(name,"Depth %d",i+1);
    TH2F* h = new TH2F(title,name,
		       ieta_bins, ieta_lo-0.5, ieta_hi+0.5,
		       iphi_bins, iphi_lo-0.5, iphi_hi+0.5);
    h->GetXaxis()->SetTitle("IETA");
    h->GetYaxis()->SetTitle("IPHI");
    h->SetStats(0);
    // Set bins to a 'badval' in order to distinguish
    // between zero-mean cells and unconnected cells
    //
    for (int binx=1; binx<=h->GetNbinsX(); binx++)
      for (int biny=1; biny<=h->GetNbinsY(); biny++)
	h->SetBinContent(binx,biny,BADVAL);
    m_hist[i] = h;
  }

  TStyle* ms=new TStyle("hvs","hvs");
  ms->SetPalette(1,0);
  ms->cd();

  for(int i=0;i<4;i++) {
    m_canvas->cd(i+1);
    m_hist[i]->Draw("COLZ");
    m_canvas->Update();
  }
  m_canvas->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
		    "HcalVisualSelector", this,
		    "onEvent(Int_t,Int_t,Int_t,TObject*)");
  
  m_canvas->cd();
}

void HcalVisualSelector::onEvent(Int_t event, Int_t x, Int_t y, TObject *selected) {
  if(event!=kButton1Double) return;
  TPad *pad=(TPad*) m_canvas->GetSelectedPad();
  if(selected==0) return; // this is to suppress "Unused" message
  //  printf("Canvas %s: event=%d, x=%d, y=%d, selected=%s\n",
  //         m_canvas->GetName(),event, x, y, selected->IsA()->GetName());
  

   char padname[256];
   strcpy(padname,m_canvas->GetSelectedPad()->GetName());

   if (strstr(padname,"_"))
     *(strstr(padname,"_"))=' ';

   char canvasname[40];
   int padnum;

   sscanf(padname,"%s %d",canvasname,&padnum);

   Float_t px= pad->AbsPixeltoX(x);
   Float_t py= pad->AbsPixeltoY(y);

   px=pad->PadtoX(px);
   py=pad->PadtoY(py);

   //printf("x=%.3g,y=%.3g\n",px,py);

   int Xbin=m_hist[padnum-1]->GetXaxis()->FindBin(px);
   int Ybin=m_hist[padnum-1]->GetYaxis()->FindBin(py);

   //printf("Xbin=%d,Ybin=%d\n",Xbin,Ybin);
   if (Xbin==0 || Ybin==0 || 
       Xbin>m_hist[padnum-1]->GetXaxis()->GetNbins() ||
       Ybin>m_hist[padnum-1]->GetYaxis()->GetNbins()) {
     printf("Please select a location within the graph boundaries.\n");
     return;
   }
   // convert to ieta/iphi
   int ieta=(int)(px+(sign(px)*0.5));
   int iphi=(int)(py+(sign(py)*0.5));
   int depth=padnum;
   printf("ieta=%d iphi=%d depth=%d\n",ieta,iphi,depth);
   if (m_cb==0) return; // need a callback

   // figure out Subdetector
   MyHcalSubdetector sd=m_cb->getSubdet(ieta,depth);
      
   MyHcalDetId id = {sd,ieta,iphi,depth};
   m_cb->plot(id);
}

void HcalVisualSelector::fill(const MyHcalDetId& id, double value) {
  //if (id.depth!=1) return;
  // what about depth?
  TH2* h = m_hist[id.depth-1];
  h->SetBinContent(h->FindBin(id.ieta*1.0,id.iphi*1.0,0.0),value);
}

void HcalVisualSelector::Update() {
  m_canvas->Flush();
  m_canvas->Update();
  m_canvas->Paint();
}
