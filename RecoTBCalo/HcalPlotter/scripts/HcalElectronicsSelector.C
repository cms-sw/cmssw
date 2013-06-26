#include "HcalElectronicsSelector.h"
#include "MyHcalClasses.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TRandom.h"
#include "TStyle.h"

inline double sign(double x) { return (x < 0.0) ? -1.0 : 1.0; }

static const float BADVAL = -1e99;

HcalElectronicsSelector::HcalElectronicsSelector(Callbacks* cb,
						 int htrChan_lo, int htrChan_hi,
						 int fpga_lo, int fpga_hi,int crate)
{
  m_crate=crate;
  m_cb=cb;
  m_canvas=new TCanvas("HcalSelector");
 
  
  int htrChan_bins=htrChan_hi-htrChan_lo+1;
  int fpga_bins=(fpga_hi-fpga_lo+1)*2;
  
  char name[10];
  char title[256];
 
  sprintf(title,"htrChan/fpga Space Crate %d",crate);
  sprintf(name,"Crate %d",crate);
  TH2F* h = new TH2F(title,name,
		     fpga_bins, fpga_lo-0.25, fpga_hi+0.75,
		     htrChan_bins, htrChan_lo-0.5, htrChan_hi+0.5);
		  
  h->GetYaxis()->SetTitle("HTR CHANNEL");
  h->GetXaxis()->SetTitle("Slot");
  h->SetStats(0);
  // Set bins to a 'badval' in order to distinguish
  // between zero-mean cells and unconnected cells
  //
  for (int binx=1; binx<=h->GetNbinsX(); binx++)
    for (int biny=1; biny<=h->GetNbinsY(); biny++)
      h->SetBinContent(binx,biny,BADVAL);
  m_hist = h;
 
  TStyle* ms=new TStyle("hvs","hvs");
  ms->SetPalette(1,0);
  ms->cd();
  
 
  m_canvas->cd();
    m_hist->Draw("COLZ");
    m_canvas->Update();
  
  m_canvas->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
		    "HcalElectronicsSelector", this,
		    "onEvent(Int_t,Int_t,Int_t,TObject*)");//add crate
  
  m_canvas->cd();
}

void HcalElectronicsSelector::onEvent(Int_t event, Int_t x, Int_t y, TObject *selected)
{
  if(event!=kButton1Double) return;
  TPad *pad=(TPad*) m_canvas->GetSelectedPad();
  if(selected==0) return; // this is to suppress "Unused" message
  //  printf("Canvas %s: event=%d, x=%d, y=%d, selected=%s\n",
  //         m_canvas->GetName(),event, x, y, selected->IsA()->GetName());
  

   char padname[256];
   strcpy(padname,m_canvas->GetSelectedPad()->GetName());

   char canvasname[40];
  
   Float_t px= pad->AbsPixeltoX(x);
   Float_t py= pad->AbsPixeltoY(y);

   px=pad->PadtoX(px);
   py=pad->PadtoY(py);

   //printf("x=%.3g,y=%.3g\n",px,py);

   int Xbin=m_hist->GetXaxis()->FindBin(px);
   int Ybin=m_hist->GetYaxis()->FindBin(py);

   //printf("Xbin=%d,Ybin=%d\n",Xbin,Ybin);
   if (Xbin==0 || Ybin==0 || 
       Xbin>m_hist->GetXaxis()->GetNbins() ||
       Ybin>m_hist->GetYaxis()->GetNbins()) {
     printf("Please select a location within the graph boundaries.\n");
     return;
   }
   
   int htrChan=(int)(py+(sign(py)*0.5));
   int fpga=(int)(px*2);
   
   int crate=m_crate;

   printf("htrChan=%d fpga=%d Crate=%d\n",htrChan,fpga,crate);
   if (m_cb==0) return; // need a callback
   
   int fbr=((htrChan-1)/3)+1;
   int fbrchan=(htrChan-1)%3;
   int slt=(fpga/2);
   int tb=(fpga%2)?(1):(0);
   
   MyElectronicsId id = {fbr,fbrchan,crate,slt,tb};
   m_cb->plot(id);
}

void HcalElectronicsSelector::fill(const MyElectronicsId& id, double value) {
 
  TH2* h = m_hist;
  int htrChan=(id.fiber-1)*3+1+id.fiberChan;
  double fpga= (id.tb==1)?(id.Slot+0.5):(id.Slot);

  h->SetBinContent(h->FindBin(fpga*1.0,htrChan*1.0,0.0),value);
}

void HcalElectronicsSelector::Update() {
  m_canvas->Flush();
  m_canvas->Update();
  m_canvas->Paint();
}


