//
// simple Root Macro to plot and verify shower containment corrections
// author : Stefano Argiro'
//

{

  TFile f("testContCorrect.root");
  TTree *t =(TTree*) f.Get("test");

  TH2D *energy3x3 = 
    new TH2D("e3x3vsx","e3x3vsx",100, -10.,10.,100,120.,150.);
  TH2D *corrected3x3= 
    new TH2D("corr3x3vsx","corr3x3vsx",100, -10.,10.,100,120.,150.);
  TH2D *energy5x5= 
    new TH2D("e5x5vsx","e5x5vsx",100, -10.,10.,100,120.,150.);
  TH2D *corrected5x5= 
    new TH2D("corr5x5vsx","corr5x5vsx",100, -10.,10.,100,10.,150.);

  t->Project("e3x3vsx","energy3x3:posx");
  t->Project("corr3x3vsx","corrected3x3:posx");
  t->Project("e5x5vsx","energy5x5:posx");
  t->Project("corr5x5vsx","corrected5x5:posx");

  TCanvas * canvas = new TCanvas("corrections","corrections");
  canvas->Divide(2,2);

  canvas->cd(1);
  energy3x3.Draw();

  canvas->cd(2);
  corrected3x3.Draw();

  canvas->cd(3);
  energy5x5.Draw();

  canvas->cd(4);
  corrected5x5.Draw();
}
