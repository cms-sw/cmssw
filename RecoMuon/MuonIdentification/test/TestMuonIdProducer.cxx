{
   gStyle->SetPalette(1,0);
   TFile f("/afs/cern.ch/user/d/dmytro/scratch0/muons.root");
   
   c3 = new TCanvas("c3","HCAL energy deposition");
   // Events->Draw("muons.energy().had>>h3(100,0,10)","fabs(muons.eta())<1");
   Events->Draw("muons.calEnergy().had>>h3(100,0,10)");
   
   c4 = new TCanvas("c4","ECAL energy deposition (barrel)");
   Events->Draw("muons.calEnergy().em>>h4(100,0,2)","abs(muons.track().get().eta())<1");

   gROOT->LoadMacro("resolution_fit.cxx");
   c5 = new TCanvas("c5","Muon DT segments",400,800);
   c5->Divide(2,4);
   c5->cd(1);
   Events->Draw("muons.dX(0)>>hh1(100,-50,50)","muons.dXErr(0)<100");
   resolution_fit(hh1,"dX[0]");
   c5->cd(2);
   Events->Draw("muons.dY(0)>>hh2(100,-50,50)","muons.dYErr(0)<100");
   resolution_fit(hh2,"dY[0]");
   c5->cd(3);
   Events->Draw("muons.dX(1)>>hh3(100,-50,50)","muons.dXErr(1)<100");
   resolution_fit(hh3,"dX[1]");
   c5->cd(4);
   Events->Draw("muons.dY(1)>>hh4(100,-50,50)","muons.dYErr(1)<100");
   resolution_fit(hh4,"dY[1]");
   c5->cd(5);
   Events->Draw("muons.dX(2)>>hh5(100,-50,50)","muons.dXErr(2)<100");
   resolution_fit(hh5,"dX[2]");
   c5->cd(6);
   Events->Draw("muons.dY(2)>>hh6(100,-50,50)","muons.dYErr(2)<100");
   resolution_fit(hh6,"dY[2]");
   c5->cd(7);
   Events->Draw("muons.dX(3)>>hh7(100,-50,50)","muons.dXErr(3)<100");
   resolution_fit(hh7,"dX[3]");
   
   c6 = new TCanvas("c6","Muon pulls",400,800);
   c6->Divide(2,4);
   c6->cd(1);
   Events->Draw("muons.dX(0)/muons.dXErr(0)>>hp1(100,-10,10)","muons.dXErr(0)<100");
   resolution_fit(hp1,"pull X[0]");
   c6->cd(2);
   Events->Draw("muons.dY(0)/muons.dYErr(0)>>hp2(100,-10,10)","muons.dYErr(0)<100");
   resolution_fit(hp2,"pull Y[0]");
   c6->cd(3);
   Events->Draw("muons.dX(1)/muons.dXErr(1)>>hp3(100,-10,10)","muons.dXErr(1)<100");
   resolution_fit(hp3,"pull X[1]");
   c6->cd(4);
   Events->Draw("muons.dY(1)/muons.dYErr(1)>>hp4(100,-10,10)","muons.dYErr(1)<100");
   resolution_fit(hp4,"pull Y[1]");
   c6->cd(5);
   Events->Draw("muons.dX(2)/muons.dXErr(2)>>hp5(100,-10,10)","muons.dXErr(2)<100");
   resolution_fit(hp5,"pull X[2]");
   c6->cd(6);
   Events->Draw("muons.dY(2)/muons.dYErr(2)>>hp6(100,-10,10)","muons.dYErr(2)<100");
   resolution_fit(hp6,"pull Y[2]");
   c6->cd(7);
   Events->Draw("muons.dX(3)/muons.dXErr(3)>>hp7(100,-10,10)","muons.dXErr(3)<100");
   resolution_fit(hp7,"pull X[3]");

}
