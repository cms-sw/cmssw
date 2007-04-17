{
gROOT->Macro("init.C");

Chain std("Eff","Out_std/out_b_singlegamma_pf_*root");

ResidualFitter eri1("eri1","truc",30,-3,3,100,0,200,100,-0.05,0.05);
std.Draw("clusters_.eta-particles_.eta:particles_.e:particles_.eta>>eri1","","goff");
eri1.SetAutoRange(3);
eri1.SetFitOptions("W");
eri1.FitSlicesZ();
eri1.cd();
eri1_sigma.Draw("colz");

float max = 0.025;
float min = 0;
eri1_sigma.SetMaximum(max);
eri1_sigma.SetMinimum(min);

PFResolutionMap* etanew = (PFResolutionMap*) eri1_sigma.Clone("etanew");


PFResolutionMap etaold("etaold","../../PFProducer/data/resmap_ECAL_eta.dat");
etaold.SetMaximum(max);
etanew.SetMinimum(min);
}
