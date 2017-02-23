{
  gROOT->ProcessLine(" .L tdrstyle.C");
  gROOT->ProcessLine("setTDRStyle()");
  gROOT->ProcessLine(".L HIPplots.cc+");
  gROOT->ProcessLine(".L plotter.C+");
}
