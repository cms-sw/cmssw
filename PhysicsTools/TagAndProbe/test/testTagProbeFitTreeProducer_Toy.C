using namespace RooFit ;

void testTagProbeFitTreeProducer_Toy(){
  // create test input file
  TFile out("testTagProbeFitTreeProducer_Toy.root","recreate");
  out.mkdir("Test")->cd();
  // pt
  RooRealVar pt("pt","pt",1,10);
  RooGenericPdf ptPdf("ptPdf","1/pt",pt);
  // eta
  RooRealVar eta("eta","eta",-2.4,2.4);
  // passing
  RooCategory passing("passing","passing");
  passing.defineType("Passed",1);
  passing.defineType("Failed",0);
  // efficiency value
  RooRealVar a("a","a",0.9) ;
  RooRealVar b("b","b",1) ;
  RooRealVar c("c","c",3) ;
  // define efficiency as a function or just a constant
//  RooFormulaVar efficiency("efficiency", "a/(1+exp(-b*(pt-c)))", RooArgList(a, b, c, pt)) ;
  RooRealVar efficiency("efficiency", "efficiency", 0.95, 0.0, 1.0);
  // save a plot of the efficiency curve
  RooPlot* curve = pt.frame();
  efficiency.plotOn(curve);
  curve->Write("efficiencyCurve");
  RooEfficiency passingPdf("passingPdf", "passingPdf", efficiency, passing, "Passed");
  // mass
  RooRealVar mass("mass", "mass", 2.6, 3.6);
  RooRealVar mean("mean", "mean", 3.1, 3.0, 3.2);
  RooRealVar sigma("sigma", "sigma", 0.03, 0.01, 0.05);
  RooGaussian signal("signal", "signal", mass, mean, sigma);
  RooRealVar numSignalAll("numSignalAll", "numSignalAll", 10000., 0., 1e10);
  RooExtendPdf signalExt("signalExt", "signalExt", signal, numSignalAll);
  RooProdPdf signalPdf("signalPdf", "signalPdf", RooArgSet(signalExt, ptPdf), Conditional(passingPdf,passing));
  
  RooRealVar cPass("cPass", "cPass", 0.1, -1, 1);
  RooChebychev backgroundPass("backgroundPass", "backgroundPass", mass, cPass);
  RooRealVar numBackgroundPass("numBackgroundPass", "numBackgroundPass", 10000., 0., 1e10);
  RooExtendPdf backgroundPassPdf("backgroundPassPdf", "backgroundPassPdf", backgroundPass, numBackgroundPass);
  
  RooRealVar cFail("cFail", "cFail", 0.1, -1, 1);
  RooChebychev backgroundFail("backgroundFail", "backgroundFail", mass, cFail);
  RooRealVar numBackgroundFail("numBackgroundFail", "numBackgroundFail", 10000., 0., 1e10);
  RooExtendPdf backgroundFailPdf("backgroundFailPdf", "backgroundFailPdf", backgroundFail, numBackgroundFail);
  
  RooSimultaneous backgroundPdf("backgroundPdf", "backgroundPdf", passing);
  backgroundPdf.addPdf(backgroundPassPdf, "Passed");
  backgroundPdf.addPdf(backgroundFailPdf, "Failed");
  RooProdPdf backgroundPtPdf("backgroundPtPdf", "backgroundPtPdf", backgroundPdf, ptPdf);
  // mc_true, True=signal, False=background
  RooCategory mc_true("mcTrue","mcTrue");
  mc_true.defineType("True",1);
  mc_true.defineType("False",0);

  RooSimultaneous model("model", "model", mc_true);
  model.addPdf(signalPdf, "True");
  model.addPdf(backgroundPtPdf, "False");

  RooDataSet* data = model.generate(RooArgSet(mass, pt, eta, passing, mc_true));
  data->tree()->Write("fitter_tree");
  
  out.Close();
}

