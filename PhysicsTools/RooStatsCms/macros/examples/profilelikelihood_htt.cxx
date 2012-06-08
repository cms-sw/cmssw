/*
A macro to show how a Profile Likelihood scan can be performed and the limits 
obtained.
*/

{

/*
Load the library and get the model from the card.
Here we do not do for a Combined model but we call the TotModel.
*/
gSystem->Load("libRooStatsCms.so");
RscAbsPdfBuilder::setDataCard("example_qqhtt.rsc");
RscTotModel model("qqhtt_single");
RooAbsPdf* f=model.getPdf();

// Get the pdf of the model, generate a datasample and do the scan!
RooRandom::randomGenerator()->SetSeed(10); // make the macro reproducible..

RooDataSet* data = f->generate(*model.getVars(),RooFit::Extended());

LikelihoodCalculator lcalc(*f,*data);
PLScan myscan("test scan","",lcalc.getNLL(),"qqhtt_single_sig_yield",0,20,30);

// Get the result of the scan and take out some numbers
PLScanResults* res = myscan.doScan();
res->SetTitle("Likelihood scan for example qq#rightarrowH#rightarrow#tau#tau");

double deltanll95 = res->getDeltaNLLfromCL(.95);
double UL95=res->getUL(deltanll95);
double LL95=res->getLL(deltanll95);
std::cout << "Your 95% CL interval is [" << LL95 << "," << UL95<<"] .\n";

double deltanll68 = res->getDeltaNLLfromCL(.68);
double UL68=res->getUL(deltanll68);
double LL68=res->getLL(deltanll68);
std::cout << "Your 68% CL interval is [" << LL68 << "," << UL68<<"] .\n";

// Take out the plot, put some details in it and we're done!
PLScanPlot* p = res->getPlot();
p->addCLline(deltanll95,0.95,LL95,UL95);
p->addCLline(deltanll68,0.68,LL68,UL68);
p->draw();
p->dumpToImage("PLScanPlot_68CL_95CL.png");
p->dumpToFile("PLScanPlot_68CL_95CL.root","RECREATE");

}
