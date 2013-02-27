/*
A macro to show how a calculation of A.Read confidences levels can be performed
once built the -2lnQ distributions.
*/

{
// Load the library and get the combined model from the card
gSystem->Load("libRooStatsCms.so");
RscAbsPdfBuilder::setDataCard("example_qqhtt.rsc");
RscCombinedModel model("qqhtt");

// Get the pdf of the model(sb and bonly components)
RooAbsPdf* sb_model=model.getPdf();
RooAbsPdf* b_model=model.getBkgPdf();

// The object that collects all the constraints and the blocks of correlated constraints
ConstrBlockArray constr_array("ConstrArray", "The array of constraints");

// Facility to read the card and fill the constraints array properly
RscConstrArrayFiller filler("ConstrFiller", "The array Filler", model.getConstraints());
filler.fill(&constr_array);

// The -2lnQ calculator
LimitCalculator calc("m2lnQmethod","-2lnQ method",sb_model,b_model,&model.getVars(),&constr_array);

/*
Calculate the -2lnQ distributions,2000 toys, fluctuating the constraints.
We do not provide data: we will take later the median of the -2lnQ histo
relative to the SB datasets.
*/

bool fluctuate = true;
LimitResults* res = calc.calculate (2000,fluctuate);

// Get theplot and impose to observe the median of the SB distribution
LimitPlot* p = res->getPlot("temp","temp",200);
res->LimitResults::setM2lnQValue_data(Rsc::getMedian(p->getSBhisto()));
delete p;

// Now the plot that will be displayed
p = res->getPlot("-2lnQmethod","-2lnQ distributions", 100);
p->draw();

// A written summary
cout << "\nWARNING\nTo build a nice example for RSC the yiled of the signal was brought down from 10.33 to 3 events to have some results in a reasonable amount of toys (1000). It is clearly NOT what is stated in the TDR. This is a didactical example!!\n\n";

cout << "\nThe Summary:\n";
cout << " - CLsb = " << res->getCLsb() << endl;
cout << " - 1-CLb = " << 1 - res->getCLb() << endl;

}
