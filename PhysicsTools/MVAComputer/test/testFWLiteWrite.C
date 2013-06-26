// ***************************************************************************
// *                                                                         *
// *        IMPORTANT NOTE: You would never want to do this by hand!         *
// *                                                                         *
// * This is for testing purposes only. Use PhysicsTools/MVATrainer instead. *
// *                                                                         *
// ***************************************************************************

// In order for this example to work (which you should not do in FWLite
// anyway) you have to uncomment the commented-out parts in
// src/classes_def.xml to add definitions of BitSet dictionaries back

void testFWLiteWrite()
{
	using namespace PhysicsTools::Calibration;

// set up some dummy calibration by hand for testing

	MVAComputer calibration;

// vars

	Variable var;
	var.name = "test";
	calibration.inputSet.push_back(var);

	var.name = "normal";
	calibration.inputSet.push_back(var);

	var.name = "toast";
	calibration.inputSet.push_back(var);

// normalize

	ProcNormalize norm;

	PhysicsTools::BitSet testSet1(3);
	testSet1[0] = testSet1[1] = true;
	norm.inputVars = convert(testSet1);

	HistogramF pdf(3, 4.0, 5.5);
	pdf.setBinContent(1, 1.0);
	pdf.setBinContent(2, 1.5);
	pdf.setBinContent(3, 1.0);
	norm.categoryIdx = -1;
	norm.distr.push_back(pdf);
	norm.distr.push_back(pdf);

	calibration.addProcessor(&norm);

// likelihood

	ProcLikelihood lkh;

	PhysicsTools::BitSet testSet2(5);
	testSet2[2] = true;
	lkh.inputVars = convert(testSet2);

	pdf = HistogramF(6, 0.0, 1.0);
	pdf.setBinContent(1, 1.0);
	pdf.setBinContent(2, 1.5);
	pdf.setBinContent(3, 1.0);
	pdf.setBinContent(4, 1.0);
	pdf.setBinContent(5, 1.5);
	pdf.setBinContent(6, 1.0);
	ProcLikelihood::SigBkg sigBkg;
	sigBkg.signal = pdf;
	pdf = HistogramF(9, 0.0, 1.0);
	pdf.setBinContent(1, 1.0);
	pdf.setBinContent(2, 1.5);
	pdf.setBinContent(3, 1.0);
	pdf.setBinContent(4, 1.0);
	pdf.setBinContent(5, 1.5);
	pdf.setBinContent(6, 1.0);
	pdf.setBinContent(7, 1.5);
	pdf.setBinContent(8, 1.0);
	pdf.setBinContent(9, 1.7);
	sigBkg.background = pdf;
	sigBkg.useSplines = true;
	lkh.categoryIdx = -1;
	lkh.neverUndefined = true;
	lkh.individual = false;
	lkh.logOutput = false;
	lkh.keepEmpty = true;
	lkh.pdfs.push_back(sigBkg);

	calibration.addProcessor(&lkh);

// likelihood 2

	PhysicsTools::BitSet testSet3(6);
	testSet3[2] = testSet3[3] = true;
	lkh.inputVars = convert(testSet3);
	sigBkg.useSplines = true;
	lkh.pdfs.push_back(sigBkg);

	calibration.addProcessor(&lkh);

// optional

	ProcOptional opt;

	PhysicsTools::BitSet testSet4(7);
	testSet4[5] = testSet4[6] = true;
	opt.inputVars = convert(testSet4);

	opt.neutralPos.push_back(0.6);
	opt.neutralPos.push_back(0.7);

	calibration.addProcessor(&opt);

// PCA

	ProcMatrix pca;

	PhysicsTools::BitSet testSet5(9);
	testSet5[4] = testSet5[7] = testSet5[8] = true;
	pca.inputVars = convert(testSet5);

	pca.matrix.rows = 2;
	pca.matrix.columns = 3;
	pca.matrix.elements.push_back(0.2);
	pca.matrix.elements.push_back(0.3);
	pca.matrix.elements.push_back(0.4);
	pca.matrix.elements.push_back(0.8);
	pca.matrix.elements.push_back(0.7);
	pca.matrix.elements.push_back(0.6);

	calibration.addProcessor(&pca);

// linear

	ProcLinear lin;

	PhysicsTools::BitSet testSet6(11);
	testSet6[9] = testSet6[10] = true;
	lin.inputVars = convert(testSet6);

	lin.coeffs.push_back(0.3);
	lin.coeffs.push_back(0.7);
	lin.offset = 0.0;

	calibration.addProcessor(&lin);

// output

	calibration.output = 11;

// write the calibration to a file called "test.mva"

	PhysicsTools::MVAComputer::writeCalibration("test.mva", &calibration);
}
