void testFWLiteEvaluation()
{
	using namespace PhysicsTools;

	MVAComputer mva("TrainedGauss.mva");

	double x, y;

	TreeReader reader;
	reader.addSingle("x", &x);
	reader.addSingle("y", &y);

	x = 2, y = 2;
	cout << "at (+2.0, +2.0): " << reader.fill(&mva) << endl;

	x = 0.1, y = 0.1;
	cout << "at (+0.1, +0.1): " << reader.fill(&mva) << endl;

	x = 0, y = 0;
	cout << "at (+0.0, +0.0): " << reader.fill(&mva) << endl;

	x = -0.1, y = -0.1;
	cout << "at (-0.1, -0.1): " << reader.fill(&mva) << endl;

	x = -2, y = -2;
	cout << "at (-2.0, -2.0): " << reader.fill(&mva) << endl;
}
