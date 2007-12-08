void testFWLiteEvaluation()
{
	using namespace PhysicsTools;

	MVAComputer mva("TrainedGauss.mva");

	Variable::Value vars[2];
	vars[0].setName("x");
	vars[1].setName("y");

	vars[0].setValue(2);
	vars[1].setValue(2);
	cout << "at (+2.0, +2.0): " << mva.eval(vars, vars + 2) << endl;

	vars[0].setValue(0.1);
	vars[1].setValue(0.1);
	cout << "at (+0.1, +0.1): " << mva.eval(vars, vars + 2) << endl;

	vars[0].setValue(0);
	vars[1].setValue(0);
	cout << "at (+0.0, +0.0): " << mva.eval(vars, vars + 2) << endl;

	vars[0].setValue(-0.1);
	vars[1].setValue(-0.1);
	cout << "at (-0.1, -0.1): " << mva.eval(vars, vars + 2) << endl;

	vars[0].setValue(-2);
	vars[1].setValue(-2);
	cout << "at (-2.0, -2.0): " << mva.eval(vars, vars + 2) << endl;
}
