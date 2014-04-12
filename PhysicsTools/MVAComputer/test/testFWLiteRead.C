void testFWLiteRead()
{
	using namespace PhysicsTools;

	MVAComputer mva("test.mva");

	// note that there is also a TTree interface which can read entries
	// directly from a ROOT tree. This interface is found in the
	// class "TreeReader" in this package.
	//
	// Note that the TreeReader can also be used for ROOT-like
	// interfacing of the MVAComputer::eval method, like the
	// TTree::Fill() method. This might come in handy and might be
	// simpler to use than the ValueList interface.
	//
	// See "MVATrainer/test/testFWLiteEvaluation.C" for an example.

	Variable::ValueList vars;
	vars.add("toast", 4.4);
	vars.add("toast", 4.5);
	vars.add("test", 4.6);
	vars.add("toast", 4.7);
	vars.add("test", 4.8);
	vars.add("normal", 4.9);

	cout << "This is expected to give 0.955976:" << endl;

	cout << mva.eval(vars) << endl;
}
