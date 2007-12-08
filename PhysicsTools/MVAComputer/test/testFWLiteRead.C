void testFWLiteRead()
{
	using namespace PhysicsTools;

	MVAComputer mva("test.mva");

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
