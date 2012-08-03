#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "PhysicsTools/MVAComputer/interface/BitSet.h"
#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/TreeReader.h"

#ifdef __GCCXML__
namespace { // anonymous
struct dictionary {

std::vector<PhysicsTools::Variable::Value> vv;

};
static void dummy(PhysicsTools::MVAComputer *mva)
{
	PhysicsTools::Variable::ValueList v;

	mva->eval(v);
	mva->eval(v.values());
	mva->eval(v.begin(), v.end());
	mva->eval(v.data(), v.data());

	mva->deriv(v);
	mva->deriv(v.values());
	mva->deriv(v.begin(), v.end());
	mva->deriv(v.data(), v.data());
}

} // anonymous namespace

#endif
