#ifndef PhysicsTools_MVATrainer_Source_h
#define PhysicsTools_MVATrainer_Source_h

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariableSet.h"

namespace PhysicsTools {

class MVATrainer;
namespace { class TrainInterceptor; }

class Source {
    public:
	Source(AtomicId name, bool trained = false) :
		trained(trained), name(name) {}
	virtual ~Source() {}

	inline AtomicId getName() const { return name; }

	inline SourceVariable *getOutput(AtomicId name) const
	{ return outputs.find(name); }

	inline bool isTrained() const { return trained; }

    protected:
	friend class MVATrainer;
	friend class TrainInterceptor;

	inline SourceVariableSet &getInputs() { return inputs; }
	inline SourceVariableSet &getOutputs() { return outputs; }

	inline const SourceVariableSet &getInputs() const { return inputs; }
	inline const SourceVariableSet &getOutputs() const { return outputs; }

	bool			trained;

    private:
	AtomicId		name;
	SourceVariableSet	inputs;
	SourceVariableSet	outputs;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_Source_h
