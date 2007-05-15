#ifndef Source_h
#define Source_h

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariableSet.h"

class MVATrainer;

class Source {
    public:
	Source(PhysicsTools::AtomicId name, bool trained = false) :
		trained(trained), name(name) {}
	virtual ~Source() {}

	inline PhysicsTools::AtomicId getName() const { return name; }

	inline SourceVariable *getOutput(PhysicsTools::AtomicId name) const
	{ return outputs.find(name); }

	inline bool isTrained() const { return trained; }

    protected:
	friend class MVATrainer;

	inline SourceVariableSet &getInputs() { return inputs; }
	inline const SourceVariableSet &getInputs() const { return inputs; }

	inline SourceVariableSet &getOutputs() { return outputs; }
	inline const SourceVariableSet &getOutputs() const { return outputs; }

	bool				trained;

    private:
	PhysicsTools::AtomicId		name;
	SourceVariableSet		inputs;
	SourceVariableSet		outputs;
};

#endif // Source_h
