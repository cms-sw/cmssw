#ifndef PhysicsTools_MVATrainer_SourceVariableSet_h
#define PhysicsTools_MVATrainer_SourceVariableSet_h

#include <cstddef>
#include <vector>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

namespace PhysicsTools {

class SourceVariable;

class SourceVariableSet {
    public:
	typedef std::size_t size_type;

	SourceVariableSet() {}
	~SourceVariableSet() {}

	SourceVariable *find(AtomicId name) const;
	bool append(SourceVariable *var, int offset = 0);
	std::vector<SourceVariable*> get() const;
	inline size_type size() const { return vars.size(); }

    private:
	struct PosVar {
		unsigned int	pos;
		SourceVariable	*var;

		static bool VarNameLess(const PosVar &var, AtomicId name)
		{ return var.var->getName() < name; }
	};

	std::vector<PosVar>	vars;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_SourceVariable_h
