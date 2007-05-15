#ifndef SourceVariableSet_h
#define SourceVariableSet_h

#include <cstddef>
#include <vector>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

class SourceVariable;

class SourceVariableSet {
    public:
	typedef std::size_t size_type;

	SourceVariableSet() {}
	~SourceVariableSet() {}

	SourceVariable *find(PhysicsTools::AtomicId name) const;
	bool append(SourceVariable *var, int offset = 0);
	std::vector<SourceVariable*> get() const;
	inline size_type size() const { return vars.size(); }

    private:
	struct PosVar {
		unsigned int	pos;
		SourceVariable	*var;

		static bool VarNameLess(const PosVar &var,
		                        PhysicsTools::AtomicId name)
		{ return var.var->getName() < name; }
	};

	std::vector<PosVar>	vars;
};

#endif // SourceVariable_h
