#ifndef PhysicsTools_MVATrainer_SourceVariableSet_h
#define PhysicsTools_MVATrainer_SourceVariableSet_h

#include <cstddef>
#include <vector>
#include <algorithm>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

namespace PhysicsTools {

class SourceVariable;

class SourceVariableSet {
    public:
	typedef std::size_t size_type;

	SourceVariableSet() {}
	~SourceVariableSet() {}

	enum Magic {
		kRegular = 0,
		kTarget,
		kWeight
	};

	SourceVariable *find(AtomicId name) const; SourceVariable
	*find(Magic magic) const;

	bool append(SourceVariable *var, Magic magic = kRegular,
	            int offset = -1);
	std::vector<SourceVariable*> get(bool withMagic = false) const;
	size_type size(bool withMagic = false) const
	{
		if (withMagic)
			return vars.size();
		else
			return std::count_if(vars.begin(), vars.end(),
			                     std::mem_fun_ref(&PosVar::noMagic));
	}

    private:
	struct PosVar {
		unsigned int	pos;
		SourceVariable	*var;
		Magic		magic;

		bool noMagic() const { return magic == kRegular; }

		static bool VarNameLess(const PosVar &var, AtomicId name)
		{ return var.var->getName() < name; }
	};

	std::vector<PosVar>	vars;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_SourceVariable_h
