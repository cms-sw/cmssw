#include <iostream>
#include <algorithm>
#include <vector>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariableSet.h"

namespace PhysicsTools {

bool SourceVariableSet::append(SourceVariable *var, int offset)
{
	std::vector<PosVar>::iterator pos =
			std::lower_bound(vars.begin(), vars.end(),
			                 var->getName(), PosVar::VarNameLess);

	if (pos != vars.end() && pos->var == var)
		return true;

	PosVar item;
	item.pos = vars.size() + offset;
	item.var = var;

	vars.insert(pos, 1, item);

	return false;
}

SourceVariable *SourceVariableSet::find(AtomicId name) const
{
	std::vector<PosVar>::const_iterator pos =
			std::lower_bound(vars.begin(), vars.end(),
			                 name, PosVar::VarNameLess);

	if (pos == vars.end() || pos->var->getName() != name)
		return 0;

	return pos->var;
}

std::vector<SourceVariable*> SourceVariableSet::get() const
{
	std::vector<SourceVariable*> result(vars.size());

	for(std::vector<PosVar>::const_iterator iter = vars.begin();
	    iter != vars.end(); iter++)
		result[iter->pos] = iter->var;

	return result;
}

} // namespace PhysicsTools
