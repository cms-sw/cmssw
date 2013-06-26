#include <iostream>
#include <algorithm>
#include <vector>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/SourceVariable.h"
#include "PhysicsTools/MVATrainer/interface/SourceVariableSet.h"

namespace PhysicsTools {

bool SourceVariableSet::append(SourceVariable *var, Magic magic, int offset)
{
	std::vector<PosVar>::iterator pos =
			std::lower_bound(vars.begin(), vars.end(),
			                 var->getName(), PosVar::VarNameLess);

	if (pos != vars.end() && (pos->var == var ||
	                          (pos->var->getSource() == var->getSource() &&
	                           pos->var->getName() == var->getName())))
		return true;

	PosVar item;
	item.pos = offset < 0 ? vars.size() : offset;
	item.var = var;
	item.magic = magic;

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

SourceVariable *SourceVariableSet::find(Magic magic) const
{
	for(std::vector<PosVar>::const_iterator pos = vars.begin();
	    pos != vars.end(); ++pos)
		if (pos->magic == magic)
			return pos->var;

	return 0;
}

std::vector<SourceVariable*> SourceVariableSet::get(bool withMagic) const
{
	std::vector<SourceVariable*> result(vars.size());

	for(std::vector<PosVar>::const_iterator iter = vars.begin();
	    iter != vars.end(); iter++)
		result[iter->pos] = iter->var;

	if (!withMagic) {
		unsigned int pos = vars.size();
		for(std::vector<PosVar>::const_iterator iter = vars.begin();
		    iter != vars.end(); iter++)
			if (iter->magic != kRegular) {
				result.erase(result.begin() +
				             (iter->pos - (iter->pos >= pos)));
				pos = iter->pos;
			}
	}

	return result;
}

} // namespace PhysicsTools
