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
	for(const auto & var : vars)
		if (var.magic == magic)
			return var.var;

	return 0;
}

std::vector<SourceVariable*> SourceVariableSet::get(bool withMagic) const
{
	std::vector<SourceVariable*> result(vars.size());

	for(const auto & var : vars)
		result[var.pos] = var.var;

	if (!withMagic) {
		unsigned int pos = vars.size();
		for(const auto & var : vars)
			if (var.magic != kRegular) {
				result.erase(result.begin() +
				             (var.pos - (var.pos >= pos)));
				pos = var.pos;
			}
	}

	return result;
}

} // namespace PhysicsTools
