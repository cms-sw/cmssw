#ifndef PhysicsTools_MVATrainer_SourceVariable_h
#define PhysicsTools_MVATrainer_SourceVariable_h

#include <vector>
#include <set>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"

namespace PhysicsTools {

class Source;

class SourceVariable : public Variable {
    public:
	SourceVariable(Source *source, AtomicId name,
	               Variable::Flags flags) :
		Variable(name, flags), source(source) {}
	~SourceVariable() {}

	Source *getSource() const { return source; }

    private:
	Source	*source;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_SourceVariable_h
