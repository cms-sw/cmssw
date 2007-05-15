#ifndef SourceVariable_h
#define SourceVariable_h

#include <vector>
#include <set>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"

class Source;

class SourceVariable : public PhysicsTools::Variable {
    public:
	SourceVariable(Source *source, PhysicsTools::AtomicId name,
	               PhysicsTools::Variable::Flags flags) :
		PhysicsTools::Variable(name, flags), source(source) {}
	~SourceVariable() {}

	Source *getSource() const { return source; }

    private:
	Source	*source;
};

#endif // SourceVariable_h
