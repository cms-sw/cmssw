#ifndef Processor_h
#define Processor_h

#include <vector>
#include <string>

#include <xercesc/dom/DOM.hpp>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/ProcessRegistry.h"

#include "PhysicsTools/MVATrainer/interface/Source.h"

class MVATrainer;

class Processor : public Source,
	public PhysicsTools::ProcessRegistry<Processor, PhysicsTools::AtomicId,
	                                     MVATrainer>::Factory {
    public:
	template<typename Instance_t>
	struct Registry {
		typedef typename PhysicsTools::ProcessRegistry<
			Processor,
			PhysicsTools::AtomicId,
			MVATrainer
		>::Registry<Instance_t, PhysicsTools::AtomicId> Type;
	};

	inline Processor(const char *name,
	                 const PhysicsTools::AtomicId *id,
	                 MVATrainer *trainer) :
		Source(*id), name(name), trainer(trainer) {}
	virtual ~Processor() {}

	virtual PhysicsTools::Variable::Flags getDefaultFlags() const
	{ return PhysicsTools::Variable::FLAG_NONE; }

	virtual void
	configure(XERCES_CPP_NAMESPACE_QUALIFIER DOMElement *config) = 0;

	virtual PhysicsTools::Calibration::VarProcessor *getCalib() const = 0;

	virtual void trainBegin() = 0;
	virtual void trainData(const std::vector<double> *values,
	                       bool target) = 0;
	virtual void trainEnd() = 0;

	inline const char *getId() const { return name.c_str(); }

    protected:
	virtual void *requestObject(const std::string &name) const
	{ return 0; }

	std::string	name;
	MVATrainer	*trainer;
};

#endif // Processor_h
