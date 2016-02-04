#ifndef PhysicsTools_MVAComputer_MVAModuleHelper_h
#define PhysicsTools_MVAComputer_MVAModuleHelper_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     MVAModuleHelper
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: MVAModuleHelper.h,v 1.3 2011/04/20 07:07:37 kukartse Exp $
//

#include <functional>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#include <boost/bind.hpp>

#include "FWCore/Framework/interface/EventSetup.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

namespace PhysicsTools {

/** \class MVAModuleHelperDefaultFiller
 *
 * \short Default template for MVAModuleHelper "Filler" template argument
 *
 * Simply calls a "double compute(const AtomicID &name) const" method
 * on the object for each variable requested.
 *
 ************************************************************/
template<typename Object>
struct MVAModuleHelperDefaultFiller {
	MVAModuleHelperDefaultFiller(const PhysicsTools::AtomicId &name) {}

	double operator()(const Object &object,
	                  const PhysicsTools::AtomicId &name)
	{ return object.compute(name); }
};

/** \class MVAModuleHelper
 *
 * \short Template for automated variable collection and MVA computation in EDM modules
 *
 * The class MVAModuleHelper can be embedded in EDM modules.  It automatically
 * collects the variables listed in the MVA training description using type
 * traits and passes them on to the computer.  The calibration and or trainer
 * is automatically collected from the EventSetup.
 *
 ************************************************************/
template<class Record, typename Object,
         class Filler = MVAModuleHelperDefaultFiller<Object> >
class MVAModuleHelper {
    public:
	MVAModuleHelper(const std::string &label) : label(label) {}
	MVAModuleHelper(const MVAModuleHelper &orig) : label(orig.label) {}
	~MVAModuleHelper() {}

	void setEventSetup(const edm::EventSetup &setup);
	void setEventSetup(const edm::EventSetup &setup, const char *esLabel);

	double operator()(const Object &object) const;

	void train(const Object &object, bool target, double weight = 1.0) const;

    private:
	void init(const PhysicsTools::Calibration::MVAComputerContainer *container);

	const std::string		label;
	PhysicsTools::MVAComputerCache	cache;

	class Value {
	    public:
	    	Value(const std::string &name) :
	    		name(name), filler(name) {}
	    	Value(const std::string &name, double value) :
			name(name), filler(name), value(value) {}

		inline bool update(const Object &object) const
		{
			value = filler(object, name);
			return !std::isfinite(value);
		}

		PhysicsTools::AtomicId getName() const { return name; }
		double getValue() const { return value; }

	    private:
		PhysicsTools::AtomicId		name;
		Filler				filler;

		mutable double			value;
	};

	std::vector<Value>			values;
};

template<class Record, typename Object, class Filler>
void MVAModuleHelper<Record, Object, Filler>::setEventSetup(
						const edm::EventSetup &setup)
{
	edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer> handle;
	setup.get<Record>().get(handle);
	const PhysicsTools::Calibration::MVAComputerContainer *container = handle.product();
	if (cache.update(container, label.c_str()) && cache)
		init(container);
}

template<class Record, typename Object, class Filler>
void MVAModuleHelper<Record, Object, Filler>::setEventSetup(
			const edm::EventSetup &setup, const char *esLabel)
{
	edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer> handle;
	setup.get<Record>().get(esLabel, handle);
	const PhysicsTools::Calibration::MVAComputerContainer *container = handle.product();
	if (cache.update(container, label.c_str()) && cache)
		init(container);
}

template<class Record, typename Object, class Filler>
void MVAModuleHelper<Record, Object, Filler>::init(
	const PhysicsTools::Calibration::MVAComputerContainer *container)
{
	const std::vector<PhysicsTools::Calibration::Variable> &vars =
					container->find(label).inputSet;
	values.clear();
	for(std::vector<PhysicsTools::Calibration::Variable>::const_iterator
			iter = vars.begin(); iter != vars.end(); ++iter)
		if (std::strncmp(iter->name.c_str(), "__", 2) != 0)
			values.push_back(Value(iter->name));
}

template<class Record, typename Object, class Filler>
double MVAModuleHelper<Record, Object, Filler>::operator()(
						const Object &object) const
{
	std::for_each(values.begin(), values.end(),
	              boost::bind(&Value::update, _1, object));
	return cache->eval(values);
}

template<class Record, typename Object, class Filler>
void MVAModuleHelper<Record, Object, Filler>::train(
		const Object &object, bool target, double weight) const
{
	static const PhysicsTools::AtomicId kTargetId("__TARGET__");
	static const PhysicsTools::AtomicId kWeightId("__WEIGHT__");

	if (!cache)
		return;

	using boost::bind;
	if (std::accumulate(values.begin(), values.end(), 0,
	                    bind(std::plus<int>(), _1,
	                         bind(&Value::update, _2, object))))
		return;

	PhysicsTools::Variable::ValueList list;
	list.add(kTargetId, target);
	list.add(kWeightId, weight);
	for(typename std::vector<Value>::const_iterator iter = values.begin();
	    iter != values.end(); ++iter)
		list.add(iter->getName(), iter->getValue());

	cache->eval(list);
}

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_MVAModuleHelper_h
