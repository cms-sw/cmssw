// -*- C++ -*-
//
// Package:     PhysicsToolsObjects
// Class  :     MVAComputer
// 

// Implementation:
//     getProcessor() and addProcessor() methods to add processors to
//     the discriminator computer calibration object. POOL doesn't support
//     polymorph pointers, so this is implemented using multiple containers
//     for each possible sub-class and an index array from which the
//     array of pointers can be reconstructed. In order to avoid having
//     to handle each sub-class container by hand here, the generated
//     reflex dictionary is used to find and read/write the std::vector<...>
//     containers for the individual classes in the private data members.
//     So changes can be solely done in the header files and does not leave
//     a trail elsewhere.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: MVAComputer.cc,v 1.14 2013/01/25 17:01:38 wmtan Exp $
//
#include <functional>
#include <algorithm>
#include <typeinfo>
#include <iostream>
#include <cstring>
#include <cstddef>

#include <boost/thread.hpp>

#include <Reflex/Reflex.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

namespace PhysicsTools {
namespace Calibration {

std::string VarProcessor::getInstanceName() const
{
	static const char prefix[] = "PhysicsTools::Calibration::";
        edm::TypeID typeID(typeid(*this));
	std::string type(typeID.className());
	if (type.size() <= sizeof prefix - 1 ||
	    type.substr(0, sizeof prefix - 1) != prefix)
		throw cms::Exception("MVAComputerCalibration")
			<< "getInstanceName failed for "
			<< typeid(*this).name() << "." << std::endl;

	return type.substr(sizeof prefix - 1);
}

std::string ProcExternal::getInstanceName() const
{
	return method;
}

static MVAComputer::CacheId getNextMVAComputerCacheId()
{
	static boost::mutex mutex;
	static MVAComputer::CacheId nextCacheId = 0;

	boost::mutex::scoped_lock scoped_lock(mutex);
	return ++nextCacheId;
}

MVAComputer::MVAComputer() :
	cacheId(getNextMVAComputerCacheId())
{
}

MVAComputer::MVAComputer(const MVAComputer &orig) :
	inputSet(orig.inputSet),
	output(orig.output),
	cacheId(orig.cacheId)
{
	for(std::vector<VarProcessor*>::const_iterator iter =
						orig.processors.begin();
	    iter != orig.processors.end(); ++iter)
		addProcessor(*iter);
}

MVAComputer::~MVAComputer()
{
	for(std::vector<VarProcessor*>::iterator iter = processors.begin();
	    iter != processors.end(); ++iter)
		delete *iter;
	processors.clear();
}

MVAComputer &MVAComputer::operator = (const MVAComputer &orig)
{
	inputSet = orig.inputSet;
	output = orig.output;
	cacheId = orig.cacheId;

	for(std::vector<VarProcessor*>::iterator iter = processors.begin();
	    iter != processors.end(); ++iter)
		delete *iter;
	processors.clear();

	for(std::vector<VarProcessor*>::const_iterator iter =
						orig.processors.begin();
	    iter != orig.processors.end(); ++iter)
		addProcessor(*iter);

	return *this;
}

std::vector<VarProcessor*> MVAComputer::getProcessors() const
{
	return processors;
}

void MVAComputer::addProcessor(const VarProcessor *proc)
{
	cacheId = getNextMVAComputerCacheId();

	ROOT::Reflex::Type baseType = ROOT::Reflex::GetType<VarProcessor>();
	ROOT::Reflex::Type type =
				ROOT::Reflex::Type::ByTypeInfo(typeid(*proc));
	ROOT::Reflex::Type refType(type,
				ROOT::Reflex::CONST & ROOT::Reflex::REFERENCE);
	if (!type.Name().size())
		throw cms::Exception("MVAComputerCalibration")
			<< "Calibration class " << typeid(*proc).name()
			<< " not registered with ROOT::Reflex."
			<< std::endl;

	ROOT::Reflex::Object obj =
		ROOT::Reflex::Object(baseType, const_cast<void*>(
			static_cast<const void*>(proc))).CastObject(type);

	// find and call copy constructor
	for(ROOT::Reflex::Member_Iterator iter = type.FunctionMember_Begin();
	    iter != type.FunctionMember_End(); iter++) {
		const ROOT::Reflex::Type &ctor = iter->TypeOf();
		if (!iter->IsConstructor() ||
		    ctor.FunctionParameterSize() != 1 ||
		    ctor.FunctionParameterAt(0).Id() != refType.Id())
			continue;

		ROOT::Reflex::Object copy = type.Construct(ctor,
			ROOT::Reflex::Tools::MakeVector<void*>(obj.Address()));

		processors.push_back(static_cast<VarProcessor*>(copy.Address()));
		return;
	}

	throw cms::Exception("MVAComputerCalibration")
			<< "Calibration class " << typeid(*proc).name()
			<< " has no copy ctor registered with ROOT::Reflex."
			<< std::endl;
}

static MVAComputerContainer::CacheId getNextMVAComputerContainerCacheId()
{
	static MVAComputerContainer::CacheId nextCacheId = 0;
	return ++nextCacheId;
}

MVAComputerContainer::MVAComputerContainer() :
	cacheId(getNextMVAComputerContainerCacheId())
{
}

MVAComputer &MVAComputerContainer::add(const std::string &label)
{
	cacheId = getNextMVAComputerContainerCacheId();

	entries.push_back(std::make_pair(label, MVAComputer()));
	return entries.back().second;
}

namespace {
	struct Comparator :
		public std::unary_function<const std::string&, bool> {

		inline Comparator(const std::string &label) : label(label) {}

		inline bool
		operator () (const MVAComputerContainer::Entry &entry) const
		{ return entry.first == label; }

		const std::string &label;
	};
}

const MVAComputer &MVAComputerContainer::find(const std::string &label) const
{
	std::vector<Entry>::const_iterator pos =
				std::find_if(entries.begin(), entries.end(),
				             Comparator(label));

	if (pos == entries.end())
		throw cms::Exception("MVAComputerCalibration")
			<< "Calibration record " << label
			<< " not found in MVAComputerContainer." << std::endl;

	return pos->second;
}

} // namespace Calibration
} // namespace PhysicsTools
