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
// $Id: MVAComputer.cc,v 1.4.2.1 2008/02/15 17:23:39 saout Exp $
//
#include <functional>
#include <algorithm>
#include <typeinfo>
#include <iostream>
#include <cstring>
#include <cstddef>
#include <string>
#include <vector>
#include <map>

#include <boost/thread.hpp>

#include <Reflex/Reflex.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

namespace PhysicsTools {
namespace Calibration {

std::string VarProcessor::getInstanceName() const
{
	static const char prefix[] = "PhysicsTools::Calibration::";
	std::string type = ROOT::Reflex::Tools::Demangle(typeid(*this));
	if (type.size() <= sizeof prefix - 1 ||
	    type.substr(0, sizeof prefix - 1) != prefix)
		throw cms::Exception("MVAComputerCalibration")
			<< "getInstanceName failed for "
			<< typeid(*this).name() << "." << std::endl;

	return type.substr(sizeof prefix - 1);
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

MVAComputer::~MVAComputer()
{
}

std::vector<VarProcessor*> MVAComputer::getProcessors() const
{
	std::vector<VarProcessor*> procs;

	ROOT::Reflex::Type thisType =
				ROOT::Reflex::GetType<MVAComputer>();
	ROOT::Reflex::Type baseType = ROOT::Reflex::GetType<VarProcessor>();

	ROOT::Reflex::Object thisObj(thisType,
			const_cast<void*>(static_cast<const void*>(this)));

	for(ROOT::Reflex::Member_Iterator iter = thisType.DataMember_Begin();
	    iter != thisType.DataMember_End(); iter++) {
		const ROOT::Reflex::Type &type = iter->TypeOf().FinalType();

		if (!type.IsTemplateInstance() ||
		    std::strncmp(type.Name_c_str(), "std::vector<", 12) != 0,
		    type.TemplateArgumentSize() != 1)
			continue;

		ROOT::Reflex::Type itemType = type.TemplateArgumentAt(0);
		bool found = false;
		for(ROOT::Reflex::Base_Iterator iter2 = itemType.Base_Begin();
		    iter2 != itemType.Base_End(); iter2++) {
			ROOT::Reflex::Type subType = iter2->ToType();
			if (subType.Id() == baseType.Id()) {
				found = true;
				break;
			}
		}

		if (!found)
			continue;

		ROOT::Reflex::Object obj = iter->Get(thisObj);
		ROOT::Reflex::Object sizeObj = obj.Invoke("size");
		std::size_t size = ROOT::Reflex::Object_Cast<std::size_t>(
							obj.Invoke("size"));

		for(std::size_t i = 0; i < size; i++) {
			ROOT::Reflex::Object item = obj.Invoke("at", i);
			procs.push_back(static_cast<VarProcessor*>(
					item.CastObject(baseType).Address()));
		}
	}

	std::vector<VarProcessor*> result;
	for(std::vector<unsigned int>::const_iterator iter =
							processors_.begin();
	    iter != processors_.end(); iter++) {
		if (*iter >= procs.size())
			throw cms::Exception("MVAComputerCalibration")
				<< "Inconsistent calibration data."
				<< std::endl;

		result.push_back(procs[*iter]);
	}

	return result;
}

void MVAComputer::addProcessor(const VarProcessor *proc)
{
	cacheId = getNextMVAComputerCacheId();

	ROOT::Reflex::Type thisType =
				ROOT::Reflex::GetType<MVAComputer>();
	ROOT::Reflex::Type baseType = ROOT::Reflex::GetType<VarProcessor>();
	ROOT::Reflex::Type type =
				ROOT::Reflex::Type::ByTypeInfo(typeid(*proc));

	if (!type.Name().size())
		throw cms::Exception("MVAComputerCalibration")
			<< "Calibration class " << typeid(*proc).name()
			<< " not registered with ROOT::Reflex."
			<< std::endl;

	ROOT::Reflex::Object thisObj(thisType, this);
	ROOT::Reflex::Object obj =
		ROOT::Reflex::Object(baseType, const_cast<void*>(
			static_cast<const void*>(proc))).CastObject(type);

	unsigned int idx = 0;

	for(ROOT::Reflex::Member_Iterator iter = thisType.DataMember_Begin();
	    iter != thisType.DataMember_End(); iter++) {
		const ROOT::Reflex::Type &iType = iter->TypeOf().FinalType();

		if (!iType.IsTemplateInstance() ||
		    std::strncmp(iType.Name_c_str(), "std::vector<", 12) != 0,
		    iType.TemplateArgumentSize() != 1)
			continue;

		ROOT::Reflex::Type itemType = iType.TemplateArgumentAt(0);
		bool found = false;
		for(ROOT::Reflex::Base_Iterator iter2 = itemType.Base_Begin();
		    iter2 != itemType.Base_End(); iter2++) {
			ROOT::Reflex::Type subType = iter2->ToType();
			if (subType.Id() == baseType.Id()) {
				found = true;
				break;
			}
		}

		if (!found)
			continue;

		ROOT::Reflex::Object iObj = iter->Get(thisObj);
		ROOT::Reflex::Object sizeObj = iObj.Invoke("size");
		std::size_t size = ROOT::Reflex::Object_Cast<std::size_t>(
							iObj.Invoke("size"));

		idx += size;

		if (itemType.Id() == type.Id()) {
			for(std::vector<unsigned int>::iterator iter2 =
							processors_.begin();
			    iter2 != processors_.end(); iter2++) {
				if (*iter2 >= idx)
					(*iter2)++;
			}

			iObj.Invoke("push_back",
				ROOT::Reflex::Tools::MakeVector<void*>(
							obj.Address()));

			processors_.push_back(idx);
			return;
		}
	}

	throw cms::Exception("MVAComputerCalibration")
		<< "Calibration class " << type.Name()
		<< " not registered with Calibration::MVAComputer."
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
