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
// $Id: MVAComputer.cc,v 1.10 2010/01/26 19:40:03 saout Exp $
//
#include <functional>
#include <algorithm>
#include <typeinfo>
#include <iostream>
#include <cstring>
#include <cstddef>

#include <boost/thread.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

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

void MVAComputer::addProcessor(const VarProcessor* proc)
{
  cacheId = getNextMVAComputerCacheId();
  edm::TypeWithDict baseType(typeid(VarProcessor));
  edm::TypeWithDict type(typeid(*proc));
  if (!type.name().size()) {
    throw cms::Exception("MVAComputerCalibration")
      << "Calibration class "
      << typeid(*proc).name()
      << " does not have a dictionary."
      << std::endl;
  }
  edm::TypeWithDict refType(type, kIsConstant | kIsReference);
  edm::FunctionWithDict func = type.functionMemberByName(type.name(), refType.name(), false);
  if (!func) {
    throw cms::Exception("MVAComputerCalibration")
      << "Calibration class "
      << typeid(*proc).name()
      << " does not have a copy constructor."
      << std::endl;
  }
  std::vector<void*> args;
  // Note: This is attempting to compensate if the actual processor type does
  //       not have VarProcessor as its first base class.  Set the further
  //       note below about why this is a waste of time.
  edm::ObjectWithDict src = edm::ObjectWithDict(baseType, const_cast<VarProcessor*>(proc)).castObject(type);
  void* src_p = src.address();
  args.push_back(&src_p);
  void* retval = 0;
  // Note: The type arg does not really matter here.
  edm::ObjectWithDict ret(type, &retval);
  func.invoke(&ret, args);
  // Note: This does *not* properly do the upcast this pointer adjustment,
  //       it assumes that the actual class has VarProcessor as its first
  //       base class.  This mistake was present in the original code.
  processors.push_back(static_cast<VarProcessor*>(retval));
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
