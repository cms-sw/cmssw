/**
 * \file PedeSteerer.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.11 $
 *  $Date$
 *  (last update by $Author$)
 */

#include "PedeSteerer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "Geometry/Vector/interface/GlobalPoint.h"

#include <fstream>

//___________________________________________________________________________

PedeSteerer::PedeSteerer(AlignableTracker *alignableTracker, AlignmentParameterStore *store,
			 const edm::ParameterSet &config) : 
  mySteerFile(config.getParameter<std::string>("steerFile").c_str(), std::ios::out)
{
  // opens steerFileName as text output file

  if (!mySteerFile.is_open()) {
    edm::LogError("Alignment") << "@SUB=PedeSteerer::PedeSteerer"
			       << "Could not open " << config.getParameter<std::string>("steerFile")
			       << " as output file.";
  }

  this->buildMap(alignableTracker);

  this->fixParameters(store, config.getParameter<edm::ParameterSet>("fixParameters"));
}

//___________________________________________________________________________

PedeSteerer::~PedeSteerer()
{
  // closes file
  mySteerFile.close();
}

//___________________________________________________________________________
/// Return 32-bit unique label for alignable, 0 indicates failure.
/// So far works only within the tracker.
// uint32_t 
unsigned int PedeSteerer::alignableLabel(const Alignable *alignable) const
{
  if (!alignable) return 0;

  AlignableToIdMap::const_iterator position = myAlignableToIdMap.find(alignable);
  if (position != myAlignableToIdMap.end()) {
    return position->second;
  } else {
    //throw cms::Exception("LogicError") 
    edm::LogWarning("LogicError")
      << "@SUB=PedeSteerer::alignableLabel" << "alignable "
      << typeid(*alignable).name() << " not in map";
    return 0;
  }

  /*
// following ansatz does not work since the maximum label allowed by pede is 99 999 999...
//   TrackerAlignableId idProducer;
//   const DetId detId(alignable->geomDetId()); // does not work: only AlignableDet(Unit) has DetId...
//  if (detId.det() != DetId::Tracker) {
  const unsigned int detOffset = 28; // would like to use definition from DetId
  const TrackerAlignableId::UniqueId uniqueId(idProducer.alignableUniqueId(alignable));
  const uint32_t detId = uniqueId.first; // uniqueId is a pair...
  const uint32_t det = detId >> detOffset; // most significant bits are detector part
  if (det != DetId::Tracker) {
    //throw cms::Exception("LogicError") 
    edm::LogWarning("LogicError") << "@SUB=PedeSteerer::alignableLabel "
      << "Expecting DetId::Tracker (=" << DetId::Tracker << "), but found "
      << det << " which would make the pede labels ambigous. "
      << typeid(*alignable).name() << " " << detId;
    return 0;
  }
  // FIXME: Want: const AlignableObjectId::AlignableObjectIdType type = 
  const unsigned int aType = static_cast<unsigned int>(uniqueId.second);// alignable->alignableObjectId();
  if (aType != ((aType << detOffset) >> detOffset)) {
    // i.e. too many bits (luckily we are  not the muon system...)
    throw cms::Exception("LogicError")  << "@SUB=PedeSteerer::alignableLabel "
      << "Expecting alignableTypeId with at most " << 32 - detOffset
      << " bits, but the number is " << aType
      << " which would make the pede labels ambigous.";
    return 0;
  }

  const uint32_t detIdWithoutDet = (detId - (det << detOffset));
  return detIdWithoutDet + (aType << detOffset);
*/
}

//_________________________________________________________________________
unsigned int PedeSteerer::parameterLabel(unsigned int aliLabel, unsigned int parNum) const
{

  return aliLabel+ parNum; // FIXME: check whether alignable has >= parNum + 1 parameters

  /*
  const unsigned int bitOffset = 20;
  const unsigned int patterLength = 3;
  unsigned int aMask = 0;
  for (unsigned int i = 0; i < patterLength; ++i) {
    aMask += (1 << i);
  }
  const unsigned int bitMask = (aMask << bitOffset);

  if (aliLabel & bitMask) {
    throw cms::Exception("LogicError") 
      << "bits to put parNum in are not empty. Mask " << bitMask
      << ", aliLabel " << aliLabel;
  }

  if (parNum != ((parNum << bitOffset) >> bitOffset)) {
    throw cms::Exception("LogicError") 
      << "parNum = " << parNum << " requires more than " << patterLength
      << " bits";
  }

  aliLabel += (parNum << bitOffset);
  return aliLabel;
  */
}

//_________________________________________________________________________
unsigned int PedeSteerer::buildMap(Alignable *highestLevelAli)
{

  myAlignableToIdMap.clear(); // just in case of re-use...
  if (!highestLevelAli) return 0;

  std::vector<Alignable*> allComps;
  allComps.push_back(highestLevelAli);
  highestLevelAli->recursiveComponents(allComps);

  unsigned int id = 1;
  for (std::vector<Alignable*>::const_iterator iter = allComps.begin();
       iter != allComps.end(); ++iter) {
    myAlignableToIdMap.insert(AlignableToIdPair(*iter, id));
    id += RigidBodyAlignmentParameters::N_PARAM; // FIXME: We rely on rigidbody model...
  }

  return allComps.size();
}

//_________________________________________________________________________
unsigned int PedeSteerer::fixParameters(AlignmentParameterStore *store,
					const edm::ParameterSet &config)
{
  // return number of fixed parameters
  // currently fix only full alignables...
  if (!store) return 0;

  const std::vector<double> etaRanges(config.getParameter<std::vector<double> >("etaRanges"));
  const std::vector<double> zRanges  (config.getParameter<std::vector<double> >("zRanges"  ));
  const std::vector<double> rRanges  (config.getParameter<std::vector<double> >("rRanges"  ));
  const std::vector<double> phiRanges(config.getParameter<std::vector<double> >("phiRanges"));

  unsigned int numFixed = 0;

  const std::vector<Alignable*> &alignables = store->alignables();
  for (std::vector<Alignable*>::const_iterator iAli = alignables.begin();
       iAli != alignables.end(); ++iAli) {
    const GlobalPoint position((*iAli)->globalPosition());
    if (this->insideRanges(static_cast<double>(position.eta()), etaRanges) ||
	this->insideRanges(static_cast<double>(position.z()), zRanges) ||
	this->insideRanges(static_cast<double>(position.perp()), rRanges)
	//	|| this->outsidePhiRanges(position.phi(), phiRanges)
	) {
      const AlignmentParameters *params = (*iAli)->alignmentParameters();
      if (!params) {
	edm::LogError("Alignment") << "@SUB=PedeSteerer::fixParameters" 
				   << "no parameters for Alignable in AlignmentParameterStore";
	continue;
      }
      const unsigned int aliLabel = this->alignableLabel(*iAli);
      const unsigned int numSelParams = params->numSelected();
      for (unsigned int iParam = 0; iParam < numSelParams; ++iParam) {
	mySteerFile << this->parameterLabel(aliLabel, iParam) << "  0.0  -1.0";
	if (0) { // debug
	  mySteerFile << " eta " << position.eta() << ", z " << position.z()
		      << ", r " << position.perp() << ", phi " << position.phi();
	}
	mySteerFile << std::endl; // "\n"?
	++numFixed;
      }
    }
  }

  return numFixed;
}

//_________________________________________________________________________
bool PedeSteerer::insideRanges(double value, const std::vector<double> &ranges) const
{
  // might become templated on <double> ?

  //  if (ranges.empty()) return false; // no ranges defined: all is fine

  if (ranges.size()%2 != 0) {
    edm::LogError("Alignment") << "@SUB=PedeSteerer::insideRanges" 
			       << "need even number of entries in ranges instead of "
			       << ranges.size();
    return false;
  }

  for (unsigned int i = 0; i < ranges.size(); i += 2) {
    if (value >= ranges[i] && value < ranges[i+1]) {
      return true;
    }
  }

  return false;
}
