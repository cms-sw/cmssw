// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     BuilderVPSet
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Wed May 18 15:43:02 EDT 2005
// $Id: BuilderVPSet.cc,v 1.4 2006/03/02 22:58:11 paterno Exp $
//

// system include files
#include <sstream>

// user include files
#include "FWCore/ParameterSet/src/BuilderVPSet.h"
#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace std;
//
// constants, enums and typedefs
//
namespace edm {
   namespace pset {
//
// static data member definitions
//

//
// constructors and destructor
//
      BuilderVPSet::BuilderVPSet(std::vector<ParameterSet>& fillme,
                                 const NamedPSets& blocks,
                                 const NamedPSets& psets):main_(fillme),blocks_(blocks),
      psets_(psets) {}

// BuilderVPSet::BuilderVPSet(const BuilderVPSet& rhs)
// {
//    // do actual copying here;
// }

//BuilderVPSet::~BuilderVPSet()
//{
//}

//
// assignment operators
//
// const BuilderVPSet& BuilderVPSet::operator=(const BuilderVPSet& rhs)
// {
//   //An exception safe implementation is
//   BuilderVPSet temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void BuilderVPSet::visitString(const StringNode& n)
{
   // the pset would have already been build, so go locate it
   // to get its ID to store in the current pset array. huh?
   //cout << " n.value_ " << endl;
   NamedPSets::const_iterator itPSet = psets_.find(n.value_);
   if(itPSet == psets_.end()) {
      throw edm::Exception(errors::Configuration,"StringError")
	<< "ParameterSet: problem processing string names.\n"
	<< "Could not find ParameterSet named '"
	<< n.value_ << "' used on line " << n.line;
   }
   main_.push_back(*(itPSet->second));
}

void BuilderVPSet::visitContents(const ContentsNode& n)
{
   boost::shared_ptr<ParameterSet> newPSet = makePSet(*(n.nodes_), 
                                              blocks_,
                                              psets_);
   main_.push_back(*newPSet);
}

//
// const member functions
//

//
// static member functions
//
   }
}
