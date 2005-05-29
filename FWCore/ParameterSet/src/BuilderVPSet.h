#ifndef PARAMETERSET_BUILDERVPARAMETERSET_H
#define PARAMETERSET_BUILDERVPARAMETERSET_H
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     BuilderVPSet
// 
/**\class BuilderVPSet BuilderVPSet.h FWCore/ParameterSet/interface/BuilderVPSet.h

 Description: Visitor used to fill VPSet objects

 Usage:
    Used internally to parse the Nodes created by flex/bison

*/
//
// Author:      Chris Jones
// Created:     Wed May 18 15:38:32 EDT 2005
// $Id: BuilderVPSet.h,v 1.1 2005/05/19 02:42:35 chrjones Exp $
//

// system include files
#include <vector>
#include <map>

// user include files
#include "FWCore/ParameterSet/interface/Nodes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations
namespace edm {
   namespace pset {
      typedef std::map<std::string, boost::shared_ptr<ParameterSet> > NamedPSets;
struct BuilderVPSet : public Visitor
{
   explicit BuilderVPSet(std::vector<ParameterSet>& fillme,
                         const NamedPSets& blocks,
                         const NamedPSets& psets );
   //virtual ~BuilderPSet();
   
   virtual void visitString(const StringNode&);
   virtual void visitContents(const ContentsNode&);
   
   std::vector<ParameterSet>& main_;
   
   const NamedPSets& blocks_;
   const NamedPSets& psets_;
};
   }
}

#endif /* PARAMETERSET_BUILDERVPARAMETERSET_H */
