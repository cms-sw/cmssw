#ifndef GENERICDAUHEPMCFILTER_h
#define GENERICDAUHEPMCFILTER_h
// -*- C++ -*-
//
// Package:    GenericDauHepMCFilter
// Class:      GenericDauHepMCFilter
//
/**\class GenericDauHepMCFilter GenericDauHepMCFilter.cc

Description: Filter events using MotherId and ChildrenIds infos

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Daniele Pedrini
//         Created:  Apr 29 2008
// $Id: GenericDauHepMCFilter.h,v 1.2 2010/07/21 04:23:24 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "GeneratorInterface/Core/interface/BaseHepMCFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//

class GenericDauHepMCFilter : public BaseHepMCFilter {
  public:
     GenericDauHepMCFilter(const edm::ParameterSet&);
    ~GenericDauHepMCFilter();

    virtual bool filter(const HepMC::GenEvent* evt);
    
  private:
    // ----------memeber function----------------------

    // ----------member data ---------------------------

    int particleID;
    bool chargeconju;
    int ndaughters;
    std::vector<int> dauIDs;    
    double minptcut;
    double maxptcut;
    double minetacut;
    double maxetacut;
    };
#endif
