//
// $Id: PATObjectFilter.h,v 1.1 2008/03/06 09:23:10 llista Exp $
//

#ifndef PhysicsTools_PatAlgos_PATObjectFilter_h
#define PhysicsTools_PatAlgos_PATObjectFilter_h


#include "PhysicsTools/UtilAlgos/interface/AnySelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/MinNumberSelector.h"
#include "PhysicsTools/UtilAlgos/interface/MaxNumberSelector.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"


namespace pat {

  typedef ObjectCountFilter<edm::View<reco::Candidate>, AnySelector, AndSelector<MinNumberSelector, MaxNumberSelector> > PATCandViewCountFilter;

}


#endif
