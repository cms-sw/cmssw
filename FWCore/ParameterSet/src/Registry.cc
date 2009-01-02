// ----------------------------------------------------------------------
// $Id: Registry.cc,v 1.10 2008/04/29 21:37:49 paterno Exp $
//
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"


namespace edm
{
  namespace pset
  {

    bool
    insertParameterSetIntoRegistry(Registry* reg, ParameterSet const& p)
    {
      ParameterSet tracked_part = p.trackedPart();
      return reg->insertMapped(tracked_part);
    }

    void 
    loadAllNestedParameterSets(Registry* reg, ParameterSet const& main)
    {
      std::vector<ParameterSet> all_main_psets;
      explode(main, all_main_psets);
      std::vector<ParameterSet>::const_iterator i = all_main_psets.begin();
      std::vector<ParameterSet>::const_iterator e = all_main_psets.end();
      for (; i != e; ++i) reg->insertMapped(*i);
      reg->extra().setID(main.id());
    }

    edm::ParameterSetID
    getProcessParameterSetID(Registry const* reg)
    {
      return reg->extra().id();
    }

    void fillMap(Registry* reg, regmap_type& fillme)
    {
      typedef Registry::const_iterator iter;
      fillme.clear();
      for (iter i=reg->begin(), e=reg->end(); i!=e; ++i)
	fillme[i->first].pset_ = i->second.toStringOfTracked();
    }
  } // namespace pset

  edm::ParameterSet getProcessParameterSet()
  {
    edm::pset::Registry* reg = edm::pset::Registry::instance();
    edm::ParameterSetID id = edm::pset::getProcessParameterSetID(reg);

    edm::ParameterSet result;
    if (!reg->getMapped(id, result))
      throw edm::Exception(errors::EventCorruption, "Uknown ParameterSetID")
	<< "Unable to find the ParameterSet for id: "
	<< id
	<< ";\nthis was supposed to be the process ParameterSet\n";

    return result;
  }


} // namespace edm

