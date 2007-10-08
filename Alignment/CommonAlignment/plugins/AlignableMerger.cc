#include "Alignment/CommonAlignment/interface/AlignSetup.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/plugins/AlignableMerger.h"

AlignableMerger::AlignableMerger(const edm::ParameterSet& cfg):
  theInLists( cfg.getParameter<Strings>("inLists") ),
  theOutList( cfg.getParameter<std::string>("@module_label") )
{
}

void AlignableMerger::analyze(const edm::Event&,
			      const edm::EventSetup&)
{
  align::Alignables& out = AlignSetup<align::Alignables>::get(theOutList);

  if (out.size() > 0)
  {
    edm::LogWarning("AlignableMerger")
      << theOutList << " already exists. Input lists will be appended.";
  }

  for (unsigned int i = 0; i < theInLists.size(); ++i)
  {
    const align::Alignables& in = AlignSetup<align::Alignables>::find(theInLists[i]);

    out.insert( out.end(), in.begin(), in.end() );
  }
}
