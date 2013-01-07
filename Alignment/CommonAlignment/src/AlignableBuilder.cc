#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

#include "Alignment/CommonAlignment/interface/AlignableBuilder.h"


//__________________________________________________________________________________________________
AlignableBuilder::LevelInfo::LevelInfo( align::StructureType type,
                                        bool flat,
                                        unsigned int maxComponent ) :
  type_(type),
  flat_(flat),
  maxComponent_(maxComponent)
{
}

//__________________________________________________________________________________________________
AlignableBuilder::AlignableBuilder(align::StructureType moduleType, Counters& counters, const TrackerTopology* tTopo):
  theModuleType(moduleType),
  theCounters(counters),
  theTopology(tTopo)
{
}

//__________________________________________________________________________________________________
void AlignableBuilder::addLevelInfo(align::StructureType type,
				    bool flat,
				    unsigned int maxComponent)
{
  theLevelInfos.push_back( LevelInfo(type, flat, maxComponent) );
}

//__________________________________________________________________________________________________
void AlignableBuilder::buildAll( AlignSetup<align::Alignables>& setup ) const
{
  build(0, theModuleType, setup); // build the first level above the modules

  for (unsigned int l = 1; l < theLevelInfos.size(); ++l) 
    build(l, theLevelInfos[l - 1].type_, setup);
}

//__________________________________________________________________________________________________
unsigned int AlignableBuilder::maxComponent(unsigned int level) const
{
  unsigned int max = 1;

  for (unsigned int l = level; l < theLevelInfos.size(); ++l)
  {
    max *= theLevelInfos[l].maxComponent_;
  }

  return max;
}

//__________________________________________________________________________________________________
unsigned int AlignableBuilder::index( unsigned int level, align::ID id, const TrackerTopology* tTopo) const
{
  const LevelInfo& info = theLevelInfos[level];

  if (theLevelInfos.size() - 1 > level)
  {
    return index(level + 1, id, tTopo) * info.maxComponent_ + theCounters.get(info.type_)(id, tTopo) - 1;
  }

  return theCounters.get(info.type_)(id, tTopo) - 1;
}


//__________________________________________________________________________________________________
void AlignableBuilder::build( unsigned int level, align::StructureType dauType,
                              AlignSetup<align::Alignables>& setup ) const
{
  const LevelInfo& momInfo = theLevelInfos[level];

  align::StructureType momType = momInfo.type_;

  const align::Alignables& daus = setup.find( AlignableObjectId::idToString(dauType) );

  unsigned int nDau = daus.size();

  align::Alignables& moms = setup.get( AlignableObjectId::idToString(momType) );

  moms.reserve(nDau);

  // In order not to depend on the order of the daughter list,
  // we define flags to indicate the existence of a mother;
  // 0 if it hasn't been created.
  // We use vector instead of map for speed.
  align::Alignables tempMoms(maxComponent(level), 0); // init all to 0

  for (unsigned int i = 0; i < nDau; ++i)
  {
    Alignable* dau = daus[i];

    Alignable*& mom = tempMoms[index( level, dau->id(), theTopology )];

    if (0 == mom)
    { 
      // create new mom with id and rot of 1st dau
      if ( momInfo.flat_ )
        mom = new AlignableComposite( dau->id(), momType, dau->globalRotation() );
      else
        mom = new AlignableComposite( dau->id(), momType, align::RotationType() );

      moms.push_back(mom);
    }

    mom->addComponent(dau);
  }
}
