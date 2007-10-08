#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignSetup.h"
#include "Alignment/CommonAlignment/interface/Counters.h"

#include "Alignment/CommonAlignment/interface/AlignableBuilder.h"

AlignableBuilder::LevelInfo::LevelInfo(align::StructureType type,
				       bool flat,
				       unsigned int maxComponent):
  type_(type),
  flat_(flat),
  maxComponent_(maxComponent)
{
}

AlignableBuilder::AlignableBuilder(align::StructureType moduleType):
  theModuleType(moduleType)
{
}

void AlignableBuilder::addLevelInfo(align::StructureType type,
				    bool flat,
				    unsigned int maxComponent)
{
  theLevelInfos.push_back( LevelInfo(type, flat, maxComponent) );
}

void AlignableBuilder::buildAll() const
{
  build(0, theModuleType); // build the first level above the modules

  for (unsigned int l = 1; l < theLevelInfos.size(); ++l) build(l, theLevelInfos[l - 1].type_);
}

unsigned int AlignableBuilder::maxComponent(unsigned int level) const
{
  unsigned int max = 1;

  for (unsigned int l = level; l < theLevelInfos.size(); ++l)
  {
    max *= theLevelInfos[l].maxComponent_;
  }

  return max;
}

unsigned int AlignableBuilder::index(unsigned int level,
				     align::ID id) const
{
  const LevelInfo& info = theLevelInfos[level];

  if (theLevelInfos.size() - 1 > level)
  {
    return index(level + 1, id) * info.maxComponent_ + Counters::get(info.type_)(id) - 1;
  }

  return Counters::get(info.type_)(id) - 1;
}

void AlignableBuilder::build(unsigned int level,
			     align::StructureType dauType) const
{
  static AlignableObjectId objId;

  const LevelInfo& momInfo = theLevelInfos[level];

  align::StructureType momType = momInfo.type_;

  const align::Alignables& daus = AlignSetup<align::Alignables>::find( objId.typeToName(dauType) );

  unsigned int nDau = daus.size();

  align::Alignables& moms = AlignSetup<align::Alignables>::get( objId.typeToName(momType) );

  moms.reserve(nDau);

  // In order not to depend on the order of the daughter list,
  // we define flags to indicate the existence of a mother;
  // 0 if it hasn't been created.
  // We use vector instead of map for speed.
  align::Alignables tempMoms(maxComponent(level), 0); // init all to 0

  for (unsigned int i = 0; i < nDau; ++i)
  {
    Alignable* dau = daus[i];

    Alignable*& mom = tempMoms[index( level, dau->id() )];

    if (0 == mom)
    { // create new mom with id and rot of 1st dau
      mom = new AlignableComposite( dau->id(), momType, momInfo.flat_ ?
				    dau->globalRotation() : align::RotationType() );
      moms.push_back(mom);
    }

    mom->addComponent(dau);
  }
}
