#include "PhysicsTools/PatUtils/interface/PatSelectorByFlags.h"

using pat::SelectorByFlags;

SelectorByFlags::SelectorByFlags(const std::string &bitToTest) : 
    mask_(~ pat::Flags::get(bitToTest)) 
{
}

SelectorByFlags::SelectorByFlags(const std::vector<std::string> bitsToTest) : 
    mask_(~ pat::Flags::get(bitsToTest)) 
{
}
