#include "../interface/RooSimultaneousOpt.h"
#include "../interface/CachingNLL.h"

RooAbsReal* 
RooSimultaneousOpt::createNLL(RooAbsData& data, const RooLinkedList& cmdList) 
{
    return new cacheutils::CachingSimNLL(this, &data);
}
