
#include "RecoBTag/FeatureTools/interface/sorting_modules.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

namespace btagbtvdeep{

std::vector<std::size_t> invertSortingVector(const std::vector<SortingClass<std::size_t> > & in){
    std::size_t max=0;
    for(const auto& s:in){
        if(s.get()>max)max=s.get();
    }

    std::vector<std::size_t> out(max+1,0);
    for(std::size_t i=0;i<in.size();i++){
        out.at(in[i].get())=i;
    }

    return out;
}

}
