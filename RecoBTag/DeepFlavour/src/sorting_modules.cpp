


#include "../interface/sorting_modules.h"
#include <iostream>
namespace sorting{

std::vector<size_t> invertSortingVector(const std::vector<sortingClass<size_t> > & in){
    size_t max=0;
    for(const auto& s:in){
        if(s.get()>max)max=s.get();
    }

    if(max>1e3){
        for(const auto& s:in){
          std::cout << s.get() << std::endl;
        }
        throw std::runtime_error("sorting vector size more than 1k ");
        
    }
    std::vector<size_t> out(max+1,0);
    for(size_t i=0;i<in.size();i++){
        out.at(in.at(i).get())=i;
    }

    return out;
}

}
