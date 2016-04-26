#include "PhysicsTools/Heppy/interface/TriggerBitChecker.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include <assert.h>
#include <iostream>

namespace heppy {

TriggerBitChecker::TriggerBitChecker(const std::string &path) : paths_(1,returnPathStruct(path)) { rmstar(); }

TriggerBitChecker::TriggerBitChecker(const std::vector<std::string> &paths) : paths_(paths.size()) { 
    for(size_t i = 0; i < paths.size(); ++ i) paths_[i] = returnPathStruct(paths[i]);
    rmstar(); 
}

TriggerBitChecker::pathStruct TriggerBitChecker::returnPathStruct(const std::string &path) const {
    pathStruct newPathStruct(path);
    if( path[0] > 48 /*'0'*/ && path[0] <= 57 /*'9'*/ ) {
        newPathStruct.first = atoi(path.substr(0,path.find('-')).c_str());
        newPathStruct.last = atoi(path.substr(path.find('-')+1,path.find(':')-path.find('-')-1).c_str());
        newPathStruct.pathName = path.substr(path.find(':')+1);
    }
    return newPathStruct;
}

bool TriggerBitChecker::check(const edm::EventBase &event, const edm::TriggerResults &result) const {
    if (result.parameterSetID() != lastID_) { syncIndices(event, result); lastID_ = result.parameterSetID(); }
    for (std::vector<unsigned int>::const_iterator it = indices_.begin(), ed = indices_.end(); it != ed; ++it) {
        if (result.accept(*it)) return true;
    }
    return false;
}

bool TriggerBitChecker::check_unprescaled(const edm::EventBase &event, const edm::TriggerResults &result_tr, const pat::PackedTriggerPrescales &result) const {
    if (result_tr.parameterSetID() != lastID_) { syncIndices(event, result_tr); lastID_ = result_tr.parameterSetID(); }
    bool outcome = true;
    for (std::vector<unsigned int>::const_iterator it = indices_.begin(), ed = indices_.end(); it != ed; ++it) {
        if (result.getPrescaleForIndex(*it)!=1) {outcome = false; break;}
    }
    return outcome; // true only if all paths are unprescaled
}                                                                                                                                                                 

int TriggerBitChecker::getprescale(const edm::EventBase &event, const edm::TriggerResults &result_tr, const pat::PackedTriggerPrescales &result) const {
    if (result_tr.parameterSetID() != lastID_) { syncIndices(event, result_tr); lastID_ = result_tr.parameterSetID(); }
        if (indices_.size() == 0 ){
	  //            std::cout << " trying to check an inexistent trigger" << std::endl;
            return  -999;                                                                                            
        }
        if (indices_.size() > 1 ){
            std::cout << " trying to get prescale for multiple trigger objects at the same time" << std::endl;
            assert(0);                                                                                            
        }

    return result.getPrescaleForIndex(*(indices_.begin())) ;
}                                                                                                                                                                 


void TriggerBitChecker::syncIndices(const edm::EventBase &event, const edm::TriggerResults &result) const {
    indices_.clear();
    const edm::TriggerNames &names = event.triggerNames(result);
    std::vector<pathStruct>::const_iterator itp, bgp = paths_.begin(), edp = paths_.end();
    for (size_t i = 0, n = names.size(); i < n; ++i) {
        const std::string &thispath = names.triggerName(i);
        for (itp = bgp; itp != edp; ++itp) {
            if (thispath.find(itp->pathName) == 0 && event.id().run() >= itp->first && event.id().run() <= itp->last) indices_.push_back(i);
        }
    }
}

void TriggerBitChecker::rmstar() {
    std::vector<pathStruct>::iterator itp, bgp = paths_.begin(), edp = paths_.end();
    for (itp = bgp; itp != edp; ++itp) {
        std::string::size_type idx = itp->pathName.find("*");
        if (idx != std::string::npos) itp->pathName.erase(idx);
    }
}
}
