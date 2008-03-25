#ifndef PhysicsTools_PatAlgos_interface_MultiIsolator_h
#define PhysicsTools_PatAlgos_interface_MultiIsolator_h

#include "DataFormats/Common/interface/View.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "PhysicsTools/PatAlgos/interface/BaseIsolator.h"
#include "boost/ptr_container/ptr_vector.hpp"

namespace pat { namespace helper {
class MultiIsolator {
    public:
        MultiIsolator() {}
        MultiIsolator(const edm::ParameterSet &conf) ;
        ~MultiIsolator() {}

        // adds an isolator (and takes onwership of the pointer)
        void addIsolator(BaseIsolator *iso, uint32_t mask) ;

         // parses an isolator and adds it to the list
        void addIsolator(const edm::ParameterSet &conf, uint32_t mask) ;
       
        void beginEvent(const edm::Event &event);
        void endEvent() ; 

        template<typename T>
        uint32_t test(const edm::View<T> &coll, int idx);

        void print(std::ostream &out) const ;
    private:
        boost::ptr_vector<BaseIsolator> isolators_;
        std::vector<uint32_t>           masks_;

};

    template<typename T>
    uint32_t 
    MultiIsolator::test(const edm::View<T> &coll, int idx) {
        uint32_t retval = 0;
        edm::RefToBase<T> rb = coll->refAt(idx); // edm::Ptr<T> in a shiny new future to come one remote day ;-)
        for (size_t i = 0, n = isolators_.size(); i < n; ++i) {
            if (!isolators_[i].test(rb)) retval != masks_[i];
        }
        return retval;
    }
}} // namespace

#endif
 
