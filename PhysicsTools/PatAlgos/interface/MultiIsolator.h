#ifndef PhysicsTools_PatAlgos_interface_MultiIsolator_h
#define PhysicsTools_PatAlgos_interface_MultiIsolator_h

#include "DataFormats/Common/interface/View.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "PhysicsTools/PatAlgos/interface/BaseIsolator.h"
#include "boost/ptr_container/ptr_vector.hpp"

#include "DataFormats/PatCandidates/interface/Isolation.h"

namespace pat { namespace helper {
class MultiIsolator {
    public:
        typedef std::vector<std::pair<pat::IsolationKeys,float> > IsolationValuePairs;
        MultiIsolator() {}
        MultiIsolator(const edm::ParameterSet &conf, edm::ConsumesCollector && iC, bool cuts=true) ;
        ~MultiIsolator() {}

        // adds an isolator (and takes onwership of the pointer)
        void addIsolator(BaseIsolator *iso, uint32_t mask, pat::IsolationKeys key) ;

        // parses an isolator and adds it to the list
        void addIsolator(const edm::ParameterSet &conf, edm::ConsumesCollector & iC, bool withCut, uint32_t mask, pat::IsolationKeys key) ;

        // Parses out an isolator, and returns a pointer to it.
        // For an empty PSet, it returns a null pointer.
        // You own the returned pointer!
        static BaseIsolator * make(const edm::ParameterSet &conf, edm::ConsumesCollector & iC, bool withCut) ;

        void beginEvent(const edm::Event &event, const edm::EventSetup &eventSetup);
        void endEvent() ;

        template<typename T>
        uint32_t test(const edm::View<T> &coll, int idx) const;

        template<typename T>
        void fill(const edm::View<T> &coll, int idx, IsolationValuePairs& isolations) const ;

        /// Fill Isolation from a Ref, Ptr or RefToBase to the object
        template<typename RefType>
        void fill(const RefType &ref, IsolationValuePairs& isolations) const ;

        void print(std::ostream &out) const ;

        std::string printSummary() const ;

        /// True if it has a non null configuration
        bool enabled() const  { return !isolators_.empty(); }
    private:
        boost::ptr_vector<BaseIsolator> isolators_;
        std::vector<uint32_t>           masks_;
        std::vector<pat::IsolationKeys> keys_;
};

    template<typename T>
    uint32_t
    MultiIsolator::test(const edm::View<T> &coll, int idx) const {
        uint32_t retval = 0;
        edm::RefToBase<T> rb = coll.refAt(idx); // edm::Ptr<T> in a shiny new future to come one remote day ;-)
        for (size_t i = 0, n = isolators_.size(); i < n; ++i) {
            if (!isolators_[i].test(rb)) retval |= masks_[i];
        }
        return retval;
    }

    template<typename RefType>
    void
    MultiIsolator::fill(const RefType &rb, IsolationValuePairs & isolations) const
    {
        isolations.resize(isolators_.size());
        for (size_t i = 0, n = isolators_.size(); i < n; ++i) {
           isolations[i].first  = keys_[i];
           isolations[i].second = isolators_[i].getValue(rb);
        }
    }


    template<typename T>
    void
    MultiIsolator::fill(const edm::View<T> &coll, int idx, IsolationValuePairs & isolations) const
    {
        edm::RefToBase<T> rb = coll.refAt(idx);
        fill(rb, isolations);
    }

}} // namespace

#endif

