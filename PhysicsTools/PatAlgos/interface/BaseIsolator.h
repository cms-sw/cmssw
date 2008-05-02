#ifndef PhysicsTools_PatAlgos_interface_BaseIsolator_h
#define PhysicsTools_PatAlgos_interface_BaseIsolator_h

#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include <iomanip>

namespace pat { namespace helper {
class BaseIsolator {
    public:
        typedef edm::ValueMap<float> Isolation;
        BaseIsolator() {}
        BaseIsolator(const edm::ParameterSet &conf) ;
        virtual ~BaseIsolator() {}
        virtual void beginEvent(const edm::Event &event) = 0;
        virtual void endEvent() = 0;

        /// Tests if the value associated to this item is strictly below the cut.
        template<typename AnyRef> bool test(const AnyRef &ref) const { 
            bool ok = (getValue(ref.id(), ref.key()) < cut_);
            try_++; if (!ok) fail_++;
            return ok;
        }
        virtual std::string description() const = 0;
        void print(std::ostream &out) const {
            using namespace std;
            out << "Isolation " << description() << " < " << setw(10) << cut_ << 
                   ": try " << setw(10) << try_ << ", fail " << setw(10) << fail_;
        }
    protected:
        virtual float getValue(const edm::ProductID &id, size_t index) const = 0;
        edm::InputTag input_;
        float cut_;
        mutable uint64_t try_, fail_;
}; // class BaseIsolator
} } // namespaces

inline std::ostream & operator<<(std::ostream &stream, const pat::helper::BaseIsolator &iso) {  
    iso.print(stream); 
    return stream;
}
#endif
