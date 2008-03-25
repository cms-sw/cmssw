#ifndef PhysicsTools_PatAlgos_interface_SimpleIsolator_h
#define PhysicsTools_PatAlgos_interface_SimpleIsolator_h

#include "PhysicsTools/PatAlgos/interface/BaseIsolator.h"

namespace pat { namespace helper {
class SimpleIsolator : public BaseIsolator {
    public:
        SimpleIsolator() {}
        SimpleIsolator(const edm::ParameterSet &conf) ;
        virtual ~SimpleIsolator() {}
        virtual void beginEvent(const edm::Event &event) ;
        virtual void endEvent() ;

        virtual std::string description() const { return input_.encode(); }
    protected:
        edm::Handle<Isolation> handle_;
        virtual float getValue(const edm::ProductID &id, size_t index) const {
            return handle_->get(id, index);
        }
}; // class SimpleIsolator
} } // namespaces

#endif
