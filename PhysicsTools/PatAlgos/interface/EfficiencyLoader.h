#ifndef PhysicsTools_PatAlgos_EfficiencyLoader_h
#define PhysicsTools_PatAlgos_EfficiencyLoader_h

#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/PatCandidates/interface/LookupTableRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


namespace pat { namespace helper {
class EfficiencyLoader {
    public:
        /// Empty constructor
        EfficiencyLoader() {}

        /// Constructor from a PSet
        EfficiencyLoader(const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC) ;

        /// 'true' if this there is at least one efficiency configured
        bool enabled() const { return !names_.empty(); }

        /// To be called for each new event, reads in the ValueMaps for efficiencies
        void newEvent(const edm::Event &event);

        /// Sets the efficiencies for this object, using the reference to the original objects
        template<typename T, typename R>
        void setEfficiencies( pat::PATObject<T> &obj,  const R & originalRef ) const ;

    private:
        std::vector<std::string>   names_;
        std::vector<edm::EDGetTokenT<edm::ValueMap<pat::LookupTableRecord> > > tokens_;
        std::vector<edm::Handle< edm::ValueMap<pat::LookupTableRecord> > > handles_;
}; // class

template<typename T, typename R>
void
EfficiencyLoader::setEfficiencies( pat::PATObject<T> &obj,  const R & originalRef ) const
{
    for (size_t i = 0, n = names_.size(); i < n; ++i) {
        obj.setEfficiency(names_[i], (* handles_[i])[originalRef] );
    }
}

} }

#endif
