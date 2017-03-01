#ifndef PhysicsTools_PatAlgos_KinResolutionsLoader_h
#define PhysicsTools_PatAlgos_KinResolutionsLoader_h

#include "DataFormats/PatCandidates/interface/PATObject.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "PhysicsTools/PatAlgos/interface/KinematicResolutionProvider.h"


namespace pat { namespace helper {
class KinResolutionsLoader {
    public:
        /// Empty constructor
        KinResolutionsLoader() {}

        /// Constructor from a PSet
        KinResolutionsLoader(const edm::ParameterSet &iConfig) ;

        /// 'true' if this there is at least one efficiency configured
        bool enabled() const { return !patlabels_.empty(); }
     
        /// To be called for each new event, reads in the EventSetup object 
        void newEvent(const edm::Event &event, const edm::EventSetup &setup);

        /// Sets the efficiencies for this object, using the reference to the original objects
        template<typename T>
        void setResolutions( pat::PATObject<T> &obj ) const ;

        /// Method for documentation and validation of PSet
        static void fillDescription(edm::ParameterSetDescription & iDesc);
    private:
        /// Labels of the resolutions in PAT
        std::vector<std::string>   patlabels_;
        /// Labels of the KinematicResolutionProvider in the EventSetup
        std::vector<std::string>   eslabels_;
        /// Handles to the EventSetup
        std::vector<edm::ESHandle<KinematicResolutionProvider> > handles_;
}; // class

template<typename T>
void
KinResolutionsLoader::setResolutions( pat::PATObject<T> &obj ) const
{
    for (size_t i = 0, n = patlabels_.size(); i < n; ++i) {
        obj.setKinResolution( handles_[i]->getResolution(obj), patlabels_[i]);
    }
}

} }

#endif
