#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "PhysicsTools/PatAlgos/interface/StringResolutionProvider.h"
#include "PhysicsTools/PatAlgos/interface/KinematicResolutionRcd.h"

class StringResolutionProviderESProducer : public edm::ESProducer 
                                         {
        public:
                StringResolutionProviderESProducer() { }
                StringResolutionProviderESProducer(const edm::ParameterSet &iConfig) ;

                std::unique_ptr<KinematicResolutionProvider>  produce(const KinematicResolutionRcd &rcd) ;

        private:
                edm::ParameterSet cfg_;
};

StringResolutionProviderESProducer::StringResolutionProviderESProducer(const edm::ParameterSet &iConfig) :
           cfg_(iConfig) {
   std::string myName = iConfig.getParameter<std::string>("@module_label");
   setWhatProduced(this,myName);
}

std::unique_ptr<KinematicResolutionProvider> 
StringResolutionProviderESProducer::produce(const KinematicResolutionRcd &rcd) {
        return std::make_unique<StringResolutionProvider>(cfg_);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE( StringResolutionProviderESProducer );
