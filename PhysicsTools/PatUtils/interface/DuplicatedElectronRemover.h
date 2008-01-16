#ifndef PhysicsTools_PatUtils_DuplicatedElectronRemover_h
#define PhysicsTools_PatUtils_DuplicatedElectronRemover_h

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

#include <memory>
#include <vector>

namespace pat { 
    
    class DuplicatedElectronRemover {
        public:
            DuplicatedElectronRemover() { }
            ~DuplicatedElectronRemover() { }

            std::auto_ptr< std::vector<size_t> > duplicatesToRemove(const std::vector<reco::PixelMatchGsfElectron> &electrons) ;
            
        private:
            
    };
}

#endif
