#include "PhysicsTools/NanoAOD/plugins/NanoAODBaseCrossCleaner.h"

class NanoAODSimpleCrossCleaner : public NanoAODBaseCrossCleaner {
public:
      NanoAODSimpleCrossCleaner(const edm::ParameterSet&p):NanoAODBaseCrossCleaner(p){}
      ~NanoAODSimpleCrossCleaner() override{}

      void objectSelection( const edm::View<pat::Jet> & jets, const edm::View<pat::Muon>  & muons, const edm::View<pat::Electron> & eles,
                                    const edm::View<pat::Tau> & taus, const edm::View<pat::Photon>  & photons,
                                    std::vector<uint8_t> & jetBits, std::vector<uint8_t> & muonBits, std::vector<uint8_t> & eleBits,
                                    std::vector<uint8_t> & tauBits, std::vector<uint8_t> & photonBits) override     {

 	    for(size_t i=0;i<jets.size();i++){
		for(const auto & m : jets[i].overlaps("muons")) {
			if(muonBits[m.key()]) jetBits[i]=0; //prefer muons
		}
                for(const auto & m : jets[i].overlaps("electrons")) {
                        if(eleBits[m.key()]) jetBits[i]=0; //prefer electrons
                }

            }
	}
 
};
DEFINE_FWK_MODULE(NanoAODSimpleCrossCleaner);

