#ifndef PhysicsTools_PatAlgos_KinematicResolutionProvider_H
#define PhysicsTools_PatAlgos_KinematicResolutionProvider_H

namespace reco { class Candidate; }
namespace pat { class CandKinResolution; }
namespace edm { class ParameterSet; class EventSetup; }
class KinematicResolutionProvider {
   public:
        virtual void setup(const edm::EventSetup &iSetup) const { }
        virtual pat::CandKinResolution getResolution(const reco::Candidate &c) const = 0;
};

#endif
