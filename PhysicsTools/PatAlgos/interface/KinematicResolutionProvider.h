#ifndef PhysicsTools_PatAlgos_KinematicResolutionProvider_H
#define PhysicsTools_PatAlgos_KinematicResolutionProvider_H

/**
   \class   KinematicResolutionProvider KinematicResolutionProvider.h "PhysicsTools/PatAlgos/interface/KinematicResolutionProvider.h"

   \brief   Interface for derived classes to provide object resolutions for PAT

   This vitrtual base class is an interface for all derived classes that provide 
   resolution factors for PAT. The following functions need to be implemented by 
   any derived class: 

   * getResolution

   a setup function is provided but might need to be re-implemented. 
*/

namespace reco { class Candidate; }
namespace pat  { class CandKinResolution; }
namespace edm  { class ParameterSet; class EventSetup; }

class KinematicResolutionProvider {

 public:
  virtual ~KinematicResolutionProvider() = default;
  /// everything that needs to be done before the event loop
  virtual void setup(const edm::EventSetup &iSetup) const { }
  /// get a CandKinResolution object from the service; this
  /// function needs to be implemented by any derived class
  virtual pat::CandKinResolution getResolution(const reco::Candidate &c) const = 0;
};

#endif
