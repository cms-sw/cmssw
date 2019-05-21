#ifndef FASTSIM_PARTICLE_H
#define FASTSIM_PARTICLE_H

#include "DataFormats/Math/interface/LorentzVector.h"

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {
  //! Definition of a generic FastSim Particle which can be propagated through the detector (formerly ParticlePropagator)
  /*!
        Contains all information necessary for the propagation and tracking of a particle:
    */
  class Particle {
  public:
    //! Constructor.
    /*!
            \param pdgId The pdgId of the particle.
            \param position The position of the particle.
            \param momentum The momentum of the particle.
        */
    Particle(int pdgId, const math::XYZTLorentzVector& position, const math::XYZTLorentzVector& momentum)
        : pdgId_(pdgId),
          charge_(-999.),
          position_(position),
          momentum_(momentum),
          remainingProperLifeTimeC_(-999.)  // lifetime in ct
          ,
          simTrackIndex_(-1),
          simVertexIndex_(-1),
          genParticleIndex_(-1),
          isOnForwardLayer_(false),
          isOnLayerIndex_(-1),
          energyDeposit_(0),
          isLooper_(false),
          motherDeltaR_(-1),
          motherPpdId_(0),
          motherSimTrackIndex_(-999) {
      ;
    }

    ////////
    // setters
    //////////

    //! Set the index of the SimTrack of this particle.
    void setSimTrackIndex(int index) { simTrackIndex_ = index; }

    //! Set the index of the origin SimVertex of this particle.
    void setSimVertexIndex(int index) { simVertexIndex_ = index; }

    //! Set index of the particle in the genParticle vector (if applies).
    void setGenParticleIndex(int index) { genParticleIndex_ = index; }

    //! Particle is stable
    void setStable() { remainingProperLifeTimeC_ = -1.; }

    //! Set the particle's remaining proper lifetime if not stable [in ct]
    /*!
        \param remainingProperLifeTimeC Important: defined in units of c*t!
    */
    void setRemainingProperLifeTimeC(double remainingProperLifeTimeC) {
      remainingProperLifeTimeC_ = remainingProperLifeTimeC;
    }

    //! Set the charge of the particle.
    void setCharge(double charge) { charge_ = charge; }

    //! Set layer this particle is currently on
    void setOnLayer(bool isForward, int index) {
      isOnForwardLayer_ = isForward;
      isOnLayerIndex_ = index;
    }

    //! Reset layer this particle is currently on (i.e. particle is not on a layer anyomre)
    void resetOnLayer() { isOnLayerIndex_ = -1; }

    //! Set the energy the particle deposited in the tracker layer that was last hit (ionization).
    /*!
        This energy is then assigned with a SimHit (if any).
        \param energyDeposit The energy loss of the particle in the tracker material layer.
        \sa fastsim::EnergyLoss::interact(fastsim::Particle & particle, const SimplifiedGeometry & layer,std::vector<std::unique_ptr<fastsim::Particle> > & secondaries,const RandomEngineAndDistribution & random)
        \sa fastsim::TrackerSimHitProducer::interact(Particle & particle,const SimplifiedGeometry & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random)
    */
    void setEnergyDeposit(double energyDeposit) { energyDeposit_ = energyDeposit; }

    //! This particle is about to do a loop in the tracker or the direction of the momentum is going inwards.
    /*!
        The TrackerSimHitProducer has the option not to store SimHits in this case, since tracking not possible (just leads to fakes).
        \sa fastsim::TrackerSimHitProducer::interact(Particle & particle,const SimplifiedGeometry & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random)
    */
    void setLooper() { isLooper_ = true; }

    //! Set delta R to mother particle (only necessary if mother and daughter charged).
    /*!
        Needed for FastSim (cheat) tracking: daughter can continue the track of the mother.
        \param motherMomentum The momentum 4-vector of the mother particle.
        \sa fastsim::ParticleManager::addSecondaries(const math::XYZTLorentzVector & vertexPosition, int parentSimTrackIndex, std::vector<std::unique_ptr<Particle> > & secondaries)
    */
    void setMotherDeltaR(const math::XYZTLorentzVector& motherMomentum) {
      motherDeltaR_ = (momentum_.Vect().Unit().Cross(motherMomentum.Vect().Unit())).R();
    }

    //! Set pdgId of the mother particle (only necessary if mother and daughter charged).
    /*!
        Needed for FastSim (cheat) tracking: daughter can continue the track of the mother.
        \param motherPpdId The pdgId of the mother particle.
        \sa fastsim::ParticleManager::addSecondaries(const math::XYZTLorentzVector & vertexPosition, int parentSimTrackIndex, std::vector<std::unique_ptr<Particle> > & secondaries)
    */
    void setMotherPdgId(int motherPpdId) { motherPpdId_ = motherPpdId; }

    //! Set the index of the SimTrack of the mother particle (only necessary if mother and daughter charged).
    /*!
        Needed for FastSim (cheat) tracking: daughter can continue the track of the mother.
        \param id The Id of the SimTrack of the mother particle.
        \sa fastsim::ParticleManager::addSecondaries(const math::XYZTLorentzVector & vertexPosition, int parentSimTrackIndex, std::vector<std::unique_ptr<Particle> > & secondaries)
    */
    void setMotherSimTrackIndex(int id) { motherSimTrackIndex_ = id; }

    //! Reset all information stored about the mother particle.
    void resetMother() {
      motherDeltaR_ = -1;
      motherPpdId_ = 0;
      motherSimTrackIndex_ = -999;
    }

    ////////
    // ordinary getters
    //////////

    //! Return pdgId of the particle.
    int pdgId() const { return pdgId_; }

    //! Return charge of the particle.
    double charge() const { return charge_; }

    //! Return position of the particle.
    const math::XYZTLorentzVector& position() const { return position_; }

    //! Return momentum of the particle.
    const math::XYZTLorentzVector& momentum() const { return momentum_; }

    //! Return the particle's remaining proper lifetime[in ct]
    /*!
        Returns -1 in case particle is stable.
        \return Important: defined in units of c*t!
    */
    double remainingProperLifeTimeC() const { return remainingProperLifeTimeC_; }

    //! Return index of the SimTrack.
    int simTrackIndex() const { return simTrackIndex_; }

    //! Return index of the origin vertex.
    /*!
        Returns -1 for primary vertex.
    */
    int simVertexIndex() const { return simVertexIndex_; }

    //! Return index of the particle in the genParticle vector.
    /*!
        Returns -1 if not a genParticle.
    */
    int genParticleIndex() const { return genParticleIndex_; }

    //! Check if particle is on layer
    bool isOnLayer(bool isForward, int index) { return isOnForwardLayer_ == isForward && isOnLayerIndex_ == index; }

    //! Returns true if particle is considered stable.
    bool isStable() const { return remainingProperLifeTimeC_ == -1.; }

    //! Check if charge of particle was set.
    bool chargeIsSet() const { return charge_ != -999.; }

    //! Check if remaining proper lifetime of particle is set.
    bool remainingProperLifeTimeIsSet() const { return remainingProperLifeTimeC_ != -999.; }

    //! Return Lorentz' gamma factor
    double gamma() const { return momentum().Gamma(); }

    //! Return the energy the particle deposited in the tracker layer that was last hit (ionization).
    /*!
        This energy can then be assigned with a SimHit (if any).
        \sa fastsim::EnergyLoss::interact(fastsim::Particle & particle, const SimplifiedGeometry & layer,std::vector<std::unique_ptr<fastsim::Particle> > & secondaries,const RandomEngineAndDistribution & random)
        \sa fastsim::TrackerSimHitProducer::interact(Particle & particle,const SimplifiedGeometry & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random)
    */
    double getEnergyDeposit() const { return energyDeposit_; }

    //! Check if this particle is about to do a loop in the tracker or the direction of the momentum is going inwards.
    /*!
        The TrackerSimHitProducer has the option not to store SimHits in this case, since tracking not possible (just leads to fakes).
        \sa fastsim::TrackerSimHitProducer::interact(Particle & particle,const SimplifiedGeometry & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random)
    */
    double isLooper() const { return isLooper_; }

    //! Get delta R to mother particle (only makes sense if mother and daughter charged).
    /*!
        Needed for FastSim (cheat) tracking: daughter can continue the track of the mother.
        \sa fastsim::ParticleManager::addSecondaries(const math::XYZTLorentzVector & vertexPosition, int parentSimTrackIndex, std::vector<std::unique_ptr<Particle> > & secondaries)
    */
    double getMotherDeltaR() const { return motherDeltaR_; }

    //! Get pdgIdto mother particle (only makes sense if mother and daughter charged).
    /*!
        Needed for FastSim (cheat) tracking: daughter can continue the track of the mother.
        \sa fastsim::ParticleManager::addSecondaries(const math::XYZTLorentzVector & vertexPosition, int parentSimTrackIndex, std::vector<std::unique_ptr<Particle> > & secondaries)
    */
    int getMotherPdgId() const { return motherPpdId_; }

    //! Get index of simTrack of mother particle (only makes sense if mother and daughter charged).
    /*!
        Needed for FastSim (cheat) tracking: daughter can continue the track of the mother.
        \sa fastsim::ParticleManager::addSecondaries(const math::XYZTLorentzVector & vertexPosition, int parentSimTrackIndex, std::vector<std::unique_ptr<Particle> > & secondaries)
    */
    int getMotherSimTrackIndex() const { return motherSimTrackIndex_; }

    ////////
    // non-const getters
    //////////

    //! Return position of the particle.
    math::XYZTLorentzVector& position() { return position_; }

    //! Return momentum of the particle.
    math::XYZTLorentzVector& momentum() { return momentum_; }

    friend std::ostream& operator<<(std::ostream& os, const Particle& particle);

  private:
    const int pdgId_;                   //!< pdgId of the particle
    double charge_;                     //!< charge of the particle in elemntary units
    math::XYZTLorentzVector position_;  //!< position of the particle
    math::XYZTLorentzVector momentum_;  //!< momentum of the particle
    double remainingProperLifeTimeC_;   //!< remaining proper lifetime of the particle in units of t*c
    int simTrackIndex_;                 //!< index of the simTrack
    int simVertexIndex_;                //!< index of the origin vertex
    int genParticleIndex_;              //!< index of the particle in the vector of genParticles (if applies)
    bool isOnForwardLayer_;             //!< the layer this particle is currently on: is it a ForwardLayer
    int isOnLayerIndex_;                //!< the layer this particle is currently on: the index of the layer
    double energyDeposit_;              //!< energy deposit through ionization in the previous tracker layer
    bool isLooper_;                     //!< this particle is about to do a loop or momentum goes inwards
    double motherDeltaR_;               //!< delta R to mother particle if both charged
    int motherPpdId_;                   //!< pdgId of mother particle if both charged
    int motherSimTrackIndex_;           //!< simTrack index of mother particle if both charged
  };

  //! Some basic output.
  std::ostream& operator<<(std::ostream& os, const Particle& particle);

}  // namespace fastsim

#endif
