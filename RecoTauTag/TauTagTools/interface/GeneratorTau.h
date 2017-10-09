#ifndef RECOTAUTAG_TAUMVADISCRIMINATOR_GENERATORTAU
#define RECOTAUTAG_TAUMVADISCRIMINATOR_GENERATORTAU


/*
 * Class GeneratorTau
 *
 * Tool for retrieving visible decay products and determining PDG-style tau decay mode
 * from Pythia output (in CMS genParticle format)
 *
 * Author: Evan K. Friis, UC Davis; friis@physics.ucdavis.edu
 *
 * with code and contributions from Ricardo Vasquez Sierra and Christian Veelken, UC Davis
 *
 */

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "CommonTools/Utils/interface/Angle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <vector>

typedef math::XYZTLorentzVector LorentzVector;

class GeneratorTau : public reco::GenParticle {
   public:

      enum tauDecayModeEnum {kElectron, kMuon,
         kOneProng0pi0, kOneProng1pi0, kOneProng2pi0,
         kThreeProng0pi0, kThreeProng1pi0,
         kOther, kUndefined};

      //default constructor
      GeneratorTau(const reco::GenParticle& input):GenParticle(input){
        theLeadTrack_ = NULL;
      };
      GeneratorTau();

      ~GeneratorTau(){};

      std::vector<const reco::Candidate*>         getGenChargedPions() const;
      std::vector<const reco::Candidate*>         getGenGammas() const;
      std::vector<const reco::Candidate*>         getGenNeutralPions() const;
      std::vector<const reco::Candidate*>         getStableDecayProducts() const;
      std::vector<const reco::Candidate*>         getGenNu() const;

      tauDecayModeEnum                          getDecayType() const {return theDecayMode_;};
      bool                                      isFinalStateTau() const {return aFinalStateTau_;};

      void                                      init(); //called to update class after downcasting

      std::vector<LorentzVector>                     getChargedPions() const;
      std::vector<LorentzVector>                     getGammas() const;
      LorentzVector                             getVisibleFourVector() const;
      std::vector<LorentzVector>                     getVisibleFourVectors() const;
      const reco::Candidate*                    getLeadTrack() const;
      const reco::GenParticle*         findLeadTrack();

      float                                    getVisNuAngle() const;
      float                                    getOpeningAngle(const std::vector<const reco::GenParticle*>& aCollection) const;
      float                                    getChargedOpeningAngle() const;
      float                                    getGammaOpeningAngle() const;

      void                                      decayToPDGClassification(const reco::GenParticle*, std::vector<const reco::GenParticle *>&);
      void                                      computeStableDecayProducts(const reco::GenParticle*, std::vector<const reco::GenParticle *>&);
      tauDecayModeEnum                          computeDecayMode(const reco::GenParticle*);
      LorentzVector                             convertHepMCFourVec(const reco::GenParticle* theParticle);
      std::vector<LorentzVector>                     convertMCVectorToLorentzVectors(const std::vector<const reco::GenParticle*>& theList) const;

   private:

      Angle<LorentzVector>                              angleFinder;
      DeltaR<LorentzVector>                             deltaRFinder;


      //only fill these with stable particles
      std::vector<const reco::GenParticle*>         visibleDecayProducts_;
      std::vector<const reco::GenParticle*>         genChargedPions_;
      std::vector<const reco::GenParticle*>         genNeutralPions_;
      std::vector<const reco::GenParticle*>         genGammas_;
      std::vector<const reco::GenParticle*>         stableDecayProducts_;
      std::vector<const reco::GenParticle*>         genNus_;
      const reco::GenParticle*                      theLeadTrack_;

      tauDecayModeEnum                                  theDecayMode_;
      int                                               aFinalStateTau_;

};


#endif
