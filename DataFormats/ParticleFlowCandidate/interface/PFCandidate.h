#ifndef ParticleFlowCandidate_PFCandidate_h
#define ParticleFlowCandidate_PFCandidate_h
/** \class reco::PFCandidate
 *
 * particle candidate from particle flow
 *
 */

#include <iostream>

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

namespace reco {
  /**\class PFCandidate
     \brief Particle reconstructed by the particle flow algorithm.
          
     \author Colin Bernet
     \date   February 2007
  */
  class PFCandidate : public LeafCandidate {

  public:
    
    /// particle types
    enum ParticleType {
      X=0,     // undefined
      h,       // charged hadron
      e,       // electron 
      mu,      // muon 
      gamma,   // photon
      h0       // neutral hadron
    };

    /// default constructor
    PFCandidate();
    
    PFCandidate( Charge q, 
		 const LorentzVector & p4, 
		 ParticleType particleId, 
		 reco::PFBlockRef blockRef );

    /// destructor
    virtual ~PFCandidate() {}

    /// return a clone
    virtual PFCandidate * clone() const;
    
    /// set track reference
    void    setTrackRef(const reco::TrackRef& ref);

    /// set muon reference
    void    setMuonRef(const reco::MuonRef& ref);

    void    setEcalEnergy( float ee ) {ecalEnergy_ = ee;}
    void    setHcalEnergy( float eh ) {hcalEnergy_ = eh;}
    void    setPs1Energy( float e1 ) {ps1Energy_ = e1;}
    void    setPs2Energy( float e2 ) {ps2Energy_ = e2;}

    void    rescaleMomentum( double rescaleFactor );

    enum Flags {
      NORMAL=0,
      E_PHI_SMODULES,
      E_ETA_0,
      E_ETA_MODULES,
      E_BARREL_ENDCAP,
      E_PRESHOWER_EDGE,
      E_PRESHOWER,
      E_ENDCAP_EDGE,
      H_ETA_0,
      H_BARREL_ENDCAP,
      H_ENDCAP_VFCAL,
      H_VFCAL_EDGE,  
      T_TO_NUCLINT,
      T_FROM_NUCLINT,
      T_FROM_V0,
      T_FROM_GAMMACONV
    };
    
    /// set a given flag
    void setFlag(Flags theFlag, bool value);
    
    /// return a given flag
    bool flag(Flags theFlag) const;


    /// particle identification
    virtual int particleId() const { return particleId_;}
    
    /// return reference to the block
    const reco::PFBlockRef& blockRef() const { return blockRef_; } 

    /// return a reference to the corresponding track, if charged. 
    /// otherwise, return a null reference
    reco::TrackRef trackRef() const { return trackRef_; }

    /// return a reference to the corresponding muon, if a muon. 
    /// otherwise, return a null reference
    reco::MuonRef muonRef() const { return muonRef_; }    

    
    /// return indices of elements used in the block
    /*     const std::vector<unsigned>& elementIndices() const {  */
    /*       return elementIndices_; */
    /*     } */
    
    /// return reference to the block
    PFBlockRef block() const { return blockRef_; } 
        

    friend std::ostream& operator<<( std::ostream& out, 
				     const PFCandidate& c );
    




  private:
    void setFlag(unsigned shift, unsigned flag, bool value);

    bool flag(unsigned shift, unsigned flag) const;
   
    /// particle identification
    ParticleType            particleId_; 
    
    /// reference to the corresponding PFBlock
    reco::PFBlockRef        blockRef_;

    /// indices of the elements used in the PFBlock
    /*     std::vector<unsigned>   elementIndices_; */
    
    reco::TrackRef trackRef_;
    
    reco::MuonRef  muonRef_;
    
    /// corrected ECAL energy
    float        ecalEnergy_;

    /// corrected HCAL energy
    float        hcalEnergy_;

    /// corrected PS1 energy
    float        ps1Energy_;

    /// corrected PS2 energy
    float        ps2Energy_;

    /**
       \btrief all flags, packed (ecal regional, hcal regional, tracking)

       0x0    normal

       0x1    phi boundary between supermodules
       0x2    eta boundary between modules
       0x3    barrel/endcap transition
       0x4    preshower transition
       0x5    edge of ecal endcap

       0x10   barrel/endcap overlap
       0x20   endcap/vfcal  overlap
       0x30   edge of vfcal
    */
    unsigned     flags_;
  };

  /// particle ID component tag
  struct PFParticleIdTag { };

  /// get default PFBlockRef component
  /// as: pfcand->get<PFBlockRef>();
  GET_DEFAULT_CANDIDATE_COMPONENT( PFCandidate, PFBlockRef, block );

  /// get int component
  /// as: pfcand->get<int, PFParticleIdTag>();
  GET_CANDIDATE_COMPONENT( PFCandidate, int, particleId, PFParticleIdTag );

}

#endif
