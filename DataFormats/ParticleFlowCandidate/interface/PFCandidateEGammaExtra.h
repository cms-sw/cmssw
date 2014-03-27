#ifndef ParticleFlowCandidate_PFCandidateEGammaExtra_h
#define ParticleFlowCandidate_PFCandidateEGammaExtra_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"

#include <iosfwd>

namespace reco {
/** \class reco::PFCandidateEGammaExtra
 *
 * extra information on the photon/electron particle candidate from particle flow
 *
 */
  typedef std::pair<reco::PFBlockRef, unsigned> ElementInBlock;
  typedef std::vector< ElementInBlock > ElementsInBlocks;

  class PFCandidateEGammaExtra { 
  public:    
    enum StatusFlag {
      X=0,                            // undefined
      Selected,                       // selected 
      ECALDrivenPreselected,          // ECAL-driven electron pre-selected
      MVASelected,                    // Passed the internal particle-flow selection (mva selection)
      Rejected                        // Rejected 
    };

    // if you had a variable update NMvaVariables 
    enum MvaVariable {
      MVA_FIRST=0,
      MVA_LnPtGsf=MVA_FIRST,
      MVA_EtaGsf,
      MVA_SigmaPtOverPt,
      MVA_Fbrem,
      MVA_Chi2Gsf,
      MVA_NhitsKf,
      MVA_Chi2Kf,
      MVA_EtotOverPin,
      MVA_EseedOverPout,
      MVA_EbremOverDeltaP,
      MVA_DeltaEtaTrackCluster,
      MVA_LogSigmaEtaEta,
      MVA_HOverHE,
      MVA_LateBrem,
      MVA_FirstBrem,
      MVA_MVA,
      MVA_LAST
    };

    enum ElectronVetoes {
      kFailsMVA,
      kKFTracksOnGSFCluster, // any number of additional tracks on GSF cluster
      kFailsTrackAndHCALIso, // > 3 kfs on Gsf-cluster, bad h/e
      kKillAdditionalKFs,    // tracks with hcal linkbut good gsf etot/p_in 
      kItIsAPion,            // bad H/P_in, H/H+E, and E_tot/P_in
      kCrazyEoverP,          // screwey track linking / weird GSFs
      kTooLargeAngle,        // angle between GSF and RSC centroid too large
      kN_EVETOS
    };

    enum PhotonVetoes {
      kFailsTrackIso, // the photon fails tracker isolation
      kN_PHOVETOS
    };

  public:
    /// constructor
    PFCandidateEGammaExtra();
    /// constructor
    PFCandidateEGammaExtra(const GsfTrackRef&);
    /// destructor
    ~PFCandidateEGammaExtra(){;}

    /// set gsftrack reference 
    void setGsfTrackRef(const reco::GsfTrackRef& ref);   

    /// set kf track reference
    void setKfTrackRef(const reco::TrackRef & ref);

    /// set gsf electron cluster ref
    void setGsfElectronClusterRef(const reco::PFBlockRef& blk,
				  const reco::PFBlockElementCluster& ref) {
      eleGsfCluster_ = ElementInBlock(blk,ref.index());
    }

    /// return a reference to the corresponding GSF track
    reco::GsfTrackRef gsfTrackRef() const { return gsfTrackRef_; }     

    /// return a reference to the corresponding KF track
    reco::TrackRef kfTrackRef() const { return kfTrackRef_; }

    /// return a reference to the electron cluster ref
    const ElementInBlock& gsfElectronClusterRef() const { 
      return eleGsfCluster_; 
    }

    /// return a reference to the corresponding supercluster
    reco::SuperClusterRef superClusterRef() const {return scRef_ ; }

    /// return a reference to the corresponding box supercluster
    reco::SuperClusterRef superClusterPFECALRef() const {return scPFECALRef_ ; }    
    
    /// set reference to the corresponding supercluster
    void setSuperClusterRef(reco::SuperClusterRef sc) { scRef_ = sc; }

    /// set reference to the corresponding supercluster
    void setSuperClusterPFECALRef(reco::SuperClusterRef sc) { scPFECALRef_ = sc; }   
    
    /// add Single Leg Conversion TrackRef 
    void addSingleLegConvTrackRef(const reco::TrackRef& trackref);

    /// return vector of Single Leg Conversion TrackRef from 
    const std::vector<reco::TrackRef>& singleLegConvTrackRef() const {return assoSingleLegRefTrack_;}

    /// add Single Leg Conversion mva
    void addSingleLegConvMva(const float& mvasingleleg);

    /// return Single Leg Conversion mva
    const std::vector<float>& singleLegConvMva() const {return assoSingleLegMva_;}

    /// add Conversions from PF
    void addConversionRef(const reco::ConversionRef& convref);

    /// return Conversions from PF
    reco::ConversionRefVector conversionRef() const {return assoConversionsRef_;}     
    
    /// set LateBrem
    void setLateBrem(float val); 
    /// set EarlyBrem
    void setEarlyBrem(float val);

    /// set the pout (not trivial to get from the GSF track)
    void setGsfTrackPout(const math::XYZTLorentzVector& pout);
    
    /// set the cluster energies. the Pout should be saved first 
    void setClusterEnergies(const std::vector<float>& energies);

    /// set the sigmaetaeta
    void setSigmaEtaEta(float val);

    /// set the delta eta
    void setDeltaEta(float val);

    /// set the had energy. The cluster energies should be entered before
    void setHadEnergy(float val);

    /// set the result (mostly for debugging)
    void setMVA(float val);

    /// set status 
    void setStatus(StatusFlag type,bool status=true);

    /// access to the status
    bool electronStatus(StatusFlag) const ;

    /// access to the status
    int electronStatus() const {return status_;}

    /// access to mva variable status
    bool mvaStatus(MvaVariable flag) const;

    /// access to the mva variables
    const std::vector<float> & mvaVariables() const {return mvaVariables_;}

    /// access to any variable
    float mvaVariable(MvaVariable var) const;

    /// access to specific variables
    float hadEnergy() const {return hadEnergy_;}
    float sigmaEtaEta() const {return sigmaEtaEta_;}

    /// track counting for electrons and photons
    void addExtraNonConvTrack(const reco::PFBlockRef& blk,
			      const reco::PFBlockElementTrack& tkref) {
      if( !tkref.trackType(reco::PFBlockElement::T_FROM_GAMMACONV) ) {
	assoNonConvExtraTracks_.push_back(std::make_pair(blk,tkref.index()));
      }
    }
    const ElementsInBlocks& extraNonConvTracks() const {
      return assoNonConvExtraTracks_;
    }        

 private:
    void  setVariable(MvaVariable type,float var);
    
 private:
    /// Ref to the GSF track
    reco::GsfTrackRef gsfTrackRef_;
    /// Ref to the KF track
    reco::TrackRef kfTrackRef_;
    /// Ref to the electron gsf cluster;
    ElementInBlock eleGsfCluster_;

    /// Ref to (refined) supercluster
    reco::SuperClusterRef scRef_;

    /// Ref to PF-ECAL only supercluster
    reco::SuperClusterRef scPFECALRef_;    
    
    ///  vector of TrackRef from Single Leg conversions
    std::vector<reco::TrackRef> assoSingleLegRefTrack_;
    
    // information for track matching
    ElementsInBlocks assoNonConvExtraTracks_;    

    ///  vector of Mvas from Single Leg conversions
    std::vector<float> assoSingleLegMva_;

    /// vector of ConversionRef from PF
    reco::ConversionRefVector assoConversionsRef_;    
    
    /// energy of individual clusters (corrected). 
    /// The first cluster is the seed
    std::vector<float> clusterEnergies_;

    /// mva variables  -  transient !
    std::vector<float> mvaVariables_;
    
    /// status of  mva variables
    int mvaStatus_;

    /// Status of the electron
    int status_;

    /// Variables entering the MVA that should be saved
    math::XYZTLorentzVector pout_;
    float earlyBrem_;
    float lateBrem_;
    float sigmaEtaEta_;
    float hadEnergy_;
    float deltaEta_;
  };

  /// print the variables
  std::ostream& operator<<( std::ostream& out, const PFCandidateEGammaExtra& c );

}
#endif
