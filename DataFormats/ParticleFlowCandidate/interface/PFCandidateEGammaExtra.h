#ifndef ParticleFlowCandidate_PFCandidateEGammaExtra_h
#define ParticleFlowCandidate_PFCandidateEGammaExtra_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"

#include <iosfwd>

namespace reco {
/** \class reco::PFCandidateEGammaExtra
 *
 * extra information on the photon/electron particle candidate from particle flow
 *
 */
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
      kFailsClusterIso,
      kN_EVETOS
    };

    enum PhotonVetoes {
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

    /// return a reference to the corresponding GSF track
    reco::GsfTrackRef gsfTrackRef() const { return gsfTrackRef_; }     

    /// return a reference to the corresponding KF track
    reco::TrackRef kfTrackRef() const { return kfTrackRef_; }     

    /// return a reference to the corresponding supercluster
    reco::SuperClusterRef superClusterRef() const {return scRef_ ; }

    /// return a reference to the corresponding box supercluster
    reco::SuperClusterRef superClusterBoxRef() const {return scBoxRef_ ; }    
    
    /// set reference to the corresponding supercluster
    void setSuperClusterRef(reco::SuperClusterRef sc) { scRef_ = sc; }

    /// set reference to the corresponding supercluster
    void setSuperClusterBoxRef(reco::SuperClusterRef sc) { scBoxRef_ = sc; }   
    
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

    /// set veto bits
    void setElectronVetoes(unsigned bits) { elevetoes_ = bits; }
    void setPhotonVetoes(unsigned bits) { phovetoes_ = bits; }

    /// access to veto bits
    unsigned electronVetoes() const { return elevetoes_; }
    unsigned photonVetoes() const { return phovetoes_; } 

 private:
    void  setVariable(MvaVariable type,float var);
    
 private:
    /// Ref to the GSF track
    reco::GsfTrackRef gsfTrackRef_;
    /// Ref to the KF track
    reco::TrackRef kfTrackRef_;

    /// Ref to (refined) supercluster
    reco::SuperClusterRef scRef_;

    /// Ref to box supercluster
    reco::SuperClusterRef scBoxRef_;    
    
    ///  vector of TrackRef from Single Leg conversions
    std::vector<reco::TrackRef> assoSingleLegRefTrack_;

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

    unsigned elevetoes_,phovetoes_;
  };

  /// print the variables
  std::ostream& operator<<( std::ostream& out, const PFCandidateEGammaExtra& c );

}
#endif
