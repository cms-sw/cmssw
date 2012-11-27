#ifndef ParticleFlowCandidate_PFCandidate_h
#define ParticleFlowCandidate_PFCandidate_h
/** \class reco::PFCandidate
 *
 * particle candidate from particle flow
 *
 */

#include <iosfwd>

#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtraFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtraFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
namespace reco {
  /**\class PFCandidate
     \brief Particle reconstructed by the particle flow algorithm.
          
     \author Colin Bernet
     \date   February 2007
  */
  class PFCandidate : public CompositeCandidate {

  public:
    
    /// particle types
    enum ParticleType {
      X=0,     // undefined
      h,       // charged hadron
      e,       // electron 
      mu,      // muon 
      gamma,   // photon
      h0,      // neutral hadron
      h_HF,        // HF tower identified as a hadron
      egamma_HF    // HF tower identified as an EM particle
    };

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
      T_TO_DISP,
      T_FROM_DISP,
      T_FROM_V0,
      T_FROM_GAMMACONV,
      GAMMA_TO_GAMMACONV
    };
    

    enum PFVertexType {
      kCandVertex=0, 
      kTrkVertex=1,
      kComMuonVertex=2,
      kSAMuonVertex=3,
      kTrkMuonVertex=4,
      kGSFVertex=5,
      kTPFMSMuonVertex=6,
      kPickyMuonVertex=7
    };


    /// default constructor
    PFCandidate();

    /// constructor from a reference (keeps track of source relationship)
    PFCandidate( const PFCandidatePtr& sourcePtr );
    
    /*     PFCandidate( Charge q,  */
    /*                  const LorentzVector & p4,  */
    /*                  ParticleType particleId,  */
    /*                  reco::PFBlockRef blockRef ); */
    PFCandidate( Charge q, 
                 const LorentzVector & p4, 
                 ParticleType particleId );

    /// destructor
    virtual ~PFCandidate();

    /// return a clone
    virtual PFCandidate * clone() const;


    /*    /// set source ref */
    /*     void setSourceRef(const PFCandidateRef& ref) { sourceRef_ = ref; } */

    /*     size_type numberOfSourceCandidateRefs() const {return 1;} */

    /*     CandidateBaseRef sourceCandidateRef( size_type i ) const { */
    /*       return  CandidateBaseRef(sourceRef_); */
    /*     } */

    //using reco::Candidate::setSourceCandidatePtr;
    void setSourceCandidatePtr(const PFCandidatePtr& ptr) { sourcePtr_ = ptr; }

    size_t numberOfSourceCandidatePtrs() const { 
      return 1;
    }
    
    CandidatePtr sourceCandidatePtr( size_type i ) const {
      return sourcePtr_;
    }

    /// returns the pdg id corresponding to the particle type.
    /// the particle type could be removed at some point to gain some space.
    /// low priority
    int  translateTypeToPdgId( ParticleType type ) const;
    ParticleType translatePdgIdToType(int pdgid) const;

    /// set Particle Type
    void setParticleType( ParticleType type ); 

    
    /// add an element to the current PFCandidate
    /*     void addElement( const reco::PFBlockElement* element ); */
    
    /// add element in block
    void addElementInBlock( const reco::PFBlockRef& blockref,
                            unsigned elementIndex );

    /// set track reference
    void setTrackRef(const reco::TrackRef& ref);

    /// return a reference to the corresponding track, if charged. 
    /// otherwise, return a null reference
    reco::TrackRef trackRef() const;

    /// set gsftrack reference 
    void setGsfTrackRef(const reco::GsfTrackRef& ref);   

    /// return a reference to the corresponding GSF track, if an electron. 
    /// otherwise, return a null reference 
    reco::GsfTrackRef gsfTrackRef() const;

    /// set muon reference
    void setMuonRef(const reco::MuonRef& ref);

    /// return a reference to the corresponding muon, if a muon. 
    /// otherwise, return a null reference
    reco::MuonRef muonRef() const;




    /// set displaced vertex reference
    void setDisplacedVertexRef(const reco::PFDisplacedVertexRef& ref, Flags flag);

    /// return a reference to the corresponding displaced vertex,
    /// otherwise, return a null reference
    reco::PFDisplacedVertexRef displacedVertexRef(Flags type) const;

    /// set ref to original reco conversion
    void setConversionRef(const reco::ConversionRef& ref);

    /// return a reference to the original conversion
    reco::ConversionRef conversionRef() const;

    /// set ref to original reco conversion
    void setV0Ref(const reco::VertexCompositeCandidateRef& ref);

    /// return a reference to the original conversion
    reco::VertexCompositeCandidateRef v0Ref() const;

    /// return a reference to the corresponding GsfElectron if any
    reco::GsfElectronRef gsfElectronRef() const;

    /// return a reference to the electron extra
    reco::PFCandidateElectronExtraRef electronExtraRef() const;

    /// set corrected Ecal energy 
    void setEcalEnergy( float eeRaw, float eeCorr ) {
      rawEcalEnergy_ = eeRaw; ecalERatio_= std::abs(eeRaw)<1.e-6 ? 1.0 : eeCorr/eeRaw;}

    /// return corrected Ecal energy
    double ecalEnergy() const { return ecalERatio_*rawEcalEnergy_;}

    /// return corrected Ecal energy
    double rawEcalEnergy() const { return rawEcalEnergy_;}
    
    /// set corrected Hcal energy 
    void setHcalEnergy( float ehRaw, float ehCorr ) {
      rawHcalEnergy_ = ehRaw; hcalERatio_= std::abs(ehRaw)<1.e-6 ? 1.0 : ehCorr/ehRaw;}

    /// return corrected Hcal energy
    double hcalEnergy() const { return hcalERatio_*rawHcalEnergy_;}

    /// return raw Hcal energy
    double rawHcalEnergy() const { return rawHcalEnergy_;}

    /// set corrected Hcal energy 
    void setHoEnergy( float eoRaw, float eoCorr ) {
      rawHoEnergy_ = eoRaw; hoERatio_= std::abs(eoRaw)<1.e-6 ? 1.0 : eoCorr/eoRaw;}

    /// return corrected Hcal energy
    double hoEnergy() const { return hoERatio_*rawHoEnergy_;}

    /// return raw Hcal energy
    double rawHoEnergy() const { return rawHoEnergy_;}

    /// set GsfElectronRef 
    void setGsfElectronRef (const reco::GsfElectronRef & ref);

    void setSuperClusterRef (const reco::SuperClusterRef& scRef);

    /// return a reference to the corresponding SuperCluster if any
    reco::SuperClusterRef superClusterRef() const; 

    /// set ref to the corresponding reco::Photon if any
    void setPhotonRef(const reco::PhotonRef& phRef);
    
    /// return a reference to the corresponding Photon if any
    reco::PhotonRef photonRef() const; 

    /// set the PF Photon Extra Ref
    void setPFPhotonExtraRef(const reco::PFCandidatePhotonExtraRef& ref);

    /// return a reference to the photon extra
    reco::PFCandidatePhotonExtraRef photonExtraRef() const;


    /// set corrected PS1 energy
    void setPs1Energy( float e1 ) {ps1Energy_ = e1;}

    /// return corrected PS1 energy
    double pS1Energy() const { return ps1Energy_;}

    /// set corrected PS2 energy 
    void setPs2Energy( float e2 ) {ps2Energy_ = e2;}

    /// return corrected PS2 energy
    double pS2Energy() const { return ps2Energy_;}

    /// particle momentum *= rescaleFactor
    void rescaleMomentum( double rescaleFactor );

    /// set a given flag
    void setFlag(Flags theFlag, bool value);
    
    /// return a given flag
    bool flag(Flags theFlag) const;

    /// set uncertainty on momentum
    void setDeltaP(double dp ) {deltaP_ = dp;}

    /// uncertainty on 3-momentum
    double deltaP() const { return deltaP_;}

    //  int pdgId() const { return translateTypeToPdgId( particleId_ ); } 

    /// set mva for electron-pion discrimination. 
    /// For charged particles, this variable is set 
    ///   to 0 for particles that are not preided 
    ///   to 1 otherwise
    /// For neutral particles, it is set to the default value


    void set_mva_e_pi( float mva ){ mva_e_pi_=mva;}
    
    /// mva for electron-pion discrimination
    float mva_e_pi() const { return mva_e_pi_;}

    
    /// set mva for electron-muon discrimination
    void set_mva_e_mu( float mva ) { mva_e_mu_=mva;}
    
    /// mva for electron-muon discrimination
    float mva_e_mu() const { return mva_e_mu_;}

    /// set mva for pi-muon discrimination
    void set_mva_pi_mu( float mva ) { mva_pi_mu_=mva;}

    /// mva for pi-muon discrimination
    float mva_pi_mu() const { return mva_pi_mu_;}
    

    /// set mva for gamma detection
    void set_mva_nothing_gamma( float mva ) { mva_nothing_gamma_=mva;}

    /// mva for gamma detection
    float mva_nothing_gamma() const { return mva_nothing_gamma_;}

    /// set mva for neutral hadron detection
    void set_mva_nothing_nh( float mva ) { mva_nothing_nh_=mva;}

    /// mva for neutral hadron detection
    float mva_nothing_nh() const { return mva_nothing_nh_;}
    
    
    /// set mva for neutral hadron - gamma discrimination
    void set_mva_gamma_nh( float mva ) { mva_gamma_nh_=mva;}
	
    /// mva for neutral hadron - gamma discrimination
    float mva_gamma_nh() const { return mva_gamma_nh_;}

    /// set position at ECAL entrance
    void setPositionAtECALEntrance(const math::XYZPointF& pos) {
      positionAtECALEntrance_ = pos;
    } 
    
    /// set the PF Electron Extra Ref
    void setPFElectronExtraRef(const reco::PFCandidateElectronExtraRef& ref);

    /// \return position at ECAL entrance
    const math::XYZPointF& positionAtECALEntrance() const {
      return positionAtECALEntrance_;
    }
    
    /// particle identification code
    /// \todo use Particle::pdgId_ and remove this data member
    virtual  ParticleType particleId() const { return translatePdgIdToType(pdgId_);}

    
    /// return indices of elements used in the block
    /*     const std::vector<unsigned>& elementIndices() const {  */
    /*       return elementIndices_; */
    /*     } */
    /// return elements
    /*     const edm::OwnVector< reco::PFBlockElement >& elements() const  */
    /*       {return elements_;} */

    /// return elements in blocks
    typedef std::pair<reco::PFBlockRef, unsigned> ElementInBlock;
    typedef std::vector< ElementInBlock > ElementsInBlocks;

    typedef edm::RefVector<reco::PFBlockCollection> Blocks;
    typedef std::vector<unsigned> Elements;

    const ElementsInBlocks& elementsInBlocks() const { 
      
      if (elementsInBlocks_.size()!=blocksStorage_.size())
	{
	  elementsInBlocks_.resize(blocksStorage_.size());
	  for(unsigned int icopy=0;icopy!=blocksStorage_.size();++icopy)
	    elementsInBlocks_[icopy]=std::make_pair(blocksStorage_[icopy],elementsStorage_[icopy]);
	}
      return elementsInBlocks_;
    }
    
  

    static const float bigMva_;

    friend std::ostream& operator<<( std::ostream& out, 
                                     const PFCandidate& c );

    //Tips on setting the vertex efficiently
    //There are two choices: a) use the vertex_ data member, or b) point to the vertex
    //of one of the refs stored by this class. The PFVertexType enum gives the current list
    //of possible references. For these references, use the setVeretxSource method and NOT
    //the setVertex method. If none of the available refs have the vertex that you want for this
    //PFCandidate, use the setVertex method. If you find that you are using frequently two store a 
    // vertex that is the same as one of the refs in this class, you should just extend the enum
    // and modify the vertex() method accordingly.
    void setVertexSource( PFVertexType vt) { vertexType_=vt; if (vertexType_!=kCandVertex) vertex_=Point(0.,0.,0.);}

    virtual void setVertex( math::XYZPoint p) {
      vertex_=p; vertexType_ = kCandVertex;
    }

    virtual const Point & vertex() const;
    virtual double vx() const {return vertex().x();}
    virtual double vy() const {return vertex().y();}
    virtual double vz() const {return vertex().z();}

  private:
    /// Polymorphic overlap
    virtual bool overlap( const Candidate & ) const;

    void setFlag(unsigned shift, unsigned flag, bool value);

    bool flag(unsigned shift, unsigned flag) const;
   
   
    mutable ElementsInBlocks elementsInBlocks_;
    Blocks blocksStorage_;
    Elements elementsStorage_;

    /// reference to the source PFCandidate, if any
    /*     PFCandidateRef sourceRef_; */
    PFCandidatePtr sourcePtr_;


    /// corrected ECAL energy ratio (corrected/raw)
    float       ecalERatio_;

    /// corrected HCAL energy ratio (corrected/raw)
    float       hcalERatio_;

    /// corrected HO energy ratio (corrected/raw)
    float       hoERatio_;

    /// raw ECAL energy
    float       rawEcalEnergy_;

    /// raw HCAL energy
    float       rawHcalEnergy_;

    /// raw HO energy 
    float       rawHoEnergy_;

    /// corrected PS1 energy
    float       ps1Energy_;

    /// corrected PS2 energy
    float       ps2Energy_;

    /// all flags, packed (ecal regional, hcal regional, tracking)
    unsigned    flags_;

    /// uncertainty on 3-momentum
    float      deltaP_;

    PFVertexType vertexType_;

    /// mva for electron-pion discrimination
    float       mva_e_pi_;

    /// mva for electron-muon discrimination
    float       mva_e_mu_;

    /// mva for pi-muon discrimination
    float       mva_pi_mu_;
    
    /// mva for gamma detection
    float       mva_nothing_gamma_;

    /// mva for neutral hadron detection
    float       mva_nothing_nh_;

    /// mva for neutral hadron - gamma discrimination
    float       mva_gamma_nh_;

    /// position at ECAL entrance, from the PFRecTrack
    math::XYZPointF   positionAtECALEntrance_;


    //more efficiently stored refs
    void storeRefInfo(unsigned int iMask, unsigned int iBit, bool iIsValid, 
		      const edm::RefCore& iCore, size_t iKey, 
		      const edm::EDProductGetter*);
    bool getRefInfo(unsigned int iMask, unsigned int iBit, 
		    edm::ProductID& oProdID, size_t& oIndex, size_t& aIndex ) const;
    

    const edm::EDProductGetter* getter_; //transient
    unsigned short storedRefsBitPattern_;
    std::vector<unsigned long long> refsInfo_;
    std::vector<const void *> refsCollectionCache_;

  };

  /// particle ID component tag
  struct PFParticleIdTag { };

  /// get default PFBlockRef component
  /// as: pfcand->get<PFBlockRef>();
  /*   GET_DEFAULT_CANDIDATE_COMPONENT( PFCandidate, PFBlockRef, block ); */

  /// get int component
  /// as: pfcand->get<int, PFParticleIdTag>();
  GET_CANDIDATE_COMPONENT( PFCandidate, PFCandidate::ParticleType, particleId, PFParticleIdTag );

  std::ostream& operator<<( std::ostream& out, const PFCandidate& c );

  
}

#endif
