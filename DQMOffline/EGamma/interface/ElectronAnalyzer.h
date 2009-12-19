
#ifndef DQMOffline_EGamma_ElectronAnalyzer_h
#define DQMOffline_EGamma_ElectronAnalyzer_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class MagneticField ;

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

class ElectronAnalyzer : public ElectronDqmAnalyzerBase
 {
  public:

    explicit ElectronAnalyzer(const edm::ParameterSet& conf);
    virtual ~ElectronAnalyzer();

    virtual void book() ;
    virtual void analyze( const edm::Event & e, const edm::EventSetup & c) ;

  private:

    //=========================================
    // parameters
    //=========================================

    // general, I/O
    edm::InputTag electronCollection_;
    edm::InputTag matchingObjectCollection_;
    std::string matchingCondition_; //NEW
    bool readAOD_; //NEW

    // matching
    double maxPtMatchingObject_; // SURE ?
    double maxAbsEtaMatchingObject_; // SURE ?
    double deltaR_;

    // tag and probe NEW
    int Selection_;
    double massLow_;
    double massHigh_;
    bool TPchecksign_;
    bool TAGcheckclass_;
    bool PROBEetcut_;
    bool PROBEcheckclass_;

    // electron selection NEW
    double minEt_;
    double minPt_;
    double maxAbsEta_;
    bool isEB_;
    bool isEE_;
    bool isNotEBEEGap_;
    bool isEcalDriven_;
    bool isTrackerDriven_;
    double eOverPMinBarrel_;
    double eOverPMaxBarrel_;
    double eOverPMinEndcaps_;
    double eOverPMaxEndcaps_;
    double dEtaMinBarrel_;
    double dEtaMaxBarrel_;
    double dEtaMinEndcaps_;
    double dEtaMaxEndcaps_;
    double dPhiMinBarrel_;
    double dPhiMaxBarrel_;
    double dPhiMinEndcaps_;
    double dPhiMaxEndcaps_;
    double sigIetaIetaMinBarrel_;
    double sigIetaIetaMaxBarrel_;
    double sigIetaIetaMinEndcaps_;
    double sigIetaIetaMaxEndcaps_;
    double hadronicOverEmMaxBarrel_;
    double hadronicOverEmMaxEndcaps_;
    double mvaMin_;
    double tipMaxBarrel_;
    double tipMaxEndcaps_;
    double tkIso03Max_;
    double hcalIso03Depth1MaxBarrel_;
    double hcalIso03Depth1MaxEndcaps_;
    double hcalIso03Depth2MaxEndcaps_;
    double ecalIso03MaxBarrel_;
    double ecalIso03MaxEndcaps_;

    // for trigger NEW
    edm::InputTag triggerResults_;
    std::vector<std::string > HLTPathsByName_;

    // histos limits and binning
    int nbineta; int nbineta2D; double etamin; double etamax;
    int nbinphi; int nbinphi2D; double phimin; double phimax;
    int nbinpt; int nbinpteff; int nbinpt2D; double ptmax;
    int nbinp; int nbinp2D; double pmax;
    int nbineop; int nbineop2D; double eopmax; double eopmaxsht;
    int nbindeta; double detamin; double detamax;
    int nbindphi; double dphimin; double dphimax;
    int nbindetamatch; int nbindetamatch2D; double detamatchmin; double detamatchmax;
    int nbindphimatch; int nbindphimatch2D; double dphimatchmin; double dphimatchmax;
    int nbinfhits; double fhitsmax;
    int nbinlhits; double lhitsmax;
    int nbinxyz;
    int nbinpoptrue; double poptruemin; double poptruemax; //NEW
    int nbinmee; double meemin; double meemax; //NEW
    int nbinhoe; double hoemin; double hoemax; //NEW

    //=========================================
    // general attributes and utility methods
    //=========================================

    unsigned int nEvents_ ;

    float computeInvMass
     ( const reco::GsfElectron & e1,
       const reco::GsfElectron & e2 ) ;
    void fillMatchedHistos
     ( const reco::SuperClusterCollection::const_iterator & moIter,
       const reco::GsfElectron & electron ) ;

    bool selected( const reco::GsfElectronCollection::const_iterator & gsfIter , double vertexTIP ) ;
    bool generalCut( const reco::GsfElectronCollection::const_iterator & gsfIter) ;
    bool etCut( const reco::GsfElectronCollection::const_iterator & gsfIter ) ;
    bool isolationCut( const reco::GsfElectronCollection::const_iterator & gsfIter, double vertexTIP ) ;
    bool idCut( const reco::GsfElectronCollection::const_iterator & gsfIter ) ;

    bool trigger( const edm::Event & e ) ;
    unsigned int nAfterTrigger_;
    std::vector<unsigned int> HLTPathsByIndex_;

    TrajectoryStateTransform transformer_ ;
    edm::ESHandle<TrackerGeometry> pDD;
    edm::ESHandle<MagneticField> theMagField;

    float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10];
    float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10];
    float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10];

    //=========================================
    // histograms
    //=========================================

    // electron basic quantities
    MonitorElement * h_ele_vertexPt ;
//    MonitorElement * h_ele_vertexEtaVsPhi ;
    MonitorElement * h_ele_vertexEta ;
    MonitorElement * h_ele_vertexPhi ;
//    MonitorElement * h_ele_vertexP ;
//    MonitorElement * h_ele_charge ;
//    MonitorElement * h_ele_Et ;
//    MonitorElement * h_ele_vertexXvsY ;
    MonitorElement * h_ele_vertexX ;
    MonitorElement * h_ele_vertexY ;
    MonitorElement * h_ele_vertexZ ;
//    MonitorElement * h_ele_vertexTIP ;

    // # rec electrons
    MonitorElement * histNum_ ;

    // SuperClusters
    MonitorElement * histSclEn_ ;
    MonitorElement * histSclEt_ ;
    MonitorElement * histSclEta_ ;
    MonitorElement * histSclPhi_ ;
    MonitorElement * histSclSigEtaEta_ ;

    // electron track
    MonitorElement * h_ele_ambiguousTracks ;
    MonitorElement * h_ele_ambiguousTracksVsEta ;
    MonitorElement * h_ele_ambiguousTracksVsPhi ;
    MonitorElement * h_ele_ambiguousTracksVsPt ;
    MonitorElement * h_ele_foundHits ;
    MonitorElement * h_ele_foundHitsVsEta ;
    MonitorElement * h_ele_foundHitsVsPhi ;
    MonitorElement * h_ele_foundHitsVsPt ;
    MonitorElement * h_ele_lostHits ;
    MonitorElement * h_ele_lostHitsVsEta ;
    MonitorElement * h_ele_lostHitsVsPhi ;
    MonitorElement * h_ele_lostHitsVsPt ;
    MonitorElement * h_ele_chi2 ;
    MonitorElement * h_ele_chi2VsEta ;
    MonitorElement * h_ele_chi2VsPhi ;
    MonitorElement * h_ele_chi2VsPt ;

    // electron matching and ID
    //MonitorElement * h_ele_EoPout ;
    //MonitorElement * h_ele_dEtaCl_propOut ;
    //MonitorElement * h_ele_dPhiCl_propOut ;
    MonitorElement * h_ele_Eop ;
    MonitorElement * h_ele_EopVsEta ;
    MonitorElement * h_ele_EopVsPhi ;
    MonitorElement * h_ele_EopVsPt ;
    MonitorElement * h_ele_EeleOPout ;
    MonitorElement * h_ele_EeleOPoutVsEta ;
    MonitorElement * h_ele_EeleOPoutVsPhi ;
    MonitorElement * h_ele_EeleOPoutVsPt ;
    MonitorElement * h_ele_dEtaSc_propVtx ;
    MonitorElement * h_ele_dEtaSc_propVtxVsEta ;
    MonitorElement * h_ele_dEtaSc_propVtxVsPhi ;
    MonitorElement * h_ele_dEtaSc_propVtxVsPt ;
    MonitorElement * h_ele_dPhiSc_propVtx ;
    MonitorElement * h_ele_dPhiSc_propVtxVsEta ;
    MonitorElement * h_ele_dPhiSc_propVtxVsPhi ;
    MonitorElement * h_ele_dPhiSc_propVtxVsPt ;
    MonitorElement * h_ele_dEtaEleCl_propOut ;
    MonitorElement * h_ele_dEtaEleCl_propOutVsEta ;
    MonitorElement * h_ele_dEtaEleCl_propOutVsPhi ;
    MonitorElement * h_ele_dEtaEleCl_propOutVsPt ;
    MonitorElement * h_ele_dPhiEleCl_propOut ;
    MonitorElement * h_ele_dPhiEleCl_propOutVsEta ;
    MonitorElement * h_ele_dPhiEleCl_propOutVsPhi ;
    MonitorElement * h_ele_dPhiEleCl_propOutVsPt ;
    MonitorElement * h_ele_Hoe ;
    MonitorElement * h_ele_HoeVsEta ;
    MonitorElement * h_ele_HoeVsPhi ;
    MonitorElement * h_ele_HoeVsPt ;
    //MonitorElement * h_ele_outerP ;
    //MonitorElement * h_ele_outerP_mode ;
    MonitorElement * h_ele_innerPt_mean ;
    MonitorElement * h_ele_outerPt_mean ;
    MonitorElement * h_ele_outerPt_mode ;

    MonitorElement * h_ele_PinMnPout ;
    MonitorElement * h_ele_PinMnPout_mode ;

    MonitorElement * h_ele_fbrem ;
    MonitorElement * h_ele_fbremVsEta ;
    MonitorElement * h_ele_fbremVsPhi ;
    MonitorElement * h_ele_fbremVsPt ;
    MonitorElement * h_ele_classes ;

    MonitorElement * h_ele_mva ;
    MonitorElement * h_ele_provenance ;

    MonitorElement * h_ele_tkSumPt_dr03 ;
    MonitorElement * h_ele_ecalRecHitSumEt_dr03 ;
    MonitorElement * h_ele_hcalDepth1TowerSumEt_dr03 ;
    MonitorElement * h_ele_hcalDepth2TowerSumEt_dr03 ;
    MonitorElement * h_ele_tkSumPt_dr04 ;
    MonitorElement * h_ele_ecalRecHitSumEt_dr04 ;
    MonitorElement * h_ele_hcalDepth1TowerSumEt_dr04 ;
    MonitorElement * h_ele_hcalDepth2TowerSumEt_dr04 ;

    MonitorElement * h_ele_mee ;
    MonitorElement * h_ele_mee_os ;

//    // histos for matching and matched objects
//
//    MonitorElement * h_matchedEle_eta ;
//    MonitorElement * h_matchedEle_eta_golden ;
//    MonitorElement * h_matchedEle_eta_shower ;
//    //MonitorElement * h_matchedEle_eta_bbrem ;
//    //MonitorElement * h_matchedEle_eta_narrow ;
//
//    MonitorElement * h_matchedObject_Eta ;
//    MonitorElement * h_matchedObject_AbsEta ;
//    MonitorElement * h_matchedObject_Pt ;
//    MonitorElement * h_matchedObject_Phi ;
//    MonitorElement * h_matchedObject_Z ;
//
//    MonitorElement * h_matchingObject_Num ;
//    MonitorElement * h_matchingObject_Eta ;
//    MonitorElement * h_matchingObject_AbsEta ;
//    MonitorElement * h_matchingObject_P ;
//    MonitorElement * h_matchingObject_Pt ;
//    MonitorElement * h_matchingObject_Phi ;
//    MonitorElement * h_matchingObject_Z ;

 } ;

#endif



