#include <iostream>
#include <iomanip>
//

#include "DQMOffline/EGamma/plugins/PhotonAnalyzer.h"


/** \class PhotonAnalyzer
 **
 **
 **  $Id: PhotonAnalyzer
 **  authors:
 **   Nancy Marinelli, U. of Notre Dame, US
 **   Jamie Antonelli, U. of Notre Dame, US
 **
 ***/

using namespace std;


PhotonAnalyzer::PhotonAnalyzer( const edm::ParameterSet& pset )
{

    fName_                  = pset.getParameter<string>("analyzerName");
    verbosity_              = pset.getUntrackedParameter<int>("Verbosity");

    prescaleFactor_         = pset.getUntrackedParameter<int>("prescaleFactor",1);

    photon_token_           = consumes<vector<reco::Photon> >(pset.getParameter<edm::InputTag>("phoProducer"));

    barrelRecHit_token_     = consumes<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > >(pset.getParameter<edm::InputTag>("barrelRecHitProducer"));

    PhotonIDLoose_token_    = consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("photonIDLoose"));
    PhotonIDTight_token_    = consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("photonIDTight"));

    endcapRecHit_token_     = consumes<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > >(pset.getParameter<edm::InputTag>("endcapRecHitProducer"));

    triggerEvent_token_     = consumes<trigger::TriggerEvent>(pset.getParameter<edm::InputTag>("triggerEvent"));

    offline_pvToken_ = consumes<reco::VertexCollection>(pset.getUntrackedParameter<edm::InputTag>("offlinePV", edm::InputTag("offlinePrimaryVertices")));


    minPhoEtCut_            = pset.getParameter<double>("minPhoEtCut");
    photonMaxEta_           = pset.getParameter<double>("maxPhoEta");
    invMassEtCut_           = pset.getParameter<double>("invMassEtCut");
    cutStep_                = pset.getParameter<double>("cutStep");
    numberOfSteps_          = pset.getParameter<int>("numberOfSteps");

    useBinning_             = pset.getParameter<bool>("useBinning");
    useTriggerFiltering_    = pset.getParameter<bool>("useTriggerFiltering");

    minimalSetOfHistos_     = pset.getParameter<bool>("minimalSetOfHistos");
    excludeBkgHistos_       = pset.getParameter<bool>("excludeBkgHistos");

    standAlone_             = pset.getParameter<bool>("standAlone");
    outputFileName_         = pset.getParameter<string>("OutputFileName");

    isolationStrength_      = pset.getParameter<int>("isolationStrength");

    isHeavyIon_             = pset.getUntrackedParameter<bool>("isHeavyIon",false);

    parameters_ = pset;

    histo_index_photons_ = 0;
    histo_index_conversions_ = 0;
    histo_index_efficiency_ = 0;
    histo_index_invMass_ = 0;
}



PhotonAnalyzer::~PhotonAnalyzer() {}


void PhotonAnalyzer::beginJob()
{

  nEvt_=0;
  nEntry_=0;

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();



  double eMin = parameters_.getParameter<double>("eMin");
  double eMax = parameters_.getParameter<double>("eMax");
  int    eBin = parameters_.getParameter<int>("eBin");

  double etMin = parameters_.getParameter<double>("etMin");
  double etMax = parameters_.getParameter<double>("etMax");
  int    etBin = parameters_.getParameter<int>("etBin");

  double sumMin = parameters_.getParameter<double>("sumMin");
  double sumMax = parameters_.getParameter<double>("sumMax");
  int    sumBin = parameters_.getParameter<int>("sumBin");

  double etaMin = parameters_.getParameter<double>("etaMin");
  double etaMax = parameters_.getParameter<double>("etaMax");
  int    etaBin = parameters_.getParameter<int>("etaBin");

  double phiMin = parameters_.getParameter<double>("phiMin");
  double phiMax = parameters_.getParameter<double>("phiMax");
  int    phiBin = parameters_.getParameter<int>("phiBin");

  double r9Min = parameters_.getParameter<double>("r9Min");
  double r9Max = parameters_.getParameter<double>("r9Max");
  int    r9Bin = parameters_.getParameter<int>("r9Bin");

  double hOverEMin = parameters_.getParameter<double>("hOverEMin");
  double hOverEMax = parameters_.getParameter<double>("hOverEMax");
  int    hOverEBin = parameters_.getParameter<int>("hOverEBin");

  double xMin = parameters_.getParameter<double>("xMin");
  double xMax = parameters_.getParameter<double>("xMax");
  int    xBin = parameters_.getParameter<int>("xBin");

  double yMin = parameters_.getParameter<double>("yMin");
  double yMax = parameters_.getParameter<double>("yMax");
  int    yBin = parameters_.getParameter<int>("yBin");

  double numberMin = parameters_.getParameter<double>("numberMin");
  double numberMax = parameters_.getParameter<double>("numberMax");
  int    numberBin = parameters_.getParameter<int>("numberBin");

  double zMin = parameters_.getParameter<double>("zMin");
  double zMax = parameters_.getParameter<double>("zMax");
  int    zBin = parameters_.getParameter<int>("zBin");

  double rMin = parameters_.getParameter<double>("rMin");
  double rMax = parameters_.getParameter<double>("rMax");
  int    rBin = parameters_.getParameter<int>("rBin");

  double dPhiTracksMin = parameters_.getParameter<double>("dPhiTracksMin");
  double dPhiTracksMax = parameters_.getParameter<double>("dPhiTracksMax");
  int    dPhiTracksBin = parameters_.getParameter<int>("dPhiTracksBin");

  double dEtaTracksMin = parameters_.getParameter<double>("dEtaTracksMin");
  double dEtaTracksMax = parameters_.getParameter<double>("dEtaTracksMax");
  int    dEtaTracksBin = parameters_.getParameter<int>("dEtaTracksBin");

  double sigmaIetaMin = parameters_.getParameter<double>("sigmaIetaMin");
  double sigmaIetaMax = parameters_.getParameter<double>("sigmaIetaMax");
  int    sigmaIetaBin = parameters_.getParameter<int>("sigmaIetaBin");

  double eOverPMin = parameters_.getParameter<double>("eOverPMin");
  double eOverPMax = parameters_.getParameter<double>("eOverPMax");
  int    eOverPBin = parameters_.getParameter<int>("eOverPBin");

  double chi2Min = parameters_.getParameter<double>("chi2Min");
  double chi2Max = parameters_.getParameter<double>("chi2Max");
  int    chi2Bin = parameters_.getParameter<int>("chi2Bin");


  int reducedEtBin = etBin/4;
  int reducedEtaBin = etaBin/4;
  int reducedSumBin = sumBin/4;
  int reducedR9Bin = r9Bin/4;


  parts_.push_back("AllEcal");
  parts_.push_back("Barrel");
  parts_.push_back("Endcaps");

  types_.push_back("All");
  types_.push_back("GoodCandidate");
  if (!excludeBkgHistos_) types_.push_back("Background");



  ////////////////START OF BOOKING FOR ALL HISTOGRAMS////////////////

  if (dbe_) {

    dbe_->setCurrentFolder("Egamma/"+fName_+"/");

    //int values stored in MEs to keep track of how many histograms are in each folder
    totalNumberOfHistos_efficiencyFolder =  dbe_->bookInt("numberOfHistogramsInEfficiencyFolder");
    totalNumberOfHistos_photonsFolder =     dbe_->bookInt("numberOfHistogramsInPhotonsFolder");
    totalNumberOfHistos_conversionsFolder = dbe_->bookInt("numberOfHistogramsInConversionsFolder");
    totalNumberOfHistos_invMassFolder =     dbe_->bookInt("numberOfHistogramsInInvMassFolder");


    //Efficiency histograms

    dbe_->setCurrentFolder("Egamma/"+fName_+"/Efficiencies");

    //don't number these histograms with the "bookHisto" method, since they'll be erased in the offline client
    h_phoEta_Loose_ = dbe_->book1D("phoEtaLoose","Loose Photon #eta",etaBin,etaMin,etaMax);
    h_phoEta_Tight_ = dbe_->book1D("phoEtaTight","Tight Photon #eta",etaBin,etaMin,etaMax);
    h_phoEt_Loose_  = dbe_->book1D("phoEtLoose", "Loose Photon E_{T}",etBin,etMin,etMax);
    h_phoEt_Tight_  = dbe_->book1D("phoEtTight", "Tight Photon E_{T}",etBin,etMin,etMax);


    h_phoEta_preHLT_  = dbe_->book1D("phoEtaPreHLT", "Photon #eta: before HLT",etaBin,etaMin,etaMax);
    h_phoEta_postHLT_ = dbe_->book1D("phoEtaPostHLT","Photon #eta: after HLT",etaBin,etaMin,etaMax);
    h_phoEt_preHLT_   = dbe_->book1D("phoEtPreHLT",  "Photon E_{T}: before HLT",etBin,etMin,etMax);
    h_phoEt_postHLT_  = dbe_->book1D("phoEtPostHLT", "Photon E_{T}: after HLT",etBin,etMin,etMax);

    h_convEta_Loose_ = dbe_->book1D("convEtaLoose","Converted Loose Photon #eta",etaBin,etaMin,etaMax);
    h_convEta_Tight_ = dbe_->book1D("convEtaTight","Converted Tight Photon #eta",etaBin,etaMin,etaMax);
    h_convEt_Loose_  = dbe_->book1D("convEtLoose", "Converted Loose Photon E_{T}",etBin,etMin,etMax);
    h_convEt_Tight_  = dbe_->book1D("convEtTight", "Converted Tight Photon E_{T}",etBin,etMin,etMax);

    h_phoEta_Vertex_ = dbe_->book1D("phoEtaVertex","Converted Photons before valid vertex cut: #eta",etaBin,etaMin,etaMax);


    vector<MonitorElement*> temp1DVectorEta;
    vector<MonitorElement*> temp1DVectorPhi;
    vector<vector<MonitorElement*> > temp2DVectorPhi;


    for(int cut = 0; cut != numberOfSteps_; ++cut){ //looping over Et cut values
      for(uint type=0;type!=types_.size();++type){  //looping over isolation type
	currentFolder_.str("");
	currentFolder_ << "Egamma/"+fName_+"/" << types_[type] << "Photons/Et above " << (cut+1)*cutStep_ << " GeV/Conversions";
	dbe_->setCurrentFolder(currentFolder_.str());

	temp1DVectorEta.push_back(dbe_->book1D("phoConvEtaForEfficiency","Converted Photon #eta;#eta",etaBin,etaMin,etaMax));
	for(uint part=0;part!=parts_.size();++part){
	  temp1DVectorPhi.push_back(dbe_->book1D("phoConvPhiForEfficiency"+parts_[part],"Converted Photon #phi;#phi",phiBin,phiMin,phiMax));
	}
	temp2DVectorPhi.push_back(temp1DVectorPhi);
	temp1DVectorPhi.clear();
      }
      h_phoConvEtaForEfficiency_.push_back(temp1DVectorEta);
      temp1DVectorEta.clear();
      h_phoConvPhiForEfficiency_.push_back(temp2DVectorPhi);
      temp2DVectorPhi.clear();
    }




    //Invariant mass plots

    dbe_->setCurrentFolder("Egamma/"+fName_+"/InvMass");

    h_invMassAllPhotons_    = bookHisto("invMassAllIsolatedPhotons","Two photon invariant mass: All isolated photons;M (GeV)",etBin,etMin,etMax);
    h_invMassPhotonsEBarrel_    = bookHisto("invMassIsoPhotonsEBarrel", "Two photon invariant mass: isolated photons in barrel; M (GeV)",etBin,etMin,etMax);
    h_invMassPhotonsEEndcap_    = bookHisto("invMassIsoPhotonsEEndcap", "Two photon invariant mass: isolated photons in endcap; M (GeV)",etBin,etMin,etMax);
    
    h_invMassZeroWithTracks_= bookHisto("invMassZeroWithTracks",    "Two photon invariant mass: Neither has tracks;M (GeV)",  etBin,etMin,etMax);
    h_invMassOneWithTracks_ = bookHisto("invMassOneWithTracks",     "Two photon invariant mass: Only one has tracks;M (GeV)", etBin,etMin,etMax);
    h_invMassTwoWithTracks_ = bookHisto("invMassTwoWithTracks",     "Two photon invariant mass: Both have tracks;M (GeV)",    etBin,etMin,etMax);


    h_nRecoVtx_ =  bookHisto("nOfflineVtx","# of Offline Vertices",80, -0.5, 79.5);    

    ////////////////START OF BOOKING FOR PHOTON-RELATED HISTOGRAMS////////////////

    //ENERGY VARIABLES

    book3DHistoVector(h_phoE_, "1D","phoE","Energy;E (GeV)",eBin,eMin,eMax);
    book3DHistoVector(h_phoSigmaEoverE_, "1D","phoSigmaEoverE","#sigma_{E}/E; #sigma_{E}/E", 100,0.,0.08);
    book3DHistoVector(p_phoSigmaEoverEvsNVtx_, "Profile","phoSigmaEoverEvsNVtx","#sigma_{E}/E vs NVtx; N_{vtx}; #sigma_{E}/E",80, -0.5, 79.5, 100,0., 0.08);
    book3DHistoVector(h_phoEt_, "1D","phoEt","E_{T};E_{T} (GeV)", etBin,etMin,etMax);


    //NUMBER OF PHOTONS

    book3DHistoVector(h_nPho_, "1D","nPho","Number of Photons per Event;# #gamma",numberBin,numberMin,numberMax);

    //GEOMETRICAL VARIABLES

    //photon eta/phi
    book2DHistoVector(h_phoEta_, "1D","phoEta","#eta;#eta",etaBin,etaMin,etaMax) ;
    book3DHistoVector(h_phoPhi_, "1D","phoPhi","#phi;#phi",phiBin,phiMin,phiMax) ;

    //supercluster eta/phi
    book2DHistoVector(h_scEta_, "1D","scEta","SuperCluster #eta;#eta",etaBin,etaMin,etaMax) ;
    book3DHistoVector(h_scPhi_, "1D","scPhi","SuperCluster #phi;#phi",phiBin,phiMin,phiMax) ;

    //SHOWER SHAPE VARIABLES

    //r9
    book3DHistoVector(h_r9_, "1D","r9","R9;R9",r9Bin,r9Min, r9Max);
    if (standAlone_)     book2DHistoVector(h_r9VsEt_, "2D","r9VsEt2D","R9 vs E_{T};E_{T} (GeV);R9",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
    book2DHistoVector(p_r9VsEt_, "Profile","r9VsEt","Avg R9 vs E_{T};E_{T} (GeV);R9",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
    if (standAlone_)  book2DHistoVector(h_r9VsEta_, "2D","r9VsEta2D","R9 vs #eta;#eta;R9",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
    book2DHistoVector(p_r9VsEta_, "Profile","r9VsEta","Avg R9 vs #eta;#eta;R9",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);

    //sigma ieta ieta
    book3DHistoVector(h_phoSigmaIetaIeta_,   "1D","phoSigmaIetaIeta","#sigma_{i#etai#eta};#sigma_{i#etai#eta}",sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
     if (standAlone_)  book2DHistoVector(h_sigmaIetaIetaVsEta_, "2D","sigmaIetaIetaVsEta2D","#sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",reducedEtaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);
    book2DHistoVector(p_sigmaIetaIetaVsEta_, "Profile","sigmaIetaIetaVsEta","Avg #sigma_{i#etai#eta} vs #eta;#eta;#sigma_{i#etai#eta}",etaBin,etaMin,etaMax,sigmaIetaBin,sigmaIetaMin,sigmaIetaMax);

    //e1x5
     if (standAlone_)  book2DHistoVector(h_e1x5VsEt_,  "2D","e1x5VsEt2D","E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",reducedEtBin,etMin,etMax,reducedEtBin,etMin,etMax);
    book2DHistoVector(p_e1x5VsEt_,  "Profile","e1x5VsEt","Avg E1x5 vs E_{T};E_{T} (GeV);E1X5 (GeV)",etBin,etMin,etMax,etBin,etMin,etMax);
     if (standAlone_)  book2DHistoVector(h_e1x5VsEta_, "2D","e1x5VsEta2D","E1x5 vs #eta;#eta;E1X5 (GeV)",reducedEtaBin,etaMin,etaMax,reducedEtBin,etMin,etMax);
    book2DHistoVector(p_e1x5VsEta_, "Profile","e1x5VsEta","Avg E1x5 vs #eta;#eta;E1X5 (GeV)",etaBin,etaMin,etaMax,etBin,etMin,etMax);

    //e2x5
     if (standAlone_)  book2DHistoVector(h_e2x5VsEt_,  "2D","e2x5VsEt2D","E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",reducedEtBin,etMin,etMax,reducedEtBin,etMin,etMax);
    book2DHistoVector(p_e2x5VsEt_,  "Profile","e2x5VsEt","Avg E2x5 vs E_{T};E_{T} (GeV);E2X5 (GeV)",etBin,etMin,etMax,etBin,etMin,etMax);
    if (standAlone_)  book2DHistoVector(h_e2x5VsEta_, "2D","e2x5VsEta2D","E2x5 vs #eta;#eta;E2X5 (GeV)",reducedEtaBin,etaMin,etaMax,reducedEtBin,etMin,etMax);
    book2DHistoVector(p_e2x5VsEta_, "Profile","e2x5VsEta","Avg E2x5 vs #eta;#eta;E2X5 (GeV)",etaBin,etaMin,etaMax,etBin,etMin,etMax);

    //r1x5
    if (standAlone_)     book2DHistoVector(h_r1x5VsEt_,  "2D","r1x5VsEt2D","R1x5 vs E_{T};E_{T} (GeV);R1X5",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
    book2DHistoVector(p_r1x5VsEt_,  "Profile","r1x5VsEt","Avg R1x5 vs E_{T};E_{T} (GeV);R1X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
     if (standAlone_)  book2DHistoVector(h_r1x5VsEta_, "2D","r1x5VsEta2D","R1x5 vs #eta;#eta;R1X5",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
    book2DHistoVector(p_r1x5VsEta_, "Profile","r1x5VsEta","Avg R1x5 vs #eta;#eta;R1X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);

    //r2x5
    if (standAlone_)  book2DHistoVector(    h_r2x5VsEt_  ,"2D","r2x5VsEt2D","R2x5 vs E_{T};E_{T} (GeV);R2X5",reducedEtBin,etMin,etMax,reducedR9Bin,r9Min,r9Max);
    book2DHistoVector(    p_r2x5VsEt_  ,"Profile","r2x5VsEt","Avg R2x5 vs E_{T};E_{T} (GeV);R2X5",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
     if (standAlone_)  book2DHistoVector(    h_r2x5VsEta_ ,"2D","r2x5VsEta2D","R2x5 vs #eta;#eta;R2X5",reducedEtaBin,etaMin,etaMax,reducedR9Bin,r9Min,r9Max);
    book2DHistoVector(    p_r2x5VsEta_ ,"Profile","r2x5VsEta","Avg R2x5 vs #eta;#eta;R2X5",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);

    //maxEXtalOver3x3
    if (standAlone_)  book2DHistoVector(    h_maxEXtalOver3x3VsEt_  ,"2D","maxEXtalOver3x3VsEt2D","(Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",reducedEtBin,etMin,etMax,r9Bin,r9Min,r9Max);
    book2DHistoVector(    p_maxEXtalOver3x3VsEt_  ,"Profile","maxEXtalOver3x3VsEt","Avg (Max Xtal E)/E3x3 vs E_{T};E_{T} (GeV);(Max Xtal E)/E3x3",etBin,etMin,etMax,r9Bin,r9Min,r9Max);
    if (standAlone_)  book2DHistoVector(    h_maxEXtalOver3x3VsEta_ ,"2D","maxEXtalOver3x3VsEta2D","(Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",reducedEtaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);
    book2DHistoVector(    p_maxEXtalOver3x3VsEta_ ,"Profile","maxEXtalOver3x3VsEta","Avg (Max Xtal E)/E3x3 vs #eta;#eta;(Max Xtal E)/E3x3",etaBin,etaMin,etaMax,r9Bin,r9Min,r9Max);

    //TRACK ISOLATION VARIABLES
    //nTrackIsolSolid
    book2DHistoVector(    h_nTrackIsolSolid_       ,"1D","nIsoTracksSolid","Number Of Tracks in the Solid Iso Cone;# tracks",numberBin,numberMin,numberMax);
    if (standAlone_)  book2DHistoVector(    h_nTrackIsolSolidVsEt_   ,"2D","nIsoTracksSolidVsEt2D","Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",reducedEtBin,etMin, etMax,numberBin,numberMin,numberMax);
    book2DHistoVector(    p_nTrackIsolSolidVsEt_   ,"Profile","nIsoTracksSolidVsEt","Avg Number Of Tracks in the Solid Iso Cone vs E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax);
    if (standAlone_)  book2DHistoVector(    h_nTrackIsolSolidVsEta_  ,"2D","nIsoTracksSolidVsEta2D","Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",reducedEtaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
    book2DHistoVector(    p_nTrackIsolSolidVsEta_  ,"Profile","nIsoTracksSolidVsEta","Avg Number Of Tracks in the Solid Iso Cone vs #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax);

    //nTrackIsolHollow
    book2DHistoVector(    h_nTrackIsolHollow_      ,"1D","nIsoTracksHollow","Number Of Tracks in the Hollow Iso Cone;# tracks",numberBin,numberMin,numberMax);
    if (standAlone_)  book2DHistoVector(    h_nTrackIsolHollowVsEt_  ,"2D","nIsoTracksHollowVsEt2D","Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",reducedEtBin,etMin, etMax,numberBin,numberMin,numberMax);
    book2DHistoVector(    p_nTrackIsolHollowVsEt_  ,"Profile","nIsoTracksHollowVsEt","Avg Number Of Tracks in the Hollow Iso Cone vs E_{T};E_{T};# tracks",etBin,etMin,etMax,numberBin,numberMin,numberMax);
    if (standAlone_)  book2DHistoVector(    h_nTrackIsolHollowVsEta_ ,"2D","nIsoTracksHollowVsEta2D","Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",reducedEtaBin,etaMin, etaMax,numberBin,numberMin,numberMax);
    book2DHistoVector(    p_nTrackIsolHollowVsEta_ ,"Profile","nIsoTracksHollowVsEta","Avg Number Of Tracks in the Hollow Iso Cone vs #eta;#eta;# tracks",etaBin,etaMin, etaMax,numberBin,numberMin,numberMax);

    //trackPtSumSolid
    book2DHistoVector(    h_trackPtSumSolid_       ,"1D","isoPtSumSolid","Track P_{T} Sum in the Solid Iso Cone;P_{T} (GeV)",sumBin,sumMin,sumMax);
    if (standAlone_)  book2DHistoVector(    h_trackPtSumSolidVsEt_   ,"2D","isoPtSumSolidVsEt2D","Track P_{T} Sum in the Solid Iso Cone;E_{T} (GeV);P_{T} (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
    book2DHistoVector(    p_trackPtSumSolidVsEt_   ,"Profile","isoPtSumSolidVsEt","Avg Track P_{T} Sum in the Solid Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax);
    if (standAlone_)  book2DHistoVector(    h_trackPtSumSolidVsEta_  ,"2D","isoPtSumSolidVsEta2D","Track P_{T} Sum in the Solid Iso Cone;#eta;P_{T} (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
    book2DHistoVector(    p_trackPtSumSolidVsEta_  ,"Profile","isoPtSumSolidVsEta","Avg Track P_{T} Sum in the Solid Iso Cone vs #eta;#eta;P_{T} (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);

    //trackPtSumHollow
    book2DHistoVector(    h_trackPtSumHollow_      ,"1D","isoPtSumHollow","Track P_{T} Sum in the Hollow Iso Cone;P_{T} (GeV)",sumBin,sumMin,sumMax);
    if (standAlone_)  book2DHistoVector(    h_trackPtSumHollowVsEt_  ,"2D","isoPtSumHollowVsEt2D","Track P_{T} Sum in the Hollow Iso Cone;E_{T} (GeV);P_{T} (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
    book2DHistoVector(    p_trackPtSumHollowVsEt_  ,"Profile","isoPtSumHollowVsEt","Avg Track P_{T} Sum in the Hollow Iso Cone vs E_{T};E_{T} (GeV);P_{T} (GeV)",etBin,etMin,etMax,sumBin,sumMin,sumMax);
    if (standAlone_)  book2DHistoVector(    h_trackPtSumHollowVsEta_ ,"2D","isoPtSumHollowVsEta2D","Track P_{T} Sum in the Hollow Iso Cone;#eta;P_{T} (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
    book2DHistoVector(    p_trackPtSumHollowVsEta_ ,"Profile","isoPtSumHollowVsEta","Avg Track P_{T} Sum in the Hollow Iso Cone vs #eta;#eta;P_{T} (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);


    //CALORIMETER ISOLATION VARIABLES

    //ecal sum
    book2DHistoVector(    h_ecalSum_      ,"1D","ecalSum","Ecal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax);
    book2DHistoVector(    h_ecalSumEBarrel_,"1D","ecalSumEBarrel","Ecal Sum in the IsoCone for Barrel;E (GeV)",sumBin,sumMin,sumMax);
    book2DHistoVector(    h_ecalSumEEndcap_,"1D","ecalSumEEndcap","Ecal Sum in the IsoCone for Endcap;E (GeV)",sumBin,sumMin,sumMax);
    if (standAlone_)  book2DHistoVector(    h_ecalSumVsEt_  ,"2D","ecalSumVsEt2D","Ecal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
    book3DHistoVector(    p_ecalSumVsEt_  ,"Profile","ecalSumVsEt","Avg Ecal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",etBin,etMin, etMax,sumBin,sumMin,sumMax);
    if (standAlone_)  book2DHistoVector(    h_ecalSumVsEta_ ,"2D","ecalSumVsEta2D","Ecal Sum in the Iso Cone;#eta;E (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
    book2DHistoVector(    p_ecalSumVsEta_ ,"Profile","ecalSumVsEta","Avg Ecal Sum in the Iso Cone vs #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);

    //hcal sum
    book2DHistoVector(    h_hcalSum_      ,"1D","hcalSum","Hcal Sum in the Iso Cone;E (GeV)",sumBin,sumMin,sumMax);
    book2DHistoVector(    h_hcalSumEBarrel_,"1D","hcalSumEBarrel","Hcal Sum in the IsoCone for Barrel;E (GeV)",sumBin,sumMin,sumMax);
    book2DHistoVector(    h_hcalSumEEndcap_,"1D","hcalSumEEndcap","Hcal Sum in the IsoCone for Endcap;E (GeV)",sumBin,sumMin,sumMax);
    if (standAlone_)  book2DHistoVector(    h_hcalSumVsEt_  ,"2D","hcalSumVsEt2D","Hcal Sum in the Iso Cone;E_{T} (GeV);E (GeV)",reducedEtBin,etMin, etMax,reducedSumBin,sumMin,sumMax);
    book3DHistoVector(    p_hcalSumVsEt_  ,"Profile","hcalSumVsEt","Avg Hcal Sum in the Iso Cone vs E_{T};E_{T} (GeV);E (GeV)",etBin,etMin, etMax,sumBin,sumMin,sumMax);
    if (standAlone_)  book2DHistoVector(    h_hcalSumVsEta_ ,"2D","hcalSumVsEta2D","Hcal Sum in the Iso Cone;#eta;E (GeV)",reducedEtaBin,etaMin, etaMax,reducedSumBin,sumMin,sumMax);
    book2DHistoVector(    p_hcalSumVsEta_ ,"Profile","hcalSumVsEta","Avg Hcal Sum in the Iso Cone vs #eta;#eta;E (GeV)",etaBin,etaMin, etaMax,sumBin,sumMin,sumMax);

    //h over e
    book3DHistoVector(    h_hOverE_       ,"1D","hOverE","H/E;H/E",hOverEBin,hOverEMin,hOverEMax);
    book2DHistoVector(    p_hOverEVsEt_   ,"Profile","hOverEVsEt","Avg H/E vs Et;E_{T} (GeV);H/E",etBin,etMin,etMax,hOverEBin,hOverEMin,hOverEMax);
    book2DHistoVector(    p_hOverEVsEta_  ,"Profile","hOverEVsEta","Avg H/E vs #eta;#eta;H/E",etaBin,etaMin,etaMax,hOverEBin,hOverEMin,hOverEMax);
    book3DHistoVector(    h_h1OverE_      ,"1D","h1OverE","H/E for Depth 1;H/E",hOverEBin,hOverEMin,hOverEMax);
    book3DHistoVector(    h_h2OverE_      ,"1D","h2OverE","H/E for Depth 2;H/E",hOverEBin,hOverEMin,hOverEMax);

    // pf isolation
    book2DHistoVector(  h_phoIsoBarrel_,"1D","phoIsoBarrel","PF photon iso Barrel;E (GeV)",reducedEtBin,etMin,25.);
    book2DHistoVector(  h_phoIsoEndcap_,"1D","phoIsoEndcap","PF photon iso Endcap;E (GeV)",reducedEtBin,etMin,25.);
    book2DHistoVector(  h_chHadIsoBarrel_,"1D","chHadIsoBarrel","PF charged Had iso Barrel;E (GeV)",reducedEtBin,etMin,25.);
    book2DHistoVector(  h_chHadIsoEndcap_,"1D","chHadIsoEndcap","PF charged Had iso Endcap;E (GeV)",reducedEtBin,etMin,25.);
    book2DHistoVector(  h_nHadIsoBarrel_,"1D","neutralHadIsoBarrel","PF neutral Had iso Barrel;E (GeV)",reducedEtBin,etMin,25.);
    book2DHistoVector(  h_nHadIsoEndcap_,"1D","neutralHadIsoEndcap","PF neutral Had iso Endcap;E (GeV)",reducedEtBin,etMin,25.);
  


    //OTHER VARIABLES
    //bad channel histograms
    book2DHistoVector(    h_phoEt_BadChannels_  ,"1D","phoEtBadChannels", "Fraction Containing Bad Channels: E_{T};E_{T} (GeV)",etBin,etMin,etMax);
    book2DHistoVector(    h_phoEta_BadChannels_ ,"1D","phoEtaBadChannels","Fraction Containing Bad Channels: #eta;#eta",etaBin,etaMin,etaMax);
    book2DHistoVector(    h_phoPhi_BadChannels_ ,"1D","phoPhiBadChannels","Fraction Containing Bad Channels: #phi;#phi",phiBin,phiMin,phiMax);


    ////////////////START OF BOOKING FOR CONVERSION-RELATED HISTOGRAMS////////////////

    dbe_->setCurrentFolder("Egamma/"+fName_+"/AllPhotons/Et Above 0 GeV/Conversions");

    //ENERGY VARIABLES

    book3DHistoVector(    h_phoConvE_  ,"1D","phoConvE","E;E (GeV)",eBin,eMin,eMax);
    book3DHistoVector(    h_phoConvEt_ ,"1D","phoConvEt","E_{T};E_{T} (GeV)",etBin,etMin,etMax);

    //GEOMETRICAL VARIABLES

    book2DHistoVector(    h_phoConvEta_ ,"1D","phoConvEta","#eta;#eta",etaBin,etaMin,etaMax);
    book3DHistoVector(    h_phoConvPhi_ ,"1D","phoConvPhi","#phi;#phi",phiBin,phiMin,phiMax);

    //NUMBER OF PHOTONS

    book3DHistoVector(    h_nConv_ ,"1D","nConv","Number Of Conversions per Event ;# conversions",numberBin,numberMin,numberMax);

    //SHOWER SHAPE VARIABLES

    book3DHistoVector(    h_phoConvR9_ ,"1D","phoConvR9","R9;R9",r9Bin,r9Min,r9Max);

    //TRACK RELATED VARIABLES

    book3DHistoVector(    h_eOverPTracks_ ,"1D","eOverPTracks","E/P;E/P",eOverPBin,eOverPMin,eOverPMax);
    book3DHistoVector(    h_pOverETracks_ ,"1D","pOverETracks","P/E;P/E",eOverPBin,eOverPMin,eOverPMax);

    book3DHistoVector(    h_dPhiTracksAtVtx_  ,"1D","dPhiTracksAtVtx", "#Delta#phi of Tracks at Vertex;#Delta#phi",dPhiTracksBin,dPhiTracksMin,dPhiTracksMax);
    book3DHistoVector(    h_dPhiTracksAtEcal_ ,"1D","dPhiTracksAtEcal", "Abs(#Delta#phi) of Tracks at Ecal;#Delta#phi",dPhiTracksBin,0.,dPhiTracksMax);
    book3DHistoVector(    h_dEtaTracksAtEcal_ ,"1D","dEtaTracksAtEcal", "#Delta#eta of Tracks at Ecal;#Delta#eta",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax);

    book3DHistoVector(    h_dCotTracks_      ,"1D","dCotTracks","#Deltacot(#theta) of Tracks;#Deltacot(#theta)",dEtaTracksBin,dEtaTracksMin,dEtaTracksMax);
    book2DHistoVector(    p_dCotTracksVsEta_ ,"Profile","dCotTracksVsEta","Avg #Deltacot(#theta) of Tracks vs #eta;#eta;#Deltacot(#theta)",etaBin,etaMin,etaMax,dEtaTracksBin,dEtaTracksMin,dEtaTracksMax);

    book2DHistoVector(    p_nHitsVsEta_ ,"Profile","nHitsVsEta","Avg Number of Hits per Track vs #eta;#eta;# hits",etaBin,etaMin,etaMax,etaBin,0,16);

    book2DHistoVector(    h_tkChi2_      ,"1D","tkChi2","#chi^{2} of Track Fitting;#chi^{2}",chi2Bin,chi2Min,chi2Max);
    book2DHistoVector(    p_tkChi2VsEta_ ,"Profile","tkChi2VsEta","Avg #chi^{2} of Track Fitting vs #eta;#eta;#chi^{2}",etaBin,etaMin,etaMax,chi2Bin,chi2Min,chi2Max);

    //VERTEX RELATED VARIABLES

    book2DHistoVector(    h_convVtxRvsZ_ ,"2D","convVtxRvsZ","Vertex Position;Z (cm);R (cm)",500,zMin,zMax,rBin,rMin,rMax);
    book2DHistoVector(    h_convVtxZEndcap_    ,"1D","convVtxZEndcap",   "Vertex Position: #eta > 1.5;Z (cm)",zBin,zMin,zMax);
    book2DHistoVector(    h_convVtxZ_    ,"1D","convVtxZ",   "Vertex Position;Z (cm)",zBin,zMin,zMax);
    book2DHistoVector(    h_convVtxR_    ,"1D","convVtxR",   "Vertex Position: #eta < 1;R (cm)",rBin,rMin,rMax);
    book2DHistoVector(    h_convVtxYvsX_ ,"2D","convVtxYvsX","Vertex Position: #eta < 1;X (cm);Y (cm)",xBin,xMin,xMax,yBin,yMin,yMax);



    book2DHistoVector(    h_vertexChi2Prob_ ,"1D","vertexChi2Prob","#chi^{2} Probability of Vertex Fitting;#chi^{2}",100,0.,1.0);


  }//end if(dbe_)


}//end BeginJob



void PhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{
  using namespace edm;

  if (nEvt_% prescaleFactor_ ) return;
  nEvt_++;
  LogInfo(fName_) << "PhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";

  // Get the trigger results
  bool validTriggerEvent=true;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  trigger::TriggerEvent triggerEvent;
  e.getByToken(triggerEvent_token_,triggerEventHandle);
  if(!triggerEventHandle.isValid()) {
    edm::LogInfo(fName_) << "Error! Can't get the product: triggerEvent_" << endl;
    validTriggerEvent=false;
  }
  if(validTriggerEvent) triggerEvent = *(triggerEventHandle.product());

  // Get the reconstructed photons
  //  bool validPhotons=true;
  Handle<reco::PhotonCollection> photonHandle;
  reco::PhotonCollection photonCollection;
  e.getByToken(photon_token_ , photonHandle);
  if ( !photonHandle.isValid()) {
    edm::LogInfo(fName_) << "Error! Can't get the product: photon_token_" << endl;
    // validPhotons=false;
  }
  //  if(validPhotons) photonCollection = *(photonHandle.product());

  // Get the PhotonId objects
  bool validloosePhotonID=true;
  Handle<edm::ValueMap<bool> > loosePhotonFlag;
  edm::ValueMap<bool> loosePhotonID;
  e.getByToken(PhotonIDLoose_token_, loosePhotonFlag);
  if ( !loosePhotonFlag.isValid()) {
    edm::LogInfo(fName_) << "Error! Can't get the product: PhotonIDLoose_token_" << endl;
    validloosePhotonID=false;
  }
  if (validloosePhotonID) loosePhotonID = *(loosePhotonFlag.product());

  bool validtightPhotonID=true;
  Handle<edm::ValueMap<bool> > tightPhotonFlag;
  edm::ValueMap<bool> tightPhotonID;
  e.getByToken(PhotonIDTight_token_, tightPhotonFlag);
  if ( !tightPhotonFlag.isValid()) {
    edm::LogInfo(fName_) << "Error! Can't get the product: PhotonIDTight_token_" << endl;
    validtightPhotonID=false;
  }
  if (validtightPhotonID) tightPhotonID = *(tightPhotonFlag.product());


  edm::Handle<reco::VertexCollection> vtxH;
  if ( !isHeavyIon_) { 
    e.getByToken(offline_pvToken_, vtxH);
    h_nRecoVtx_ ->Fill (float(vtxH->size()));
  }

  // Create array to hold #photons/event information
  int nPho[100][3][3];

  for (int cut=0; cut!=100; ++cut){
    for (unsigned int type=0; type!=types_.size(); ++type){
      for (unsigned int part=0; part!=parts_.size(); ++part){
	nPho[cut][type][part] = 0;
      }
    }
  }
  // Create array to hold #conversions/event information
  int nConv[100][3][3];

  for (int cut=0; cut!=100; ++cut){
    for (unsigned int type=0; type!=types_.size(); ++type){
      for (unsigned int part=0; part!=parts_.size(); ++part){
	nConv[cut][type][part] = 0;
      }
    }
  }



  //Prepare list of photon-related HLT filter names

  vector<int> Keys;

  for(uint filterIndex=0;filterIndex<triggerEvent.sizeFilters();++filterIndex){  //loop over all trigger filters in event (i.e. filters passed)

    string label = triggerEvent.filterTag(filterIndex).label();

    if(label.find( "Photon" ) != string::npos ) {  //get photon-related filters

      for(uint filterKeyIndex=0;filterKeyIndex<triggerEvent.filterKeys(filterIndex).size();++filterKeyIndex){  //loop over keys to objects passing this filter
	Keys.push_back(triggerEvent.filterKeys(filterIndex)[filterKeyIndex]);  //add keys to a vector for later reference
      }

    }

  }

  // sort Keys vector in ascending order
  // and erases duplicate entries from the vector
  sort(Keys.begin(),Keys.end());
  for ( uint i=0 ; i<Keys.size() ; )
   {
    if (i!=(Keys.size()-1))
     {
      if (Keys[i]==Keys[i+1]) Keys.erase(Keys.begin()+i+1) ;
      else ++i ;
     }
    else ++i ;
   }

  //We now have a vector of unique keys to TriggerObjects passing a photon-related filter

  // old int photonCounter = 0;

  /////////////////////////BEGIN LOOP OVER THE COLLECTION OF PHOTONS IN THE EVENT/////////////////////////
  for(unsigned int iPho=0; iPho < photonHandle->size(); iPho++) {
    reco::PhotonRef aPho(reco::PhotonRef(photonHandle, iPho));
    //  for( reco::PhotonCollection::const_iterator  iPho = photonCollection.begin(); iPho != photonCollection.end(); iPho++) {


    //for HLT efficiency plots

    h_phoEta_preHLT_->Fill(aPho->eta());
    h_phoEt_preHLT_->Fill( aPho->et());


    double deltaR=1000.;
    double deltaRMin=1000.;
    double deltaRMax=0.05;//sets deltaR threshold for matching photons to trigger objects


    for (vector<int>::const_iterator objectKey=Keys.begin();objectKey!=Keys.end();objectKey++){  //loop over keys to objects that fired photon triggers

      deltaR = reco::deltaR(triggerEvent.getObjects()[(*objectKey)].eta(),triggerEvent.getObjects()[(*objectKey)].phi(),aPho->superCluster()->eta(),aPho->superCluster()->phi());
      if(deltaR < deltaRMin) deltaRMin = deltaR;

    }

    if(deltaRMin > deltaRMax) {  //photon fails delta R cut
      if(useTriggerFiltering_) continue;  //throw away photons that haven't passed any photon filters
    }

    if(deltaRMin <= deltaRMax) { //photon passes delta R cut
      h_phoEta_postHLT_->Fill(aPho->eta() );
      h_phoEt_postHLT_->Fill( aPho->et() );
    }

    //    if (aPho->et()  < minPhoEtCut_) continue;
    bool isLoosePhoton(false), isTightPhoton(false);
    if ( photonSelection (aPho) )  isLoosePhoton=true ;


    nEntry_++;

    // old     edm::Ref<reco::PhotonCollection> photonref(photonHandle, photonCounter);
    // old photonCounter++;


    // old   if ( !isHeavyIon_ ) {
    // isLoosePhoton = (loosePhotonID)[photonref];
    // isTightPhoton = (tightPhotonID)[photonref];
    // }


    //find out which part of the Ecal contains the photon
    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;
    float etaPho = aPho->superCluster()->eta();
    if ( fabs(etaPho) <  1.479 )
      phoIsInBarrel=true;
    else {
      phoIsInEndcap=true;
    }

    int part = 0;
    if ( phoIsInBarrel ) part = 1;
    if ( phoIsInEndcap ) part = 2;

    /////  From 30X on, Photons are already pre-selected at reconstruction level with a looseEM isolation
    bool isIsolated=false;
    if ( isolationStrength_ == 0)  isIsolated = isLoosePhoton;
    if ( isolationStrength_ == 1)  isIsolated = isTightPhoton;

    int type=0;
    if ( isIsolated ) type=1;
    if ( !excludeBkgHistos_ && !isIsolated ) type=2;


    //get rechit collection containing this photon
    bool validEcalRecHits=true;
    edm::Handle<EcalRecHitCollection>   ecalRecHitHandle;
    EcalRecHitCollection ecalRecHitCollection;
    if ( phoIsInBarrel ) {
      // Get handle to barrel rec hits
      e.getByToken(barrelRecHit_token_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
	edm::LogError(fName_) << "Error! Can't get the product: barrelRecHit_token_";
	validEcalRecHits=false;
      }
    }
    else if ( phoIsInEndcap ) {
      // Get handle to endcap rec hits
      e.getByToken(endcapRecHit_token_, ecalRecHitHandle);
      if (!ecalRecHitHandle.isValid()) {
	edm::LogError(fName_) << "Error! Can't get the product: endcapRecHit_token";
	validEcalRecHits=false;
      }
    }
    if (validEcalRecHits) ecalRecHitCollection = *(ecalRecHitHandle.product());


    //if (aPho->isEBEEGap()) continue;  //cut out gap photons


    //filling histograms to make isolation efficiencies
    if(isLoosePhoton){
      h_phoEta_Loose_->Fill(aPho->eta());
      h_phoEt_Loose_->Fill( aPho->et() );
    }
    if(isTightPhoton){
      h_phoEta_Tight_->Fill(aPho->eta());
      h_phoEt_Tight_->Fill( aPho->et() );
    }



    for (int cut = 0; cut !=numberOfSteps_; ++cut) {  //loop over different transverse energy cuts
      double Et =  aPho->et();
      bool passesCuts = false;

      //sorting the photon into the right Et-dependant folder
      if ( useBinning_ && Et > (cut+1)*cutStep_ && ( (Et < (cut+2)*cutStep_)  | (cut == numberOfSteps_-1) ) ){
	passesCuts = true;
      }
      else if ( !useBinning_ && Et > (cut+1)*cutStep_ ){
	passesCuts = true;
      }

      if (passesCuts){

	//filling isolation variable histograms

	//tracker isolation variables	
	fill2DHistoVector(h_nTrackIsolSolid_, aPho->nTrkSolidConeDR04(), cut,type);
	fill2DHistoVector(h_nTrackIsolHollow_,aPho->nTrkHollowConeDR04(),cut,type);

	if (standAlone_)  fill2DHistoVector(h_nTrackIsolSolidVsEta_, aPho->eta(),aPho->nTrkSolidConeDR04(), cut,type);
	fill2DHistoVector(p_nTrackIsolSolidVsEta_, aPho->eta(),aPho->nTrkSolidConeDR04(), cut,type);
	if (standAlone_)  fill2DHistoVector(h_nTrackIsolHollowVsEta_,aPho->eta(),aPho->nTrkHollowConeDR04(),cut,type);
	fill2DHistoVector(p_nTrackIsolHollowVsEta_,aPho->eta(),aPho->nTrkHollowConeDR04(),cut,type);

	if (standAlone_)  fill2DHistoVector(h_nTrackIsolSolidVsEt_,  aPho->et(), aPho->nTrkSolidConeDR04(), cut,type);
	fill2DHistoVector(p_nTrackIsolSolidVsEt_,  aPho->et(), aPho->nTrkSolidConeDR04(), cut,type);
	if (standAlone_)  fill2DHistoVector(h_nTrackIsolHollowVsEt_, aPho->et(), aPho->nTrkHollowConeDR04(),cut,type);
	fill2DHistoVector(p_nTrackIsolHollowVsEt_, aPho->et(), aPho->nTrkHollowConeDR04(),cut,type);

	///////
	fill2DHistoVector(h_trackPtSumSolid_, aPho->trkSumPtSolidConeDR04(),cut,type);
	fill2DHistoVector(h_trackPtSumHollow_,aPho->trkSumPtSolidConeDR04(),cut,type);

	if (standAlone_)  fill2DHistoVector(h_trackPtSumSolidVsEta_, aPho->eta(),aPho->trkSumPtSolidConeDR04(), cut,type);
	fill2DHistoVector(p_trackPtSumSolidVsEta_, aPho->eta(),aPho->trkSumPtSolidConeDR04(), cut,type);
	if (standAlone_)  fill2DHistoVector(h_trackPtSumHollowVsEta_,aPho->eta(),aPho->trkSumPtHollowConeDR04(),cut,type);
	fill2DHistoVector(p_trackPtSumHollowVsEta_,aPho->eta(),aPho->trkSumPtHollowConeDR04(),cut,type);

	if (standAlone_)  fill2DHistoVector(h_trackPtSumSolidVsEt_,  aPho->et(), aPho->trkSumPtSolidConeDR04(), cut,type);
	fill2DHistoVector(p_trackPtSumSolidVsEt_,  aPho->et(), aPho->trkSumPtSolidConeDR04(), cut,type);
	if (standAlone_)  fill2DHistoVector(h_trackPtSumHollowVsEt_, aPho->et(), aPho->trkSumPtHollowConeDR04(),cut,type);
	fill2DHistoVector(p_trackPtSumHollowVsEt_, aPho->et(), aPho->trkSumPtHollowConeDR04(),cut,type);
	//calorimeter isolation variables

	fill2DHistoVector(h_ecalSum_,aPho->ecalRecHitSumEtConeDR04(),cut,type);
	if(aPho->isEB()){fill2DHistoVector(h_ecalSumEBarrel_,aPho->ecalRecHitSumEtConeDR04(),cut,type);}
        if(aPho->isEE()){fill2DHistoVector(h_ecalSumEEndcap_,aPho->ecalRecHitSumEtConeDR04(),cut,type);}
	if (standAlone_)  fill2DHistoVector(h_ecalSumVsEta_,aPho->eta(),aPho->ecalRecHitSumEtConeDR04(),cut,type);
	fill2DHistoVector(p_ecalSumVsEta_,aPho->eta(),aPho->ecalRecHitSumEtConeDR04(),cut,type);
 	if (standAlone_)  fill2DHistoVector(h_ecalSumVsEt_, aPho->et(), aPho->ecalRecHitSumEtConeDR04(),cut,type);
	fill3DHistoVector(p_ecalSumVsEt_, aPho->et(), aPho->ecalRecHitSumEtConeDR04(),cut,type,part);

	///////

	fill2DHistoVector(h_hcalSum_,aPho->hcalTowerSumEtConeDR04(),cut,type);
	if(aPho->isEB()){fill2DHistoVector(h_hcalSumEBarrel_,aPho->hcalTowerSumEtConeDR04(),cut,type);}
        if(aPho->isEE()){fill2DHistoVector(h_hcalSumEEndcap_,aPho->hcalTowerSumEtConeDR04(),cut,type);}
	if (standAlone_)  fill2DHistoVector(h_hcalSumVsEta_,aPho->eta(),aPho->hcalTowerSumEtConeDR04(),cut,type);
	fill2DHistoVector(p_hcalSumVsEta_,aPho->eta(),aPho->hcalTowerSumEtConeDR04(),cut,type);
 	if (standAlone_)  fill2DHistoVector(h_hcalSumVsEt_, aPho->et(), aPho->hcalTowerSumEtConeDR04(),cut,type);
	fill3DHistoVector(p_hcalSumVsEt_, aPho->et(), aPho->hcalTowerSumEtConeDR04(),cut,type,part);

	fill3DHistoVector(h_hOverE_,aPho->hadronicOverEm(),cut,type,part);
	fill2DHistoVector(p_hOverEVsEta_,aPho->eta(),aPho->hadronicOverEm(),cut,type);
	fill2DHistoVector(p_hOverEVsEt_, aPho->et(), aPho->hadronicOverEm(),cut,type);

	fill3DHistoVector(h_h1OverE_,aPho->hadronicDepth1OverEm(),cut,type,part);
	fill3DHistoVector(h_h2OverE_,aPho->hadronicDepth2OverEm(),cut,type,part);

 
	// filling pf isolation variables
	if(aPho->isEB()) { 
	  fill2DHistoVector( h_phoIsoBarrel_, aPho->photonIso(),cut,type);
	  fill2DHistoVector( h_chHadIsoBarrel_, aPho->chargedHadronIso(),cut,type);
	  fill2DHistoVector( h_nHadIsoBarrel_, aPho->neutralHadronIso(),cut,type);
	}
	if(aPho->isEE()) {
	  fill2DHistoVector( h_phoIsoEndcap_, aPho->photonIso(),cut,type);
	  fill2DHistoVector( h_chHadIsoEndcap_, aPho->chargedHadronIso(),cut,type);
	  fill2DHistoVector( h_nHadIsoEndcap_, aPho->neutralHadronIso(),cut,type);
	}


  	//filling photon histograms

	nPho[cut][0][0]++;
	nPho[cut][0][part]++;
	nPho[cut][type][0]++;
	nPho[cut][type][part]++;

	//energy variables

	fill3DHistoVector(h_phoE_, aPho->energy(),cut,type,part);
	fill3DHistoVector(h_phoSigmaEoverE_, aPho->getCorrectedEnergyError(aPho->getCandidateP4type())/aPho->energy(),cut,type,part);

	if ( !isHeavyIon_) fill3DHistoVector(p_phoSigmaEoverEvsNVtx_, float(vtxH->size()),  aPho->getCorrectedEnergyError(aPho->getCandidateP4type())/aPho->energy(),cut,type,part);
 
	fill3DHistoVector(h_phoEt_,aPho->et(),    cut,type,part);

	//geometrical variables

	fill2DHistoVector(h_phoEta_,aPho->eta(),cut,type);
	fill2DHistoVector(h_scEta_, aPho->superCluster()->eta(),cut,type);

	fill3DHistoVector(h_phoPhi_,aPho->phi(),cut,type,part);
	fill3DHistoVector(h_scPhi_, aPho->superCluster()->phi(),cut,type,part);

	//shower shape variables

	fill3DHistoVector(h_r9_,aPho->r9(),cut,type,part);
	if (standAlone_)  fill2DHistoVector(h_r9VsEta_,aPho->eta(),aPho->r9(),cut,type);
	fill2DHistoVector(p_r9VsEta_,aPho->eta(),aPho->r9(),cut,type);
	if (standAlone_)  fill2DHistoVector(h_r9VsEt_, aPho->et(), aPho->r9(),cut,type);
	fill2DHistoVector(p_r9VsEt_, aPho->et(), aPho->r9(),cut,type);

	if (standAlone_)  fill2DHistoVector(h_e1x5VsEta_,aPho->eta(),aPho->e1x5(),cut,type);
	fill2DHistoVector(p_e1x5VsEta_,aPho->eta(),aPho->e1x5(),cut,type);
 	if (standAlone_)  fill2DHistoVector(h_e1x5VsEt_, aPho->et(), aPho->e1x5(),cut,type);
 	fill2DHistoVector(p_e1x5VsEt_, aPho->et(), aPho->e1x5(),cut,type);

	if (standAlone_)  fill2DHistoVector(h_e2x5VsEta_,aPho->eta(),aPho->e2x5(),cut,type);
	fill2DHistoVector(p_e2x5VsEta_,aPho->eta(),aPho->e2x5(),cut,type);
	if (standAlone_)  fill2DHistoVector(h_e2x5VsEt_, aPho->et(), aPho->e2x5(),cut,type);
	fill2DHistoVector(p_e2x5VsEt_, aPho->et(), aPho->e2x5(),cut,type);

	if (standAlone_)  fill2DHistoVector(h_maxEXtalOver3x3VsEta_,aPho->eta(),aPho->maxEnergyXtal()/aPho->e3x3(),cut,type);
	fill2DHistoVector(p_maxEXtalOver3x3VsEta_,aPho->eta(),aPho->maxEnergyXtal()/aPho->e3x3(),cut,type);
	if (standAlone_)  fill2DHistoVector(h_maxEXtalOver3x3VsEt_, aPho->et(), aPho->maxEnergyXtal()/aPho->e3x3(),cut,type);
	fill2DHistoVector(p_maxEXtalOver3x3VsEt_, aPho->et(), aPho->maxEnergyXtal()/aPho->e3x3(),cut,type);


	if (standAlone_)  fill2DHistoVector(h_r1x5VsEta_,aPho->eta(),aPho->r1x5(),cut,type);
	fill2DHistoVector(p_r1x5VsEta_,aPho->eta(),aPho->r1x5(),cut,type);
	if (standAlone_)  fill2DHistoVector(h_r1x5VsEt_, aPho->et(), aPho->r1x5(),cut,type);
	fill2DHistoVector(p_r1x5VsEt_, aPho->et(), aPho->r1x5(),cut,type);

	if (standAlone_)  fill2DHistoVector(h_r2x5VsEta_,aPho->eta(),aPho->r2x5(),cut,type);
	fill2DHistoVector(p_r2x5VsEta_,aPho->eta(),aPho->r2x5(),cut,type);
	if (standAlone_)  fill2DHistoVector(h_r2x5VsEt_, aPho->et(), aPho->r2x5(),cut,type);
	fill2DHistoVector(p_r2x5VsEt_, aPho->et(), aPho->r2x5(),cut,type);

	fill3DHistoVector(h_phoSigmaIetaIeta_,aPho->sigmaIetaIeta(),cut,type,part);
	if (standAlone_)  fill2DHistoVector(h_sigmaIetaIetaVsEta_,aPho->eta(),aPho->sigmaIetaIeta(),cut,type);
	fill2DHistoVector(p_sigmaIetaIetaVsEta_,aPho->eta(),aPho->sigmaIetaIeta(),cut,type);



	//filling histograms for photons containing a bad ECAL channel
 	bool atLeastOneDeadChannel=false;
 	for(reco::CaloCluster_iterator bcIt = aPho->superCluster()->clustersBegin();bcIt != aPho->superCluster()->clustersEnd(); ++bcIt) { //loop over basic clusters in SC
 	  for(vector< pair<DetId, float> >::const_iterator rhIt = (*bcIt)->hitsAndFractions().begin();rhIt != (*bcIt)->hitsAndFractions().end(); ++rhIt) { //loop over rec hits in basic cluster

 	    for(EcalRecHitCollection::const_iterator it = ecalRecHitCollection.begin(); it !=  ecalRecHitCollection.end(); ++it) { //loop over all rec hits to find the right ones
 	      if  (rhIt->first ==  (*it).id() ) { //found the matching rechit
 		if (  (*it).recoFlag() == 9 ) { //has a bad channel
 		  atLeastOneDeadChannel=true;
 		  break;
 		}
 	      }
 	    }
 	  }
 	}
	if ( atLeastOneDeadChannel ) {
	  fill2DHistoVector(h_phoPhi_BadChannels_,aPho->phi(),cut,type);
	  fill2DHistoVector(h_phoEta_BadChannels_,aPho->eta(),cut,type);
	  fill2DHistoVector(h_phoEt_BadChannels_, aPho->et(), cut,type);
 	}


	// filling conversion-related histograms
	if(aPho->hasConversionTracks()){
	  nConv[cut][0][0]++;
	  nConv[cut][0][part]++;
	  nConv[cut][type][0]++;
	  nConv[cut][type][part]++;
	}

	//loop over conversions (don't forget, we're still inside the photon loop,
	// i.e. these are all the conversions for this ONE photon, not for all the photons in the event)
	reco::ConversionRefVector conversions = aPho->conversions();
	for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

	  reco::ConversionRef aConv=conversions[iConv];

	  if ( aConv->nTracks() <2 ) continue;

	  //fill histogram for denominator of vertex reconstruction efficiency plot
	  if(cut==0) h_phoEta_Vertex_->Fill(aConv->refittedPairMomentum().eta());

	  if ( !(aConv->conversionVertex().isValid()) ) continue;

  	  float chi2Prob = ChiSquaredProbability( aConv->conversionVertex().chi2(), aConv->conversionVertex().ndof() );

	  if(chi2Prob<0.0005) continue;

	  fill2DHistoVector(h_vertexChi2Prob_,chi2Prob,cut,type);



	  fill3DHistoVector(h_phoConvE_, aPho->energy(),cut,type,part);
	  fill3DHistoVector(h_phoConvEt_,aPho->et(),cut,type,part);
	  fill3DHistoVector(h_phoConvR9_,aPho->r9(),cut,type,part);

	  if (cut==0 && isLoosePhoton){
	    h_convEta_Loose_->Fill(aPho->eta());
	    h_convEt_Loose_->Fill( aPho->et() );
	  }
	  if (cut==0 && isTightPhoton){
	    h_convEta_Tight_->Fill(aPho->eta());
	    h_convEt_Tight_->Fill( aPho->et() );
	  }

	  fill2DHistoVector(h_phoConvEta_,aConv->refittedPairMomentum().eta(),cut,type);
	  fill3DHistoVector(h_phoConvPhi_,aConv->refittedPairMomentum().phi(),cut,type,part);

	
	  //we use the photon position because we'll be dividing it by a photon histogram (not a conversion histogram)
 	  fill2DHistoVector(h_phoConvEtaForEfficiency_,aPho->eta(),cut,type);
 	  fill3DHistoVector(h_phoConvPhiForEfficiency_,aPho->phi(),cut,type,part);


	  //vertex histograms
	  double convR= sqrt(aConv->conversionVertex().position().perp2());
	  double scalar = aConv->conversionVertex().position().x()*aConv->refittedPairMomentum().x() + aConv->conversionVertex().position().y()*aConv->refittedPairMomentum().y();
	  if ( scalar < 0 ) convR= -convR;

	  fill2DHistoVector(h_convVtxRvsZ_,aConv->conversionVertex().position().z(), convR,cut,type);//trying to "see" R-Z view of tracker
	  fill2DHistoVector(h_convVtxZ_,aConv->conversionVertex().position().z(), cut,type);

	  
	  if( fabs(aPho->eta()) > 1.5){//trying to "see" tracker endcaps
	    fill2DHistoVector(h_convVtxZEndcap_,aConv->conversionVertex().position().z(), cut,type);
	  }
	  else if(fabs(aPho->eta()) < 1){//trying to "see" tracker barrel
	    fill2DHistoVector(h_convVtxR_,convR,cut,type);
	    fill2DHistoVector(h_convVtxYvsX_,aConv->conversionVertex().position().x(),aConv->conversionVertex().position().y(),cut,type);
	  }

	  
	  const std::vector<edm::RefToBase<reco::Track> > tracks = aConv->tracks();


	  for (unsigned int i=0; i<tracks.size(); i++) {
	    fill2DHistoVector(h_tkChi2_,tracks[i]->normalizedChi2(),cut,type);
	    fill2DHistoVector(p_tkChi2VsEta_,aPho->eta(),tracks[i]->normalizedChi2(),cut,type);
	    fill2DHistoVector(p_dCotTracksVsEta_,aPho->eta(),aConv->pairCotThetaSeparation(),cut,type);
	    fill2DHistoVector(p_nHitsVsEta_,aPho->eta(),float(tracks[i]->numberOfValidHits()),cut,type);
	  }

	  //calculating delta eta and delta phi of the two tracks

	  float  DPhiTracksAtVtx = -99;
	  float  dPhiTracksAtEcal= -99;
	  float  dEtaTracksAtEcal= -99;

	  float phiTk1= aConv->tracksPin()[0].phi();
	  float phiTk2= aConv->tracksPin()[1].phi();
	  DPhiTracksAtVtx = phiTk1-phiTk2;
	  DPhiTracksAtVtx = phiNormalization( DPhiTracksAtVtx );

	  if (aConv->bcMatchingWithTracks().size() > 0 && aConv->bcMatchingWithTracks()[0].isNonnull() && aConv->bcMatchingWithTracks()[1].isNonnull() ) {
	    float recoPhi1 = aConv->ecalImpactPosition()[0].phi();
	    float recoPhi2 = aConv->ecalImpactPosition()[1].phi();
	    float recoEta1 = aConv->ecalImpactPosition()[0].eta();
	    float recoEta2 = aConv->ecalImpactPosition()[1].eta();

	    recoPhi1 = phiNormalization(recoPhi1);
	    recoPhi2 = phiNormalization(recoPhi2);

	    dPhiTracksAtEcal = recoPhi1 -recoPhi2;
	    dPhiTracksAtEcal = phiNormalization( dPhiTracksAtEcal );
	    dEtaTracksAtEcal = recoEta1 -recoEta2;

	  }

	
	  fill3DHistoVector(h_dPhiTracksAtVtx_,DPhiTracksAtVtx,cut,type,part);
	  fill3DHistoVector(h_dPhiTracksAtEcal_,fabs(dPhiTracksAtEcal),cut,type,part);
	  fill3DHistoVector(h_dEtaTracksAtEcal_, dEtaTracksAtEcal,cut,type,part);
	  fill3DHistoVector(h_eOverPTracks_,aConv->EoverPrefittedTracks(),cut,type,part);
	  fill3DHistoVector(h_pOverETracks_,1./aConv->EoverPrefittedTracks(),cut,type,part);
	  fill3DHistoVector(h_dCotTracks_,aConv->pairCotThetaSeparation(),cut,type,part);

	}//end loop over conversions

      }//end loop over photons passing cuts
    }//end loop over transverse energy cuts


 


    //make invariant mass plots

    if (isIsolated && aPho->et()>=invMassEtCut_){
      for(unsigned int iPho2=iPho+1; iPho2 < photonHandle->size(); iPho2++) {
	reco::PhotonRef aPho2(reco::PhotonRef(photonHandle, iPho2));

	//      for (reco::PhotonCollection::const_iterator iPho2=iPho+1; iPho2!=photonCollection.end(); iPho2++){

	//	edm::Ref<reco::PhotonCollection> photonref2(photonHandle, photonCounter); //note: it's correct to use photonCounter and not photonCounter+1
	                                                                          //since it has already been incremented earlier

	bool  isTightPhoton2(false), isLoosePhoton2(false);
	if ( photonSelection (aPho2) ) isLoosePhoton2=true;

	// Old if ( !isHeavyIon_ ) {
	//  isTightPhoton2 = (tightPhotonID)[aPho2];
	// isLoosePhoton2 = (loosePhotonID)[aPho2];
	//	}

	bool isIsolated2=false;
	if ( isolationStrength_ == 0)  isIsolated2 = isLoosePhoton2;
	if ( isolationStrength_ == 1)  isIsolated2 = isTightPhoton2;

	reco::ConversionRefVector conversions = aPho->conversions();
	reco::ConversionRefVector conversions2 = aPho2->conversions();

 	if(isIsolated2 && aPho2->et()>=invMassEtCut_){

	  math::XYZTLorentzVector p12 = aPho->p4()+aPho2->p4();
	  float gamgamMass2 = p12.Dot(p12);


	  h_invMassAllPhotons_ -> Fill(sqrt( gamgamMass2 ));
	  if(aPho->isEB() && aPho2->isEB()){h_invMassPhotonsEBarrel_ -> Fill(sqrt( gamgamMass2 ));}
	  if(aPho->isEE() || aPho2->isEE()){h_invMassPhotonsEEndcap_ -> Fill(sqrt( gamgamMass2 ));}      
	  
 	  if(conversions.size()!=0 && conversions[0]->nTracks() >= 2){
	    if(conversions2.size()!=0 && conversions2[0]->nTracks() >= 2) h_invMassTwoWithTracks_ -> Fill(sqrt( gamgamMass2 ));
	    else h_invMassOneWithTracks_ -> Fill(sqrt( gamgamMass2 ));
 	  }
	  else if(conversions2.size()!=0 && conversions2[0]->nTracks() >= 2) h_invMassOneWithTracks_ -> Fill(sqrt( gamgamMass2 ));
	  else h_invMassZeroWithTracks_ -> Fill(sqrt( gamgamMass2 ));
 	}

      }

    }



  }/// End loop over Reco photons


  //filling number of photons/conversions per event histograms
  for (int cut = 0; cut !=numberOfSteps_; ++cut) {
    for(uint type=0;type!=types_.size();++type){
      for(uint part=0;part!=parts_.size();++part){
	h_nPho_[cut][type][part]-> Fill (float(nPho[cut][type][part]));
	h_nConv_[cut][type][part]-> Fill (float(nConv[cut][type][part]));
      }
    }
  }

}//End of Analyze method

void PhotonAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& setup)
{
  if(!standAlone_){

   
    dbe_->setCurrentFolder("Egamma/"+fName_+"/");
    //keep track of how many histos are in each folder
    totalNumberOfHistos_efficiencyFolder->Fill(histo_index_efficiency_);
    totalNumberOfHistos_invMassFolder->Fill(histo_index_invMass_);
    totalNumberOfHistos_photonsFolder->Fill(histo_index_photons_);
    totalNumberOfHistos_conversionsFolder->Fill(histo_index_conversions_);
       
  }

}


void PhotonAnalyzer::endJob()
{
  //dbe_->showDirStructure();
  if(standAlone_){
    dbe_->setCurrentFolder("Egamma/"+fName_+"/");

    //keep track of how many histos are in each folder
    totalNumberOfHistos_efficiencyFolder->Fill(histo_index_efficiency_);
    totalNumberOfHistos_invMassFolder->Fill(histo_index_invMass_);
    totalNumberOfHistos_photonsFolder->Fill(histo_index_photons_);
    totalNumberOfHistos_conversionsFolder->Fill(histo_index_conversions_);


    dbe_->save(outputFileName_);
  }


}

  ////////////BEGIN AUXILIARY FUNCTIONS//////////////



float PhotonAnalyzer::phiNormalization(float & phi)
{
 const float PI    = 3.1415927;
 const float TWOPI = 2.0*PI;

 if(phi >  PI) {phi = phi - TWOPI;}
 if(phi < -PI) {phi = phi + TWOPI;}

 return phi;
}


void  PhotonAnalyzer::fill2DHistoVector(vector<vector<MonitorElement*> >& histoVector,double x, double y, int cut, int type){

  histoVector[cut][0]->Fill(x,y);
  if(histoVector[cut].size()>1)   histoVector[cut][type]->Fill(x,y); //don't try to fill 2D histos that are only in the "AllPhotons" folder

}

void  PhotonAnalyzer::fill2DHistoVector(vector<vector<MonitorElement*> >& histoVector, double x, int cut, int type){

  histoVector[cut][0]->Fill(x);
  histoVector[cut][type]->Fill(x);

}

void  PhotonAnalyzer::fill3DHistoVector(vector<vector<vector<MonitorElement*> > >& histoVector,double x, int cut, int type, int part){

  histoVector[cut][0][0]->Fill(x);
  histoVector[cut][0][part]->Fill(x);
  histoVector[cut][type][0]->Fill(x);
  histoVector[cut][type][part]->Fill(x);

}

void  PhotonAnalyzer::fill3DHistoVector(vector<vector<vector<MonitorElement*> > >& histoVector,double x, double y, int cut, int type, int part){

  histoVector[cut][0][0]->Fill(x,y);
  histoVector[cut][0][part]->Fill(x,y);
  histoVector[cut][type][0]->Fill(x,y);
  histoVector[cut][type][part]->Fill(x,y);

}


MonitorElement* PhotonAnalyzer::bookHisto(string histoName, string title, int bin, double min, double max)
{

  int histo_index = 0;
  stringstream histo_number_stream;

  //determining which folder we're in
  if(dbe_->pwd().find( "InvMass" ) != string::npos){
    histo_index_invMass_++;
    histo_index = histo_index_invMass_;
  }
  if(dbe_->pwd().find( "Efficiencies" ) != string::npos){
    histo_index_efficiency_++;
    histo_index = histo_index_efficiency_;
  }

  histo_number_stream << "h_";
  if(histo_index<10)   histo_number_stream << "0";
  histo_number_stream << histo_index;

  return dbe_->book1D(histo_number_stream.str()+"_"+histoName,title,bin,min,max);

}


void PhotonAnalyzer::book2DHistoVector(vector<vector<MonitorElement*> > &temp2DVector,
				       string histoType, string histoName, string title,
									     int xbin, double xmin,double xmax,
									     int ybin, double ymin, double ymax)
{
  int histo_index = 0;

  vector<MonitorElement*> temp1DVector;
//   vector<vector<MonitorElement*> > temp2DVector;

  //determining which folder we're in
  bool conversionPlot = false;
  if(dbe_->pwd().find( "Conversions" ) != string::npos) conversionPlot = true;
  bool TwoDPlot = false;
  if(histoName.find( "2D" ) != string::npos) TwoDPlot = true;

  if(conversionPlot){
    histo_index_conversions_++;
    histo_index = histo_index_conversions_;
  }
  else{
    histo_index_photons_++;
    histo_index = histo_index_photons_;
  }

  stringstream histo_number_stream;
  histo_number_stream << "h_";
  if(histo_index<10)   histo_number_stream << "0";
  histo_number_stream << histo_index << "_";



  for(int cut = 0; cut != numberOfSteps_; ++cut){ //looping over Et cut values

    for(uint type=0;type!=types_.size();++type){  //looping over isolation type

      currentFolder_.str("");
      currentFolder_ << "Egamma/"+fName_+"/" << types_[type] << "Photons/Et above " << (cut+1)*cutStep_ << " GeV";
      if(conversionPlot) currentFolder_ << "/Conversions";

      dbe_->setCurrentFolder(currentFolder_.str());

      string kind;
      if(conversionPlot) kind = " Conversions: ";
      else kind = " Photons: ";

      if(histoType=="1D")           temp1DVector.push_back(dbe_->book1D(      histo_number_stream.str()+histoName,types_[type]+kind+title,xbin,xmin,xmax));
      else if(histoType=="2D"){
	if((TwoDPlot && type==0) || !TwoDPlot){//only book the 2D plots in the "AllPhotons" folder
	                            temp1DVector.push_back(dbe_->book2D(      histo_number_stream.str()+histoName,types_[type]+kind+title,xbin,xmin,xmax,ybin,ymin,ymax));
	}
      }
      else if(histoType=="Profile") temp1DVector.push_back(dbe_->bookProfile( histo_number_stream.str()+histoName,types_[type]+kind+title,xbin,xmin,xmax,ybin,ymin,ymax,""));
      else cout << "bad histoType\n";
    }

    temp2DVector.push_back(temp1DVector);
    temp1DVector.clear();
  }

//   return temp2DVector;

}


void PhotonAnalyzer::book3DHistoVector(vector<vector<vector<MonitorElement*> > > &temp3DVector,
				       string histoType, string histoName, string title,
									     int xbin, double xmin,double xmax,
									     int ybin, double ymin, double ymax)
{


  int histo_index = 0;

  vector<MonitorElement*> temp1DVector;
  vector<vector<MonitorElement*> > temp2DVector;
//   vector<vector<vector<MonitorElement*> > > temp3DVector;


  //determining which folder we're in
  bool conversionPlot = false;
  if(dbe_->pwd().find( "Conversions" ) != string::npos) conversionPlot = true;


  if(conversionPlot){
    histo_index_conversions_++;
    histo_index = histo_index_conversions_;
  }
  else{
    histo_index_photons_++;
    histo_index = histo_index_photons_;
  }



  stringstream histo_number_stream;
  histo_number_stream << "h_";
  if(histo_index<10)   histo_number_stream << "0";
  histo_number_stream << histo_index << "_";

  for(int cut = 0; cut != numberOfSteps_; ++cut){     //looping over Et cut values

    for(uint type=0;type!=types_.size();++type){      //looping over isolation type

      for(uint part=0;part!=parts_.size();++part){    //looping over different parts of the ecal

	currentFolder_.str("");
	currentFolder_ << "Egamma/"+fName_+"/" << types_[type] << "Photons/Et above " << (cut+1)*cutStep_ << " GeV";
	if(conversionPlot) currentFolder_ << "/Conversions";

	dbe_->setCurrentFolder(currentFolder_.str());

	string kind;
	if(conversionPlot) kind = " Conversions: ";
	else kind = " Photons: ";

	if(histoType=="1D")           temp1DVector.push_back(dbe_->book1D(      histo_number_stream.str()+histoName+parts_[part],types_[type]+kind+parts_[part]+": "+title,xbin,xmin,xmax));
	else if(histoType=="2D")      temp1DVector.push_back(dbe_->book2D(      histo_number_stream.str()+histoName+parts_[part],types_[type]+kind+parts_[part]+": "+title,xbin,xmin,xmax,ybin,ymin,ymax));
	else if(histoType=="Profile") temp1DVector.push_back(dbe_->bookProfile( histo_number_stream.str()+histoName+parts_[part],types_[type]+kind+parts_[part]+": "+title,xbin,xmin,xmax,ybin,ymin,ymax,""));
	else cout << "bad histoType\n";


      }

      temp2DVector.push_back(temp1DVector);
      temp1DVector.clear();
    }

    temp3DVector.push_back(temp2DVector);
    temp2DVector.clear();
  }

  //  return temp3DVector;
}

bool PhotonAnalyzer::photonSelection ( const reco::PhotonRef & pho) {

  
  bool result=true;
  if ( pho->pt() <  minPhoEtCut_ )          result=false;
  if ( fabs(pho->eta())  > photonMaxEta_ )   result=false;
  if ( pho->isEBEEGap() )       result=false;

  double EtCorrHcalIso = pho->hcalTowerSumEtConeDR03() - 0.005*pho->pt();
  double EtCorrTrkIso  = pho->trkSumPtHollowConeDR03() - 0.002*pho->pt();
  
  if (pho->r9() <=0.9) {
    if (pho->isEB() && (pho->hadTowOverEm()>0.075 || pho->sigmaIetaIeta() > 0.014)) result=false;
    if (pho->isEE() && (pho->hadTowOverEm()>0.075 || pho->sigmaIetaIeta() > 0.034)) result=false;
    ///  remove after moriond    if (EtCorrEcalIso>4.0) result=false;
    if (EtCorrHcalIso>4.0) result=false;
    if (EtCorrTrkIso>4.0) result=false ;
    if ( pho->chargedHadronIso()  > 4 )  result=false;
    
  } else {
    if (pho->isEB() && (pho->hadTowOverEm()>0.082 || pho->sigmaIetaIeta() > 0.014)) result=false;
    if (pho->isEE() && (pho->hadTowOverEm()>0.075 || pho->sigmaIetaIeta() > 0.034)) result=false;
    /// remove after moriond if (EtCorrEcalIso>50.0) result=false;
    if (EtCorrHcalIso>50.0) result=false;
    if (EtCorrTrkIso>50.0) result=false;
    if ( pho->chargedHadronIso()  > 4 )  result=false;
    
  }

  

  return result;  
}
