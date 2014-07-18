#include "DataFormats/PatCandidates/interface/Electron.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "CommonTools/ParticleFlow/interface/PFPileUpAlgo.h"
#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DQM/PhysicsHWW/interface/EgammaFiduciality.h"
#include "DQM/PhysicsHWW/interface/ElectronMaker.h"


using namespace reco;
using namespace edm;
using namespace std;

typedef math::XYZTLorentzVectorF LorentzVector;
typedef math::XYZPoint Point;

ElectronMaker::ElectronMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector) {

  TrackCollection_          = iCollector.consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackInputTag"));
  GSFTrackCollection_       = iCollector.consumes<reco::GsfTrackCollection>(iConfig.getParameter<edm::InputTag>("gsftrksInputTag"));
  GSFElectron_              = iCollector.consumes<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("electronsInputTag"));
  GSFElectronCollection_    = iCollector.consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronsInputTag"));
  PFCandidateCollection_    = iCollector.consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandsInputTag"));
  thePVCollection_          = iCollector.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexInputTag"));
  BeamSpot_                 = iCollector.consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotTag"));
  ConversionCollection_     = iCollector.consumes<reco::ConversionCollection>(iConfig.getParameter<edm::InputTag>("recoConversionInputTag"));
  ClusterToken1_            = iCollector.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("cluster1InputTag"));
  ClusterToken2_            = iCollector.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("cluster2InputTag"));

}


void ElectronMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  hww.Load_els_fiduciality();
  hww.Load_els_type();
  hww.Load_els_ecalEnergy();
  hww.Load_els_trk_p4();
  hww.Load_els_p4();
  hww.Load_els_vertex_p4();
  hww.Load_els_ecalIso();
  hww.Load_els_hcalIso();
  hww.Load_els_tkIso();
  hww.Load_els_ecalIso04();
  hww.Load_els_hcalIso04();
  hww.Load_els_iso03_pf_ch();
  hww.Load_els_iso03_pf_gamma05();
  hww.Load_els_iso03_pf_nhad05();
  hww.Load_els_iso04_pf_ch();
  hww.Load_els_iso04_pf_gamma05();
  hww.Load_els_iso04_pf_nhad05();
  hww.Load_els_iso03_pf2012_ch();
  hww.Load_els_iso03_pf2012_em();
  hww.Load_els_iso03_pf2012_nh();
  hww.Load_els_iso04_pf2012_ch();
  hww.Load_els_iso04_pf2012_em();
  hww.Load_els_iso04_pf2012_nh();
  hww.Load_els_etaSC();
  hww.Load_els_eSC();
  hww.Load_els_eSCRaw();
  hww.Load_els_eSCPresh();
  hww.Load_els_nSeed();
  hww.Load_els_e1x5();
  hww.Load_els_e5x5();
  hww.Load_els_sigmaIEtaIEta();
  hww.Load_els_etaSCwidth();
  hww.Load_els_phiSCwidth();
  hww.Load_els_sigmaIPhiIPhi();
  hww.Load_els_e3x3();
  hww.Load_els_hOverE();
  hww.Load_els_eOverPIn();
  hww.Load_els_eSeedOverPOut();
  hww.Load_els_eSeedOverPIn();
  hww.Load_els_eOverPOut();
  hww.Load_els_fbrem();
  hww.Load_els_dEtaIn();
  hww.Load_els_dEtaOut();
  hww.Load_els_dPhiIn();
  hww.Load_els_dPhiOut();
  hww.Load_els_chi2();
  hww.Load_els_ndof();
  hww.Load_els_gsftrkidx();
  hww.Load_els_charge();
  hww.Load_els_trk_charge();
  hww.Load_els_sccharge();
  hww.Load_els_d0();
  hww.Load_els_d0corr();
  hww.Load_els_z0corr();
  hww.Load_els_trkidx();
  hww.Load_els_trkshFrac();
  hww.Load_els_ip3d();
  hww.Load_els_ip3derr();
  hww.Load_els_exp_innerlayers();
  hww.Load_els_conv_dist();
  hww.Load_els_conv_dcot();
  hww.Load_els_conv_old_dist();
  hww.Load_els_conv_old_dcot();

  bool validToken;

  // access the tracker
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);

  ////////////////
  // Get Tracks //
  ////////////////
 
  Handle<TrackCollection> tracks_h;
  validToken = iEvent.getByToken(TrackCollection_, tracks_h);
  if(!validToken) return;


  ////////////////
  // GSF Tracks //
  ////////////////

  Handle<GsfTrackCollection> gsftracks_h;
  validToken = iEvent.getByToken(GSFTrackCollection_, gsftracks_h);
  if(!validToken) return;


  /////////////
  // B Field //
  /////////////

  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  float evt_bField = magneticField->inTesla(GlobalPoint(0.,0.,0.)).z();
 

  ///////////////
  // Electrons //
  ///////////////

  Handle<View<GsfElectron> > els_h;
  //iEvent.getByToken(GSFElectron_, els_h);
  validToken = iEvent.getByToken(GSFElectron_, els_h);
  if(!validToken) return;
  View<GsfElectron> gsfElColl = *(els_h.product());

  Handle<GsfElectronCollection> els_coll_h;
  //iEvent.getByToken(GSFElectronCollection_, els_coll_h);    
  validToken = iEvent.getByToken(GSFElectronCollection_, els_coll_h);    
  if(!validToken) return;


  //////////////
  // PF Cands //
  //////////////

  validToken = iEvent.getByToken(PFCandidateCollection_, pfCand_h);
  if(!validToken) return;


  ////////////
  // Vertex //
  ////////////

  edm::Handle<reco::VertexCollection> vertexHandle;
  validToken = iEvent.getByToken(thePVCollection_, vertexHandle);
  if(!validToken) return;


  ///////////////////////////
  // TransientTrackBuilder //
  ///////////////////////////

  ESHandle<TransientTrackBuilder> theTTBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTBuilder);


  ////////////////////////////////////////////////
  // Get tools to get cluster shape information //
  ////////////////////////////////////////////////

  EcalClusterLazyTools* clusterTools_;
  clusterTools_ = new EcalClusterLazyTools( iEvent, iSetup, ClusterToken1_, ClusterToken2_ );
  //clusterTools_ = new EcalClusterLazyTools( iEvent, iSetup, InputTag("reducedEcalRecHitsEB"), InputTag("reducedEcalRecHitsEE") );


  //////////////
  // Beamspot //
  //////////////

  Handle<reco::BeamSpot> beamspot_h;
  validToken = iEvent.getByToken(BeamSpot_, beamspot_h);
  if(!validToken) return;
  const reco::BeamSpot &beamSpotreco = *(beamspot_h.product());




  /////////////////////////
  // Loop Over Electrons //
  /////////////////////////

  double mass     = 0.000510998918;
  size_t elsIndex = 0;
  for( View<GsfElectron>::const_iterator el = els_h->begin(); el != els_h->end(); el++, elsIndex++ ) {

      ////////////////
      // References //
      ////////////////

      const Track*                 el_track         = (const Track*)(el->gsfTrack().get());
      const RefToBase<GsfElectron> gsfElRef         = els_h->refAt(elsIndex);    
      const TrackRef               ctfTkRef         = el->closestTrack();
      const GsfTrackRef            gsfTkRef         = el->gsfTrack();
      const VertexCollection*      vertexCollection = vertexHandle.product();

      ////////////
      // Vertex //
      ////////////
      VertexCollection::const_iterator firstGoodVertex = vertexCollection->end();
      int firstGoodVertexIdx = 0;
      for (VertexCollection::const_iterator vtx = vertexCollection->begin(); vtx != vertexCollection->end(); ++vtx, ++firstGoodVertexIdx) {
          if (  !vtx->isFake() && vtx->ndof()>=4. && vtx->position().Rho()<=2.0 && fabs(vtx->position().Z())<=24.0) {
              firstGoodVertex = vtx;
              break;
          }
      }

      //////////////////////
      // Fiduciality Mask //
      //////////////////////

      int fiducialityMask = 0;  // the enum is in interface/EgammaFiduciality.h
      if ( el->isEB()        ) fiducialityMask |= 1 << ISEB;
      if ( el->isEBEEGap()   ) fiducialityMask |= 1 << ISEBEEGAP;
      if ( el->isEE()        ) fiducialityMask |= 1 << ISEE;
      if ( el->isEEGap()     ) fiducialityMask |= 1 << ISEEGAP;
      if ( el->isEBEtaGap()  ) fiducialityMask |= 1 << ISEBETAGAP;
      if ( el->isEBPhiGap()  ) fiducialityMask |= 1 << ISEBPHIGAP;
      if ( el->isEEDeeGap()  ) fiducialityMask |= 1 << ISEEDEEGAP;
      if ( el->isEERingGap() ) fiducialityMask |= 1 << ISEERINGGAP;
      if ( el->isGap()       ) fiducialityMask |= 1 << ISGAP;


      ///////////////////////////
      // Corrections & Seeding //
      ///////////////////////////

      int electronTypeMask = 0;
      if ( el->isEcalEnergyCorrected()        ) electronTypeMask |= 1 << ISECALENERGYCORRECTED;
      if ( el->trackerDrivenSeed()            ) electronTypeMask |= 1 << ISTRACKERDRIVEN;
      if ( el->ecalDrivenSeed()               ) electronTypeMask |= 1 << ISECALDRIVEN;
      if ( el->passingCutBasedPreselection()  ) electronTypeMask |= 1 << ISCUTPRESELECTED;
      if ( el->passingMvaPreselection()       ) electronTypeMask |= 1 << ISMVAPRESELECTED;


      /////////////////////
      // Lorentz Vectors //
      /////////////////////

      LorentzVector    p4In;
      LorentzVector    p4Out;
      LorentzVector    trk_p4( el_track->px(), el_track->py(), el_track->pz(), el_track->p() );
      math::XYZVectorF p3In  = el->trackMomentumAtVtx();
      math::XYZVectorF p3Out = el->trackMomentumOut();
      p4In.SetXYZT (   p3In.x() , p3In.y() , p3In.z() , sqrt( mass*mass + p3In.R() *p3In.R()  ) );
      p4Out.SetXYZT(   p3Out.x(), p3Out.y(), p3Out.z(), sqrt( mass*mass + p3Out.R()*p3Out.R() ) );


      //////////////
      // Electron //
      //////////////

      hww.els_fiduciality()        .push_back( fiducialityMask                                 );
      hww.els_type()               .push_back( electronTypeMask                                );
      hww.els_ecalEnergy()         .push_back( el->correctedEcalEnergy()                       ); 
      hww.els_p4()                 .push_back( LorentzVector( el->p4() )                       );
      hww.els_trk_p4()             .push_back( trk_p4                                          );
      hww.els_vertex_p4()          .push_back( LorentzVector(el->vx(), el->vy(), el->vz(), 0.) );


      ///////////////
      // Isolation //
      ///////////////

      hww.els_ecalIso()               .push_back( el->dr03EcalRecHitSumEt()                  );
      hww.els_hcalIso()               .push_back( el->dr03HcalTowerSumEt()                   );
      hww.els_tkIso()                 .push_back( el->dr03TkSumPt()                          );
      hww.els_ecalIso04()             .push_back( el->dr04EcalRecHitSumEt()                  );
      hww.els_hcalIso04()             .push_back( el->dr04HcalTowerSumEt()                   );


      //////////////////
      // PF Isolation //
      //////////////////

      if ( firstGoodVertex!=vertexCollection->end() ) {

          hww.els_iso03_pf_ch()      .push_back( electronIsoValuePF( *el, *firstGoodVertex, 0.3, 99999., 0.1, 0.07, 0.025, 0.025, 0  ) );
          hww.els_iso03_pf_gamma05() .push_back( electronIsoValuePF( *el, *firstGoodVertex, 0.3, 0.5   , 0.1, 0.07, 0.025, 0.025, 22 ) );
          hww.els_iso03_pf_nhad05()  .push_back( electronIsoValuePF( *el, *firstGoodVertex, 0.3, 0.5   , 0.1, 0.07, 0.025, 0.025, 130) );

          hww.els_iso04_pf_ch()      .push_back( electronIsoValuePF( *el, *firstGoodVertex, 0.4, 99999., 0.1, 0.07, 0.025, 0.025, 0  ) );
          hww.els_iso04_pf_gamma05() .push_back( electronIsoValuePF( *el, *firstGoodVertex, 0.4, 0.5   , 0.1, 0.07, 0.025, 0.025, 22 ) );
          hww.els_iso04_pf_nhad05()  .push_back( electronIsoValuePF( *el, *firstGoodVertex, 0.4,  0.5  , 0.1, 0.07, 0.025, 0.025, 130) );


          // pf iso 2012
          float pfiso_ch = 0.0;
          float pfiso_em = 0.0;
          float pfiso_nh = 0.0;

          PFIsolation2012(*el, vertexCollection, firstGoodVertexIdx, 0.3, pfiso_ch, pfiso_em, pfiso_nh);
          hww.els_iso03_pf2012_ch() .push_back( pfiso_ch );
          hww.els_iso03_pf2012_em() .push_back( pfiso_em );
          hww.els_iso03_pf2012_nh() .push_back( pfiso_nh );

          PFIsolation2012(*el, vertexCollection, firstGoodVertexIdx, 0.4, pfiso_ch, pfiso_em, pfiso_nh);
          hww.els_iso04_pf2012_ch() .push_back( pfiso_ch );
          hww.els_iso04_pf2012_em() .push_back( pfiso_em );
          hww.els_iso04_pf2012_nh() .push_back( pfiso_nh );	    

      } else {

          hww.els_iso03_pf_ch()      .push_back( -9999. );
          hww.els_iso03_pf_gamma05() .push_back( -9999. );
          hww.els_iso03_pf_nhad05()  .push_back( -9999. );

          hww.els_iso04_pf_ch()      .push_back( -9999. );
          hww.els_iso04_pf_gamma05() .push_back( -9999. );
          hww.els_iso04_pf_nhad05()  .push_back( -9999. );

          hww.els_iso03_pf2012_ch() .push_back( -9999. );
          hww.els_iso03_pf2012_em() .push_back( -9999. );
          hww.els_iso03_pf2012_nh() .push_back( -9999. );
          hww.els_iso04_pf2012_ch() .push_back( -9999. );
          hww.els_iso04_pf2012_em() .push_back( -9999. );
          hww.els_iso04_pf2012_nh() .push_back( -9999. );
      }

      //////////////////
      // Supercluster //
      //////////////////

      hww.els_etaSC()         .push_back( el->superCluster()->eta()             );
      hww.els_eSC()           .push_back( el->superCluster()->energy()          );
      hww.els_eSCRaw()        .push_back( el->superCluster()->rawEnergy()       );
      hww.els_eSCPresh()      .push_back( el->superCluster()->preshowerEnergy() );
      hww.els_nSeed()         .push_back( el->basicClustersSize() - 1           );
      hww.els_e1x5()          .push_back( el->e1x5()                            );
      hww.els_e5x5()          .push_back( el->e5x5()                            );
      hww.els_sigmaIEtaIEta() .push_back( el->sigmaIetaIeta()                   );
      hww.els_etaSCwidth()    .push_back( el->superCluster()->etaWidth()        );
      hww.els_phiSCwidth()    .push_back( el->superCluster()->phiWidth()        );


      ///////////////////////////////////////////////////////
      // Get cluster info that is not stored in the object //
      ///////////////////////////////////////////////////////

      if( el->superCluster()->seed().isAvailable() ) { 

          const BasicCluster&  clRef              = *(el->superCluster()->seed());
          const vector<float>& lcovs              = clusterTools_->localCovariances(clRef);                    // get the local covariances computed in a 5x5 around the seed
          const vector<float>  localCovariancesSC = clusterTools_->scLocalCovariances(*(el->superCluster()));  // get the local covariances computed using all crystals in the SC

          hww.els_sigmaIPhiIPhi()   .push_back( isfinite(lcovs[2])  ? lcovs[2] > 0  ? sqrt(lcovs[2]) : -1 * sqrt(-1 * lcovs[2])  : -9999. );
          hww.els_e3x3()            .push_back( clusterTools_->e3x3(clRef) );
      } 
      else {

          hww.els_sigmaIPhiIPhi()   .push_back(-9999.);
          hww.els_e3x3()            .push_back(-9999.);

      } 

 
      ////////
      // ID //
      ////////

      hww.els_hOverE()                        .push_back( el->hcalOverEcal()                   );
      hww.els_eOverPIn()                      .push_back( el->eSuperClusterOverP()             );
      hww.els_eSeedOverPOut()                 .push_back( el->eSeedClusterOverPout()           );
      hww.els_eSeedOverPIn()                  .push_back( el->eSeedClusterOverP()              );
      hww.els_eOverPOut()                     .push_back( el->eEleClusterOverPout()            );
      hww.els_fbrem()                         .push_back( el->fbrem()                          );

      hww.els_dEtaIn()                        .push_back( el->deltaEtaSuperClusterTrackAtVtx() );
      hww.els_dEtaOut()                       .push_back( el->deltaEtaSeedClusterTrackAtCalo() );
      hww.els_dPhiIn()                        .push_back( el->deltaPhiSuperClusterTrackAtVtx() );
      hww.els_dPhiOut()                       .push_back( el->deltaPhiSeedClusterTrackAtCalo() );

      ////////////
      // Tracks //
      ////////////

      hww.els_chi2()                  .push_back( el_track->chi2()                          );
      hww.els_ndof()                  .push_back( el_track->ndof()                          );
      hww.els_gsftrkidx()             .push_back( static_cast<int>((el->gsfTrack()).key())  );
      hww.els_charge()                .push_back( el->charge()                              );
      hww.els_trk_charge()            .push_back( el_track->charge()                        );
      hww.els_sccharge()              .push_back( el->scPixCharge()                         );
      hww.els_d0()                    .push_back( el_track->d0()                            );
      hww.els_d0corr()                .push_back( -1*(el_track->dxy(beamSpotreco))              );
      hww.els_z0corr()                .push_back( el_track->dz(beamSpotreco.position(el_track->vz()))                    );
 

      /////////
      // CTF //
      /////////

      if( ctfTkRef.isNonnull() ) {
          hww.els_trkidx()    . push_back( static_cast<int>  ( ctfTkRef.key()        )                                  );
          hww.els_trkshFrac() . push_back( static_cast<float>( el->ctfGsfOverlap() )                                    );
      } 
      else {
          hww.els_trkidx()    . push_back(-9999.);
          hww.els_trkshFrac() . push_back(-9999.);
      }

      
      ////////////////////
      // Regular Vertex //
      ////////////////////        
      TransientTrack tt = theTTBuilder->build(el->gsfTrack());
  
      if ( firstGoodVertex!=vertexCollection->end() ) {
          Measurement1D ip3D_regular = IPTools::absoluteImpactParameter3D(tt, *firstGoodVertex).second;

          hww.els_ip3d()      . push_back( ip3D_regular.value() );
          hww.els_ip3derr()   . push_back( ip3D_regular.error() );
      } else {

          hww.els_ip3d()      . push_back( -999. );
          hww.els_ip3derr()   . push_back( -999. );
      }


      /////////////////
      // Hit Pattern //
      /////////////////

      const HitPattern& p_inner = el_track->trackerExpectedHitsInner(); 

      hww.els_exp_innerlayers().push_back(p_inner.numberOfHits());


      /////////////////
      // Conversions //
      /////////////////

      ConversionFinder convFinder; //vector of conversion infos - all the candidate conversion tracks
      vector<ConversionInfo> v_convInfos = convFinder.getConversionInfos(*(el->core()), tracks_h, gsftracks_h, evt_bField);
  
      vector<int>           v_tkidx;
      vector<int>           v_gsftkidx;
      vector<int>           v_delmisshits;
      vector<int>           v_flag;
      vector<float>         v_dist;
      vector<float>         v_dcot;
      vector<float>         v_rad;
      vector<LorentzVector> v_pos_p4;

      for(unsigned int i_conv = 0; i_conv < v_convInfos.size(); i_conv++) {
    
          math::XYZPoint convPoint  = v_convInfos.at(i_conv).pointOfConversion();
          float          convPointX = isfinite(convPoint.x()) ? convPoint.x() : -9999.;
          float          convPointY = isfinite(convPoint.y()) ? convPoint.y() : -9999.;
          float          convPointZ = isfinite(convPoint.z()) ? convPoint.z() : -9999.;

          v_dist        .push_back( isfinite(v_convInfos.at(i_conv).dist()) ? v_convInfos.at(i_conv).dist() : -9999.  );
          v_dcot        .push_back( v_convInfos.at(i_conv).dcot()                                                     );
          v_rad         .push_back( v_convInfos.at(i_conv).radiusOfConversion()                                       );
          v_delmisshits .push_back( v_convInfos.at(i_conv).deltaMissingHits()                                         );
          v_flag        .push_back( v_convInfos.at(i_conv).flag()                                                     );
          v_pos_p4      .push_back( LorentzVector(convPointX, convPointY, convPointZ, 0)                              );

          if( v_convInfos.at(i_conv).conversionPartnerCtfTk().isNonnull() ) {
              v_tkidx.push_back(v_convInfos.at(i_conv).conversionPartnerCtfTk().key());
          }
          else {
              v_tkidx.push_back(-9999);
          }

          //
          if( v_convInfos.at(i_conv).conversionPartnerGsfTk().isNonnull() ) {
              v_gsftkidx.push_back(v_convInfos.at(i_conv).conversionPartnerGsfTk().key());
          }
          else { 
              v_gsftkidx.push_back(-9999);
          }

      } // end for loop


      ConversionInfo convInfo   = convFinder.getConversionInfo( *el, tracks_h, gsftracks_h, evt_bField );

      hww.els_conv_dist().push_back( isfinite(convInfo.dist()) ? convInfo.dist() : -9999. );
      hww.els_conv_dcot().push_back( convInfo.dcot()                                      );


      //////////////////////////////////////////////
      // Flag For Vertex Fit Conversion Rejection //
      //////////////////////////////////////////////

      Handle<ConversionCollection> convs_h;
      iEvent.getByToken(ConversionCollection_, convs_h);


      //////////////////////////////
      // Old Conversion Rejection //
      //////////////////////////////

      hww.els_conv_old_dist()        . push_back( isfinite(el->convDist())   ? el->convDist()   : -9999. );
      hww.els_conv_old_dcot()        . push_back( isfinite(el->convDcot())   ? el->convDcot()   : -9999. );


      //////////////////////
      // 2012 Electron ID //
      //////////////////////

      GsfElectronRef ele(els_coll_h, elsIndex);

  } // end Loop on Electrons

}


double ElectronMaker::electronIsoValuePF(const GsfElectron& el, const Vertex& vtx, float coner, float minptn, float dzcut,
                                         float footprintdr, float gammastripveto, float elestripveto, int filterId){

    float pfciso = 0.;
    float pfniso = 0.;
    float pffootprint = 0.;
    float pfjurveto = 0.;
    float pfjurvetoq = 0.;

    TrackRef siTrack     = el.closestTrack();
    GsfTrackRef gsfTrack = el.gsfTrack();

    if (gsfTrack.isNull() && siTrack.isNull()) return -9999.;

    float eldz = gsfTrack.isNonnull() ? gsfTrack->dz(vtx.position()) : siTrack->dz(vtx.position());
    float eleta = el.eta();

    for (PFCandidateCollection::const_iterator pf=pfCand_h->begin(); pf<pfCand_h->end(); ++pf){

        float pfeta = pf->eta();    
        float dR = deltaR(pfeta, pf->phi(), eleta, el.phi());
        if (dR>coner) continue;

        float deta = fabs(pfeta - eleta);
        int pfid = abs(pf->pdgId());
        float pfpt = pf->pt();

        if (filterId!=0 && filterId!=pfid) continue;

        if (pf->charge()==0) {
            //neutrals
            if (pfpt>minptn) {
                pfniso+=pfpt;
                if (dR<footprintdr && pfid==130) pffootprint+=pfpt;
                if (deta<gammastripveto && pfid==22)  pfjurveto+=pfpt;
            }
        } else {
            //charged  
            //avoid double counting of electron itself
            //if either the gsf or the ctf track are shared with the candidate, skip it
            const TrackRef pfTrack  = pf->trackRef();
            if (siTrack.isNonnull()  && pfTrack.isNonnull() && siTrack.key()==pfTrack.key()) continue;
            //below pfid==1 is commented out: in some cases the pfCand has a gsf even if it is not an electron... this is to improve the sync with MIT
            if (/*pfid==11 &&*/ pf->gsfTrackRef().isNonnull()) {
                if (gsfTrack.isNonnull() && gsfTrack.key()==pf->gsfTrackRef().key()) continue;
            } 
            //check electrons with gsf track
            if (pfid==11 && pf->gsfTrackRef().isNonnull()) {
                if(fabs(pf->gsfTrackRef()->dz(vtx.position()) - eldz )<dzcut) {//dz cut
                    pfciso+=pfpt;
                    if (deta<elestripveto && pfid==11) pfjurvetoq+=pfpt;
                }
                continue;//and avoid double counting
            }
            //then check anything that has a ctf track
            if (pfTrack.isNonnull()) {//charged (with a ctf track)
                if(fabs( pfTrack->dz(vtx.position()) - eldz )<dzcut) {//dz cut
                    pfciso+=pfpt;
                    if (deta<elestripveto && pfid==11) pfjurvetoq+=pfpt;
                }
            }
        } 
    }
    return pfciso+pfniso-pffootprint-pfjurveto-pfjurvetoq;
}


void ElectronMaker::PFIsolation2012(const reco::GsfElectron& el, const reco::VertexCollection* vertexCollection,
        const int vertexIndex, const float &R, float &pfiso_ch, float &pfiso_em, float &pfiso_nh)
{

    // isolation sums
    pfiso_ch = 0.0;
    pfiso_em = 0.0;
    pfiso_nh = 0.0;

    // loop on pfcandidates
    reco::PFCandidateCollection::const_iterator pf = pfCand_h->begin();
    for (pf = pfCand_h->begin(); pf != pfCand_h->end(); ++pf) {

        // skip electrons and muons
        if (pf->particleId() == reco::PFCandidate::e)     continue;
        if (pf->particleId() == reco::PFCandidate::mu)    continue;

        // deltaR between electron and cadidate
        const float dR = deltaR(pf->eta(), pf->phi(), el.eta(), el.phi());
        if (dR > R)                             continue;

        PFPileUpAlgo *pfPileUpAlgo_ = new PFPileUpAlgo();

        if (pf->particleId() == reco::PFCandidate::h) {
            int pfVertexIndex = pfPileUpAlgo_->chargedHadronVertex(*vertexCollection, *pf);
            if (pfVertexIndex != vertexIndex) continue;
        }

        // endcap region
        if (!el.isEB()) {
            if (pf->particleId() == reco::PFCandidate::h      && dR <= 0.015)   continue;
            if (pf->particleId() == reco::PFCandidate::gamma  && dR <= 0.08)    continue;
        }

        // add to isolation sum
        if (pf->particleId() == reco::PFCandidate::h)       pfiso_ch += pf->pt();
        if (pf->particleId() == reco::PFCandidate::gamma)   pfiso_em += pf->pt();
        if (pf->particleId() == reco::PFCandidate::h0)      pfiso_nh += pf->pt();

    }

}

