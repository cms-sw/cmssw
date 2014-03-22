/* \class ZMuMuAnalyzer_cynematics
 *
 * Z->mu+m- standard analysis for cross section
 * measurements. Take as input the output of the
 * standard EWK skim: zToMuMu
 *
 * Produces mass spectra and other histograms for
 * the samples in input:
 *
 *  + Z -> mu+mu-, both muons are "global" muons
 *  + Z -> mu+mu-, one muons is "global" muons, one unmatched tracks
 *  + Z -> mu+mu-, one muons is "global" muons, one unmatched stand-alone muon
 *
 *
 * \author Michele de Gruttola,
 * \modified by Davide Piccolo, INFN Naples to include gerarchyc selection of Z and histos as a finction of eta pt phi
 *
 *
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include <iostream>
#include <iterator>
#include <sstream>
using namespace edm;
using namespace std;
using namespace reco;

typedef edm::AssociationVector<reco::CandidateRefProd, std::vector<double> > IsolationCollection;

class ZMuMuAnalyzer_cynematics : public edm::EDAnalyzer {
public:
  ZMuMuAnalyzer_cynematics(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  bool isContained(const Candidate &, const Candidate &);
  virtual void endJob() override;

  OverlapChecker overlap_;
  EDGetTokenT<CandidateCollection> zMuMuToken_;
  EDGetTokenT<CandidateCollection> zMuTrackToken_;
  EDGetTokenT<CandidateCollection> zMuStandAloneToken_;
  EDGetTokenT<IsolationCollection> muIsoToken_;
  EDGetTokenT<IsolationCollection> trackIsoToken_;
  EDGetTokenT<IsolationCollection> standAloneIsoToken_;
  EDGetTokenT<CandMatchMap> zMuMuMapToken_;
  EDGetTokenT<CandMatchMap> zMuTrackMapToken_;
  EDGetTokenT<CandMatchMap> zMuStandAloneMapToken_;
  double isocut_, etacut_, ptcut_,ptSTAcut_,  minZmass_, maxZmass_;
  TH1D * h_zMuMu_numberOfCand, * h_zMuMu_numberOfCand_passed, * h_zMuMu_numberOfCand_ptpassed,* h_zMuMu_numberOfCand_etapassed,
    * h_zMuMu_numberOfCand_masspassed, * h_zMuMu_numberOfCand_isopassed, * h_zMuMu_numberOfCand_ptetapassed,
    * h_zMuMu_numberOfCand_ptetamasspassed, * h_zMuMu_mass_, * h_zMuSingleTrack_mass_, * h_zMuSingleStandAlone_mass_,
    * h_zMuSingleStandAloneOverlap_mass_, * h_zMuMuMatched_mass_, * h_zMuSingleTrackMatched_mass_,
    * h_zMuSingleStandAloneMatched_mass_,
    * h_zMuSingleStandAloneOverlapMatched_mass_;

  TH1D * h_zMuSta_numberOfCand,* h_zMuSta_numberOfCand_passed,* h_zMuSta_MCmatched_numberOfCand_passed,
    * h_zMuSta_numberOfCand_notcontained,
    * h_zMuTrack_numberOfCand, * h_zMuTrack_numberOfCand_notcontained, * h_zMuTrack_numberOfCand_passed,
    * h_zMuTrack_MCmatched_numberOfCand_passed;
  TH2D * h_OneSta_mass;

  double etamin, etamax, phimin, phimax, ptmin, ptmax;
  int numberOfIntervals;        // number of intervals in which to divide cynematic variables
  double binEta,binPhi, binPt;
  vector<TH1D *>  hmumu_eta, hmusta_eta, hmutrack_eta;
  vector<TH1D *>  hmumu_phi, hmusta_phi, hmutrack_phi;
  vector<TH1D *>  hmumu_pt, hmusta_pt, hmutrack_pt;

};

ZMuMuAnalyzer_cynematics::ZMuMuAnalyzer_cynematics(const edm::ParameterSet& pset) :
  zMuMuToken_( consumes< CandidateCollection >( pset.getParameter<InputTag>( "zMuMu" ) ) ),
  zMuTrackToken_( consumes< CandidateCollection >( pset.getParameter<InputTag>( "zMuTrack" ) ) ),
  zMuStandAloneToken_( consumes< CandidateCollection >( pset.getParameter<InputTag>( "zMuStandAlone" ) ) ),
  muIsoToken_( consumes< IsolationCollection >( pset.getParameter<InputTag>( "muIso" ) ) ),
  trackIsoToken_( consumes< IsolationCollection >( pset.getParameter<InputTag>( "trackIso" ) ) ),
  standAloneIsoToken_( consumes< IsolationCollection >( pset.getParameter<InputTag>( "standAloneIso" ) ) ),
  zMuMuMapToken_( mayConsume< CandMatchMap >( pset.getParameter<InputTag>( "zMuMuMap" ) ) ),
  zMuTrackMapToken_( mayConsume< CandMatchMap >( pset.getParameter<InputTag>( "zMuTrackMap" ) ) ),
  zMuStandAloneMapToken_( mayConsume< CandMatchMap >( pset.getParameter<InputTag>( "zMuStandAloneMap" ) ) ),
  isocut_( pset.getParameter<double>( "isocut" ) ),
  etacut_( pset.getParameter<double>( "etacut" ) ),
  ptcut_( pset.getParameter<double>( "ptcut" ) ),
  ptSTAcut_( pset.getParameter<double>( "ptSTAcut" ) ),
  minZmass_( pset.getParameter<double>( "minZmass" )),
  maxZmass_( pset.getParameter<double>( "maxZmass" )) {

  Service<TFileService> fs;
  h_zMuMu_numberOfCand = fs->make<TH1D>("ZMuMunumberOfCand","number of ZMuMu cand",10, -.5, 9.5);
  h_zMuMu_numberOfCand_passed = fs->make<TH1D>("ZMuMunumberOfCandpassed","number of ZMuMu cand selected",10, -.5, 9.5);
  h_zMuMu_numberOfCand_ptpassed = fs->make<TH1D>("ZMuMunumberOfCandptpassed","number of ZMuMu cand after pt cut selected",10, -.5, 9.5);
  h_zMuMu_numberOfCand_etapassed = fs->make<TH1D>("ZMuMunumberOfCandetapassed","number of ZMuMu cand after eta cut selected",10, -.5, 9.5);
  h_zMuMu_numberOfCand_masspassed = fs->make<TH1D>("ZMuMunumberOfCandmasspassed","number of ZMuMu cand after mass cut selected",10, -.5, 9.5);
  h_zMuMu_numberOfCand_isopassed = fs->make<TH1D>("ZMuMunumberOfCandisopassed","number of ZMuMu cand after iso cut selected",10, -.5, 9.5);
  h_zMuMu_numberOfCand_ptetapassed = fs->make<TH1D>("ZMuMunumberOfCandptetapassed","number of ZMuMu cand after pt & eta cut selected",10, -.5, 9.5);
  h_zMuMu_numberOfCand_ptetamasspassed = fs->make<TH1D>("ZMuMunumberOfCandptetamaspassed","number of ZMuMu cand after pt & eta & mass cut selected",10, -.5, 9.5);


  h_zMuMu_mass_ = fs->make<TH1D>( "ZMuMumass", "ZMuMu mass(GeV)", 200,  0., 200. );
  h_zMuSingleTrack_mass_ = fs->make<TH1D>( "ZMuSingleTrackmass", "ZMuSingleTrack mass(GeV)", 100,  0., 200. );
  h_zMuSingleStandAlone_mass_ = fs->make<TH1D>( "ZMuSingleStandAlonemass", "ZMuSingleStandAlone mass(GeV)", 50,  0., 200. );
  h_zMuSingleStandAloneOverlap_mass_ = fs->make<TH1D>( "ZMuSingleStandAloneOverlapmass", "ZMuSingleStandAloneOverlap  mass(GeV)", 50,  0., 200. );


  h_zMuMuMatched_mass_ = fs->make<TH1D>( "ZMuMuMatchedmass", "ZMuMu Matched  mass(GeV)", 200,  0., 200. );
  h_zMuSingleTrackMatched_mass_ = fs->make<TH1D>( "ZMuSingleTrackmassMatched", "ZMuSingleTrackMatched mass(GeV)", 100,  0., 200. );
  h_zMuSingleStandAloneMatched_mass_ = fs->make<TH1D>( "ZMuSingleStandAlonemassMatched", "ZMuSingleStandAloneMatched mass(GeV)", 50,  0., 200. );
  h_zMuSingleStandAloneOverlapMatched_mass_ = fs->make<TH1D>( "ZMuSingleStandAloneOverlapmassMatched", "ZMuSingleStandAloneMatched Overlap  mass(GeV)", 50,  0., 200. );

  h_zMuSta_numberOfCand = fs->make<TH1D>("ZMuStanumberOfCand","number of ZMuSta cand (if ZMuMu not selected)",10, -.5, 9.5);
  h_OneSta_mass = fs->make<TH2D>("ZOneMuStaMass","inv. mass of ZMuSta1 vs ZMuSta2 when one ZMuSta has been found (if ZMuMu not selected)",100, 0., 400, 100, 0., 400.);
  h_zMuSta_numberOfCand_notcontained = fs->make<TH1D>("ZMuStanumberOfCandnotcontained","number of independent ZMuSta cand (if ZMuMu not selected)",10, -.5, 9.5);
  h_zMuSta_numberOfCand_passed = fs->make<TH1D>("ZMuStanumberOfCandpassed","number of ZMuSta cand selected (if ZMuMu not selected)",10, -.5, 9.5);
  h_zMuSta_MCmatched_numberOfCand_passed = fs->make<TH1D>("ZMuStaMCmatchedNumberOfCandpassed","number of ZMuSta MC matched cand selected (if ZMuMu not selected)",10, -.5, 9.5);
  h_zMuTrack_numberOfCand = fs->make<TH1D>("ZMuTranumberOfCand","number of ZMuTrack cand (if ZMuMu and ZMuSTa not selected)",10, -.5, 9.5);
  h_zMuTrack_numberOfCand_notcontained = fs->make<TH1D>("ZMuTranumberOfCandnotcontaind","number of indeendent ZMuTrack cand (if ZMuMu and ZMuSTa not selected)",10, -.5, 9.5);
  h_zMuTrack_numberOfCand_passed = fs->make<TH1D>("ZMuTranumberOfCandpassed","number of ZMuTrack cand selected (if ZMuMu and ZMuSta not selected)",10, -.5, 9.5);
  h_zMuTrack_MCmatched_numberOfCand_passed = fs->make<TH1D>("ZMuTraMCmacthedNumberOfCandpassed","number of ZMuTrack MC matched cand selected (if ZMuMu and ZMuSta not selected)",10, -.5, 9.5);


  // creating histograms for each Pt, eta, phi interval

  etamin = -etacut_;
  etamax = etacut_;
  phimin = -3.1415;
  phimax = 3.1415;
  ptmin = ptcut_;
  ptmax = 100;
  numberOfIntervals = 8;        // number of intervals in which to divide cynematic variables
  binEta = (etamax - etamin)/numberOfIntervals;
  binPhi = (phimax - phimin)/numberOfIntervals;
  binPt = (ptmax - ptmin)/numberOfIntervals;
  TFileDirectory etaDirectory = fs->mkdir("etaIntervals");   // in this directory will be saved all the histos of different eta intervals
  TFileDirectory phiDirectory = fs->mkdir("phiIntervals");   // in this directory will be saved all the histos of different phi intervals
  TFileDirectory ptDirectory = fs->mkdir("ptIntervals");   // in this directory will be saved all the histos of different pt intervals

  // eta histograms creation

  for (int i=0;i<numberOfIntervals;i++) {
    double range0 = etamin + i*binEta;
    double range1= range0 + binEta;
    char a[30], b[50];
    sprintf(a,"zmumu_etaRange%d",i);
    sprintf(b,"zmumu mass eta Range %f to %f",range0,range1);
    hmumu_eta.push_back(etaDirectory.make<TH1D>(a,b,200,0.,200.));
    char asta[30], bsta[50];
    sprintf(asta,"zmusta_etaRange%d",i);
    sprintf(bsta,"zmusta mass eta Range %f to %f",range0,range1);
    hmusta_eta.push_back(etaDirectory.make<TH1D>(asta,bsta,50,0.,200.));
    char atk[30], btk[50];
    sprintf(atk,"zmutrack_etaRange%d",i);
    sprintf(btk,"zmutrack mass eta Range %f to %f",range0,range1);
    hmutrack_eta.push_back(etaDirectory.make<TH1D>(atk,btk,100,0.,200.));
  }

  // phi histograms creation

  for (int i=0;i<numberOfIntervals;i++) {
    double range0 = phimin + i*binPhi;
    double range1= range0 + binPhi;
    char a[30], b[50];
    sprintf(a,"zmumu_phiRange%d",i);
    sprintf(b,"zmumu mass phi Range %f to %f",range0,range1);
    hmumu_phi.push_back(phiDirectory.make<TH1D>(a,b,200,0.,200.));
    char asta[30], bsta[50];
    sprintf(asta,"zmusta_phiRange%d",i);
    sprintf(bsta,"zmusta mass phi Range %f to %f",range0,range1);
    hmusta_phi.push_back(phiDirectory.make<TH1D>(asta,bsta,50,0.,200.));
    char atk[30], btk[50];
    sprintf(atk,"zmutrack_phiRange%d",i);
    sprintf(btk,"zmutrack mass phi Range %f to %f",range0,range1);
    hmutrack_phi.push_back(phiDirectory.make<TH1D>(atk,btk,100,0.,200.));
  }

  // pt histograms creation

  for (int i=0;i<numberOfIntervals;i++) {
    double range0 = ptmin + i*binPt;
    double range1= range0 + binPt;
    char a[30], b[50];
    sprintf(a,"zmumu_ptRange%d",i);
    sprintf(b,"zmumu mass pt Range %f to %f",range0,range1);
    hmumu_pt.push_back(ptDirectory.make<TH1D>(a,b,200,0.,200.));
    char asta[30], bsta[50];
    sprintf(asta,"zmusta_ptRange%d",i);
    sprintf(bsta,"zmusta mass pt Range %f to %f",range0,range1);
    hmusta_pt.push_back(ptDirectory.make<TH1D>(asta,bsta,50,0.,200.));
    char atk[30], btk[50];
    sprintf(atk,"zmutrack_ptRange%d",i);
    sprintf(btk,"zmutrack mass pt Range %f to %f",range0,range1);
    hmutrack_pt.push_back(ptDirectory.make<TH1D>(atk,btk,100,0.,200.));
  }
 }

void ZMuMuAnalyzer_cynematics::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  Handle<CandidateCollection> zMuMu;
  event.getByToken(zMuMuToken_, zMuMu);
  Handle<CandidateCollection> zMuTrack;
  event.getByToken( zMuTrackToken_, zMuTrack );
  Handle<CandidateCollection> zMuStandAlone;
  event.getByToken( zMuStandAloneToken_, zMuStandAlone );

  unsigned int nZMuMu = zMuMu->size();
  unsigned int nZTrackMu = zMuTrack->size();
  unsigned int nZStandAloneMu = zMuStandAlone->size();
  //  static const double zMass = 91.1876; // PDG Z mass

  cout << "++++++++++++++++++++++++++" << endl;
  cout << "nZMuMu = " << nZMuMu << endl;
  cout << "nZTrackMu = " << nZTrackMu << endl;
  cout << "nZStandAloneMu = " << nZStandAloneMu << endl;
  cout << "++++++++++++++++++++++++++" << endl;

  // ZMuMu counters

  int ZMuMu_passed = 0;
  int ZMuMu_ptcut_counter = 0;
  int ZMuMu_etacut_counter = 0;
  int ZMuMu_masscut_counter = 0;
  int ZMuMu_isocut_counter = 0;
  int ZMuMu_ptetacut_counter = 0;
  int ZMuMu_ptetamasscut_counter = 0;
  int ZMuMu_allcut_counter = 0;

  // ZMuTrack counters

  int ZMuTrack_passed = 0;
  int ZMuTrack_notcontained = 0;
  int ZMuTrack_MCmatched_passed = 0;

  // ZMuStandalone counters
  int ZMuStandalone_notcontained = 0;
  int ZMuStandalone_passed = 0;
  int ZMuStandalone_MCmatched_passed = 0;

  Handle<CandMatchMap> zMuMuMap;
  if( nZMuMu > 0 ) {
    event.getByToken(zMuMuMapToken_, zMuMuMap);
  }

  Handle<CandMatchMap> zMuTrackMap;
  if( nZTrackMu > 0 ) {
    event.getByToken( zMuTrackMapToken_, zMuTrackMap );
  }

  Handle<CandMatchMap> zMuStandAloneMap;
  if( nZStandAloneMu > 0 ) {
    event.getByToken( zMuStandAloneMapToken_, zMuStandAloneMap );
  }

  Handle<IsolationCollection> muIso;
  event.getByToken(muIsoToken_, muIso);
  ProductID muIsoId = muIso->keyProduct().id();
  Handle<IsolationCollection> trackIso;
  event.getByToken(trackIsoToken_, trackIso);
  ProductID trackIsoId = trackIso->keyProduct().id();

  Handle<IsolationCollection> standAloneIso;
  event.getByToken(standAloneIsoToken_, standAloneIso);
  ProductID standAloneIsoId = standAloneIso->keyProduct().id();

  if (nZMuMu > 0) {
    // double mass = 1000000.;
    for( unsigned int i = 0; i < nZMuMu; i++ ) {
      bool ptcutAccept = false;
      bool etacutAccept = false;
      bool masscutAccept = false;
      bool isocutAccept = false;
      const Candidate & zmmCand = (*zMuMu)[ i ];
      CandidateRef CandRef(zMuMu,i);
      CandidateRef lep1 = zmmCand.daughter( 0 )->masterClone().castTo<CandidateRef>();
      CandidateRef lep2 = zmmCand.daughter( 1 )->masterClone().castTo<CandidateRef>();

      const  double iso1 = muIso->value( lep1.key() );
      const  double iso2 = muIso->value( lep2.key() );

      double m = zmmCand.mass();
      // check single cuts

      if (lep1->pt()>ptcut_ && lep2->pt()>ptcut_) ptcutAccept = true;
      if (fabs(lep1->eta())<etacut_ && fabs(lep2->eta())<etacut_) etacutAccept = true;
      if (m>minZmass_ && m<maxZmass_) masscutAccept = true;
      if (iso1 < isocut_ && iso2 <isocut_) isocutAccept = true;


      if (ptcutAccept) ZMuMu_ptcut_counter++;
      if (etacutAccept) ZMuMu_etacut_counter++;
      if (masscutAccept) ZMuMu_masscut_counter++;
      if (isocutAccept) ZMuMu_isocut_counter++;

      // check sequencial cuts

      if (ptcutAccept && etacutAccept) {
	ZMuMu_ptetacut_counter++;
	if (masscutAccept) {
	  ZMuMu_ptetamasscut_counter++;
	  if (isocutAccept) {
	    ZMuMu_passed++;}
	}
      }

      if (ptcutAccept && etacutAccept && masscutAccept && isocutAccept)  {
	ZMuMu_allcut_counter++;
	h_zMuMu_mass_->Fill( m );

	// check the cynematics to fill correct histograms
	for (int j=0;j<numberOfIntervals;j++) {
	  bool statusBinEta = false;
	  bool statusBinPhi = false;
	  bool statusBinPt  = false;
	  double range0 = etamin + j*binEta;
	  double range1= range0 + binEta;
	  double range0phi = phimin + j*binPhi;
	  double range1phi= range0phi + binPhi;
	  double range0pt = ptmin + j*binPt;
	  double range1pt = range0pt + binPt;
	  // eta histograms
	  if (lep1->eta()>=range0 && lep1->eta()<range1)
	    {
	      hmumu_eta[j]->Fill(m);
	      statusBinEta = true;
	    }
	  if (lep2->eta()>=range0 && lep2->eta()<range1 && !statusBinEta){
	    hmumu_eta[j]->Fill(m);                               // If eta1 is in the same bin of eta2 fill just once
	  }
	  // phi histograms
	  if (lep1->phi()>=range0phi && lep1->phi()<range1phi)
	    {
	      hmumu_phi[j]->Fill(m);
	      statusBinPhi = true;
	    }
	  if (lep2->phi()>=range0phi && lep2->phi()<range1phi && !statusBinPhi){
	    hmumu_phi[j]->Fill(m);                               // If phi1 is in the same bin of phi2 fill just once
	  }
	  // pt histograms
	  if (lep1->pt()>=range0pt && lep1->pt()<range1pt)
	    {
	      hmumu_pt[j]->Fill(m);
	      statusBinPt = true;
	    }
	  if (lep2->pt()>=range0pt && lep2->pt()<range1pt && !statusBinPt){
	    hmumu_pt[j]->Fill(m);                               // If pt1 is in the same bin of pt2 fill just once
	  }
	}

	CandMatchMap::const_iterator m0 = zMuMuMap->find(CandRef);
	if( m0 != zMuMuMap->end()) {                                            // the Z is matched to MC thruth
	    h_zMuMuMatched_mass_->Fill( m );
	}
      }
    }
  }

  h_zMuMu_numberOfCand->Fill(nZMuMu);                                             // number of Z cand found per event
  h_zMuMu_numberOfCand_passed->Fill(ZMuMu_allcut_counter);                        // number of Z cand after all cuts found per event
  h_zMuMu_numberOfCand_ptpassed->Fill(ZMuMu_ptcut_counter);                       // number of Z cand afer pt cut found per event
  h_zMuMu_numberOfCand_etapassed->Fill(ZMuMu_etacut_counter);                     // number of Z cand afer eta cut found per event
  h_zMuMu_numberOfCand_masspassed->Fill(ZMuMu_masscut_counter);                   // number of Z cand afer mass cut found per event
  h_zMuMu_numberOfCand_isopassed->Fill(ZMuMu_isocut_counter);                     // number of Z cand afer iso cut found per event
  h_zMuMu_numberOfCand_ptetapassed->Fill(ZMuMu_ptetacut_counter);                 // number of Z cand afer pt&eta cut found per event
  h_zMuMu_numberOfCand_ptetamasspassed->Fill(ZMuMu_ptetamasscut_counter);         // number of Z cand afer pt&eta&mass cut found per event


  //ZmuSingleStandAlone (check MuStandalone if MuMu has not been selected by cuts)
  //  cout << "ZMuMuanalyzer : n of zMuMu " << nZMuMu << " passed " << ZMuMu_passed << "     n. of zStaMu " << nZStandAloneMu << endl;

  if (ZMuMu_passed == 0 && nZStandAloneMu>0 ) {
    //      unsigned int index = 1000;
    for( unsigned int j = 0; j < nZStandAloneMu; j++ ) {
      const Candidate & zsmCand = (*zMuStandAlone)[ j ];
      bool skipZ = false;
      for( unsigned int i = 0; i < nZMuMu; i++ ) {              // chek if the ZMuSTandalone is contained in a ZMuMu
	const Candidate & zmmCand = (*zMuMu)[ i ];        // if yes .. the event has to be skipped
	if (isContained(zmmCand,zsmCand)) skipZ=true;
      }
      if (!skipZ) {                                       // ZSMuSTandalone not contained in a ZMuMu
	ZMuStandalone_notcontained++;
  	CandidateRef CandRef(zMuStandAlone,j);
	CandidateRef lep1 = zsmCand.daughter( 0 )->masterClone().castTo<CandidateRef>();
	CandidateRef lep2 = zsmCand.daughter( 1 )->masterClone().castTo<CandidateRef>();

	ProductID id1 = lep1.id();
	ProductID id2 = lep2.id();
	double iso1 = -1;
	double iso2 = -1;

	if( id1 == muIsoId )
	  iso1 = muIso->value( lep1.key() );
	else if ( id1 == standAloneIsoId )
	  iso1 = standAloneIso->value( lep1.key() );

	if( id2 == muIsoId )
	  iso2 = muIso->value( lep2.key() );
	else if ( id2 == standAloneIsoId )
	  iso2 = standAloneIso->value( lep2.key() );

	double ms = zsmCand.mass();
	if (lep1->pt()>ptSTAcut_ && lep2->pt()>ptSTAcut_ &&
	    fabs(lep1->eta())<etacut_ && fabs(lep2->eta())<etacut_ &&
	    ms>minZmass_ && ms<maxZmass_ && iso1<isocut_ && iso2 <isocut_) {
	  h_zMuSingleStandAlone_mass_->Fill( ms );
	  ZMuStandalone_passed++;
	  // check the cynematics to fill correct histograms
	  for (int j=0;j<numberOfIntervals;j++) {
	    double range0 = etamin + j*binEta;
	    double range1= range0 + binEta;
	    double range0phi = phimin + j*binPhi;
	    double range1phi= range0phi + binPhi;
	    double range0pt = ptmin + j*binPt;
	    double range1pt = range0pt + binPt;

	    // check which muon is a standalone (standalone means that there is a reference missing.)
	    if ((lep1->get<TrackRef,reco::StandAloneMuonTag>()).isNull())
	      {
		if (lep1->eta()>=range0 && lep1->eta()<range1)  	hmusta_eta[j]->Fill(ms);
		if (lep1->phi()>=range0phi && lep1->phi()<range1phi)	hmusta_phi[j]->Fill(ms);
		if (lep1->pt()>=range0pt && lep1->pt()<range1pt)	hmusta_pt[j]->Fill(ms);
	      }
	    if ((lep2->get<TrackRef,reco::StandAloneMuonTag>()).isNull())
	      {
		if (lep2->eta()>=range0 && lep2->eta()<range1)  	hmusta_eta[j]->Fill(ms);
		if (lep2->phi()>=range0phi && lep2->phi()<range1phi)	hmusta_phi[j]->Fill(ms);
		if (lep2->pt()>=range0pt && lep2->pt()<range1pt)	hmusta_pt[j]->Fill(ms);
	      }

	  }
	  CandMatchMap::const_iterator m0 = zMuStandAloneMap->find(CandRef);
	  if( m0 != zMuStandAloneMap->end()) {
	   ZMuStandalone_MCmatched_passed++;
	    h_zMuSingleStandAloneMatched_mass_->Fill( ms );
	  }
	}
      }
    }
    h_zMuSta_numberOfCand->Fill(nZStandAloneMu);                    // number of ZMuStandalone cand found per event (no higher priority Z selected)
    h_zMuSta_numberOfCand_notcontained->Fill(ZMuStandalone_notcontained);
    h_zMuSta_numberOfCand_passed->Fill(ZMuStandalone_passed);        // number of ZMuSTa cand after all cuts found per event (no higher prioriy Z selected)
    h_zMuSta_MCmatched_numberOfCand_passed->Fill(ZMuStandalone_MCmatched_passed);   // number of ZMuSTa MC matched cand after all cuts found per event (no higher prioriy Z selected)

  }

  //ZmuSingleTRack  (check MuTrack if MuMu has not been selected)
  if (ZMuMu_passed == 0 && ZMuStandalone_passed == 0 && nZTrackMu>0) {
    for( unsigned int j = 0; j < nZTrackMu; j++ ) {
      const Candidate & ztmCand = (*zMuTrack)[ j ];
      bool skipZ = false;
      for( unsigned int i = 0; i < nZMuMu; i++ ) {              // chek if the ZMuTrack is contained in a ZMuMu
	const Candidate & zmmCand = (*zMuMu)[ i ];        // if yes .. the event has to be skipped
	if (isContained(zmmCand,ztmCand)) skipZ=true;
      }
      if (!skipZ) {
	ZMuTrack_notcontained++;
	CandidateRef CandRef(zMuTrack,j);
	CandidateRef lep1 = ztmCand.daughter( 0 )->masterClone().castTo<CandidateRef>();
	CandidateRef lep2 = ztmCand.daughter( 1 )->masterClone().castTo<CandidateRef>();

	ProductID id1 = lep1.id();
	ProductID id2 = lep2.id();
	double iso1 = -1;
	double iso2 = -1;

	if( id1 == muIsoId )
	  iso1 = muIso->value( lep1.key() );
	else if ( id1 == trackIsoId )
	  iso1 = trackIso->value( lep1.key() );

	if( id2 == muIsoId )
	  iso2 = muIso->value( lep2.key() );
	else if ( id2 == trackIsoId )
	  iso2 = trackIso->value( lep2.key() );

	double mt = ztmCand.mass();
	if (lep1->pt()>ptcut_ && lep2->pt()>ptcut_ &&
	    fabs(lep1->eta())<etacut_ && fabs(lep2->eta())<etacut_ &&
	    mt>minZmass_ && mt<maxZmass_ && iso1<isocut_ && iso2 <isocut_) {
	  h_zMuSingleTrack_mass_->Fill( mt );
	  ZMuTrack_passed++;

	  // check the cynematics to fill correct histograms
	  for (int j=0;j<numberOfIntervals;j++) {
	    double range0 = etamin + j*binEta;
	    double range1= range0 + binEta;
	    double range0phi = phimin + j*binPhi;
	    double range1phi= range0phi + binPhi;
	    double range0pt = ptmin + j*binPt;
	    double range1pt = range0pt + binPt;

	    // check which muon is a track only (track only means that there is a reference missing.)
	    if ((lep1->get<TrackRef,reco::StandAloneMuonTag>()).isNull())
	      {
		if (lep1->eta()>=range0 && lep1->eta()<range1)  	hmutrack_eta[j]->Fill(mt);
		if (lep1->phi()>=range0phi && lep1->phi()<range1phi)	hmutrack_phi[j]->Fill(mt);
		if (lep1->pt()>=range0pt && lep1->pt()<range1pt)	hmutrack_pt[j]->Fill(mt);
	      }
	    if ((lep2->get<TrackRef,reco::StandAloneMuonTag>()).isNull())
	      {
		if (lep2->eta()>=range0 && lep2->eta()<range1)  	hmutrack_eta[j]->Fill(mt);
		if (lep2->phi()>=range0phi && lep2->phi()<range1phi)	hmutrack_phi[j]->Fill(mt);
		if (lep2->pt()>=range0pt && lep2->pt()<range1pt)	hmutrack_pt[j]->Fill(mt);
	      }
	  }
	  CandMatchMap::const_iterator m0 = zMuTrackMap->find(CandRef);
	  if( m0 != zMuTrackMap->end()) {
	    ZMuTrack_MCmatched_passed++;
	    h_zMuSingleTrackMatched_mass_->Fill( mt );
	  }
	}
      }
    }
    h_zMuTrack_numberOfCand->Fill(nZTrackMu);                     // number of ZMuTrack cand found per event (no higher priority Z selected)
    h_zMuTrack_numberOfCand_notcontained->Fill(ZMuTrack_notcontained); // number of ZMuTrack cand not cntained in ZMuMu (no higher priority Z selected)

    h_zMuTrack_numberOfCand_passed->Fill(ZMuTrack_passed);        // number of ZMuTrack cand after all cuts found per event (no higher priority Z selected)

    h_zMuTrack_MCmatched_numberOfCand_passed->Fill(ZMuTrack_MCmatched_passed);

  }
}

bool ZMuMuAnalyzer_cynematics::isContained(const Candidate & obj1, const Candidate & obj2)
{
  // check if a candidate obj2 is different from obj1  (assume that obj1 is a ZMuMu and obj2 is any other type)
  // (for example a Z can be done with two global muons, or with a global muon plus a standalone muon.
  // if the standalone muon is part of the second global muon in fact this is the same Z)

  const int maxd = 10;
  const Candidate * daughters1[maxd];
  const Candidate * daughters2[maxd];
  TrackRef trackerTrack1[maxd];
  TrackRef stAloneTrack1[maxd];
  TrackRef globalTrack1[maxd];
  TrackRef trackerTrack2[maxd];
  TrackRef stAloneTrack2[maxd];
  TrackRef globalTrack2[maxd];
  bool flag;
  unsigned int nd1 = obj1.numberOfDaughters();
  unsigned int nd2 = obj2.numberOfDaughters();
  unsigned int matched=0;

  for( unsigned int i = 0; i < nd1; ++ i ) {
    daughters1[i] = obj1.daughter( i );
    trackerTrack1[i] = daughters1[i]->get<TrackRef>();
    stAloneTrack1[i] = daughters1[i]->get<TrackRef,reco::StandAloneMuonTag>();
    globalTrack1[i]  = daughters1[i]->get<TrackRef,reco::CombinedMuonTag>();

    /*********************************************** just used for debug ********************
    if (trackerTrack1[i].isNull())
      cout << "in ZMuMu daughter " << i << " tracker ref non found " << endl;
    else
      cout << "in ZMuMu daughter " << i << " tracker ref FOUND"
	   << " id: " << trackerTrack1[i].id() << ", index: " << trackerTrack1[i].key()
	   << endl;
    if (stAloneTrack1[i].isNull())
      cout << "in ZMuMu daughter " << i << " stalone ref non found " << endl;
    else
      cout << "in ZMuMu daughter " << i << " stalone ref FOUND"
	   << " id: " << stAloneTrack1[i].id() << ", index: " << stAloneTrack1[i].key()
	   << endl;

    if (globalTrack1[i].isNull())
      cout << "in ZMuMu daughter " << i << " global ref non found " << endl;
    else
      cout << "in ZMuMu daughter " << i << " global ref FOUND"
	   << " id: " << globalTrack1[i].id() << ", index: " << globalTrack1[i].key()
	   << endl;
    */
  }
  for( unsigned int i = 0; i < nd2; ++ i ) {
    daughters2[i] = obj2.daughter( i );
    trackerTrack2[i] = daughters2[i]->get<TrackRef>();
    stAloneTrack2[i] = daughters2[i]->get<TrackRef,reco::StandAloneMuonTag>();
    globalTrack2[i]  = daughters2[i]->get<TrackRef,reco::CombinedMuonTag>();

    /******************************************** just used for debug ************
    if (trackerTrack2[i].isNull())
      cout << "in ZMuSta daughter " << i << " tracker ref non found " << endl;
    else
      cout << "in ZMuSta daughter " << i << " tracker ref FOUND"
	   << " id: " << trackerTrack2[i].id() << ", index: " << trackerTrack2[i].key()
	   << endl;
    if (stAloneTrack2[i].isNull())
      cout << "in ZMuSta daughter " << i << " standalone ref non found " << endl;
    else
      cout << "in ZMuSta daughter " << i << " standalone ref FOUND"
	   << " id: " << stAloneTrack2[i].id() << ", index: " << stAloneTrack2[i].key()
	   << endl;

    if (globalTrack2[i].isNull())
      cout << "in ZMuSta daughter " << i << " global ref non found " << endl;
    else
      cout << "in ZMuSta daughter " << i << " global ref FOUND"
	   << " id: " << globalTrack2[i].id() << ", index: " << globalTrack2[i].key()
	   << endl;

   */
  }
  if (nd1 != nd2)
    {
      cout << "ZMuMuAnalyzer::isContained WARNING n.of daughters different " << nd1 << "  " << nd2 << endl;
    }
  else
    {
      for (unsigned int i = 0; i < nd1; i++) {
	flag = false;
	for (unsigned int j = 0; j < nd2; j++) {           // if the obj2 is a standalone the trackref is alwais in the trackerTRack position
	  if ( ((trackerTrack2[i].id()==trackerTrack1[j].id()) && (trackerTrack2[i].key()==trackerTrack1[j].key())) ||
	       ((trackerTrack2[i].id()==stAloneTrack1[j].id()) && (trackerTrack2[i].key()==stAloneTrack1[j].key())) ) {
	    flag = true;
	  }
	}
	if (flag) matched++;
      }
    }
  if (matched==nd1) // return true if all the childrens of the ZMuMu have a children matched in ZMuXX
    return true;
  else
    return false;
}

void ZMuMuAnalyzer_cynematics::endJob() {

  // candidate analysis
  // ZMuMu
  double Nzmmc = h_zMuMu_numberOfCand->GetEntries();
  double Nzmmc_0Z = h_zMuMu_numberOfCand->GetBinContent(1);
  double Nzmmc_1Z = h_zMuMu_numberOfCand->GetBinContent(2);
  double Nzmmc_moreZ = Nzmmc-Nzmmc_0Z-Nzmmc_1Z;
  double Nzmmc_passed_0Z = h_zMuMu_numberOfCand_passed->GetBinContent(1);
  double Nzmmc_passed_1Z = h_zMuMu_numberOfCand_passed->GetBinContent(2);
  double Nzmmc_passed_moreZ = Nzmmc-Nzmmc_passed_0Z-Nzmmc_passed_1Z;
  double Nzmmc_ptpassed_0Z = h_zMuMu_numberOfCand_ptpassed->GetBinContent(1);
  double Nzmmc_ptpassed_1Z = h_zMuMu_numberOfCand_ptpassed->GetBinContent(2);
  double Nzmmc_etapassed_0Z = h_zMuMu_numberOfCand_etapassed->GetBinContent(1);
  double Nzmmc_etapassed_1Z = h_zMuMu_numberOfCand_etapassed->GetBinContent(2);
  double Nzmmc_masspassed_0Z = h_zMuMu_numberOfCand_masspassed->GetBinContent(1);
  double Nzmmc_masspassed_1Z = h_zMuMu_numberOfCand_masspassed->GetBinContent(2);
  double Nzmmc_isopassed_0Z = h_zMuMu_numberOfCand_isopassed->GetBinContent(1);
  double Nzmmc_isopassed_1Z = h_zMuMu_numberOfCand_isopassed->GetBinContent(2);
  double Nzmmc_ptetapassed_0Z = h_zMuMu_numberOfCand_ptetapassed->GetBinContent(1);
  double Nzmmc_ptetapassed_1Z = h_zMuMu_numberOfCand_ptetapassed->GetBinContent(2);
  double Nzmmc_ptetamasspassed_0Z = h_zMuMu_numberOfCand_ptetamasspassed->GetBinContent(1);
  double Nzmmc_ptetamasspassed_1Z = h_zMuMu_numberOfCand_ptetamasspassed->GetBinContent(2);
  double Nzmmc_ptpassed_moreZ = Nzmmc-Nzmmc_ptpassed_0Z-Nzmmc_ptpassed_1Z;
  double Nzmmc_etapassed_moreZ = Nzmmc-Nzmmc_etapassed_0Z-Nzmmc_etapassed_1Z;
  double Nzmmc_masspassed_moreZ = Nzmmc-Nzmmc_masspassed_0Z-Nzmmc_masspassed_1Z;
  double Nzmmc_isopassed_moreZ = Nzmmc-Nzmmc_isopassed_0Z-Nzmmc_isopassed_1Z;
  double Nzmmc_ptetapassed_moreZ = Nzmmc-Nzmmc_ptetapassed_0Z-Nzmmc_ptetapassed_1Z;
  double Nzmmc_ptetamasspassed_moreZ = Nzmmc-Nzmmc_ptetamasspassed_0Z-Nzmmc_ptetamasspassed_1Z;
  double Nzmsc = h_zMuSta_numberOfCand->GetEntries();
  double Nzmsc_0Z = h_zMuSta_numberOfCand->GetBinContent(1);
  double Nzmsc_1Z = h_zMuSta_numberOfCand->GetBinContent(2);
  double Nzmsc_moreZ = Nzmsc - Nzmsc_0Z - Nzmsc_1Z;
  double Nzmsc_notcontained_0Z = h_zMuSta_numberOfCand_notcontained->GetBinContent(1);
  double Nzmsc_notcontained_1Z = h_zMuSta_numberOfCand_notcontained->GetBinContent(2);
  double Nzmsc_notcontained_moreZ = Nzmsc-Nzmsc_notcontained_0Z-Nzmsc_notcontained_1Z;
  double Nzmsc_passed_0Z = h_zMuSta_numberOfCand_passed->GetBinContent(1);
  double Nzmsc_passed_1Z = h_zMuSta_numberOfCand_passed->GetBinContent(2);
  double Nzmsc_passed_moreZ = Nzmsc - Nzmsc_passed_0Z - Nzmsc_passed_1Z;
  double Nzmsc_MCmatched_passed_0Z = h_zMuSta_MCmatched_numberOfCand_passed->GetBinContent(1);
  double Nzmsc_MCmatched_passed_1Z = h_zMuSta_MCmatched_numberOfCand_passed->GetBinContent(2);
  double Nzmsc_MCmatched_passed_moreZ = Nzmsc - Nzmsc_MCmatched_passed_0Z - Nzmsc_MCmatched_passed_1Z;
  double Nzmtc = h_zMuTrack_numberOfCand->GetEntries();
  double Nzmtc_0Z = h_zMuTrack_numberOfCand->GetBinContent(1);
  double Nzmtc_1Z = h_zMuTrack_numberOfCand->GetBinContent(2);
  double Nzmtc_moreZ = Nzmtc - Nzmtc_0Z - Nzmtc_1Z;
  double Nzmtc_notcontained_0Z = h_zMuTrack_numberOfCand_notcontained->GetBinContent(1);
  double Nzmtc_notcontained_1Z = h_zMuTrack_numberOfCand_notcontained->GetBinContent(2);
  double Nzmtc_notcontained_moreZ = Nzmtc-Nzmtc_notcontained_0Z-Nzmtc_notcontained_1Z;
  double Nzmtc_passed_0Z = h_zMuTrack_numberOfCand_passed->GetBinContent(1);
  double Nzmtc_passed_1Z = h_zMuTrack_numberOfCand_passed->GetBinContent(2);
  double Nzmtc_passed_moreZ = Nzmtc - Nzmtc_passed_0Z - Nzmtc_passed_1Z;
  double Nzmtc_MCmatched_passed_0Z = h_zMuTrack_MCmatched_numberOfCand_passed->GetBinContent(1);
  double Nzmtc_MCmatched_passed_1Z = h_zMuTrack_MCmatched_numberOfCand_passed->GetBinContent(2);
  double Nzmtc_MCmatched_passed_moreZ = Nzmtc - Nzmtc_MCmatched_passed_0Z - Nzmtc_MCmatched_passed_1Z;

  cout << "--------------- Statistics ----------------------------------------------------------" << endl;
  cout << "n of ZMuMu entries   ...................................................... " << Nzmmc << endl;
  cout << "n of ZMuMu events with 0 cand ............................................. " << Nzmmc_0Z << endl;
  cout << "n of ZMuMu events with 1 cand ............................................. " << Nzmmc_1Z << endl;
  cout << "n of ZMuMu events with 2 or more cand ..................................... " << Nzmmc_moreZ << endl << endl ;

  cout << "n of ZMuMu events not selected by cuts .................................... " << Nzmmc_passed_0Z << endl;
  cout << "n of ZMuMu events with 1 cand selected by cuts ............................ " << Nzmmc_passed_1Z << endl;
  cout << "n of ZMuMu events with 2 or more cand elected by cuts ..................... " << Nzmmc_passed_moreZ << endl<< endl ;

  cout << "n of ZMuMu events not selected by pt cut .................................. " << Nzmmc_ptpassed_0Z << endl;
  cout << "n of ZMuMu events with 1 cand selected by pt cut .......................... " << Nzmmc_ptpassed_1Z << endl;
  cout << "n of ZMuMu events with 2 or more cand elected by pt cut ................... " << Nzmmc_ptpassed_moreZ << endl<< endl ;

  cout << "n of ZMuMu events not selected by eta cut ................................. " << Nzmmc_etapassed_0Z << endl;
  cout << "n of ZMuMu events with 1 cand selected by eta cut ......................... " << Nzmmc_etapassed_1Z << endl;
  cout << "n of ZMuMu events with 2 or more cand elected by eta cut .................. " << Nzmmc_etapassed_moreZ << endl<< endl ;

  cout << "n of ZMuMu events not selected by mass cut ................................ " << Nzmmc_masspassed_0Z << endl;
  cout << "n of ZMuMu events with 1 cand selected by mass cut ........................ " << Nzmmc_masspassed_1Z << endl;
  cout << "n of ZMuMu events with 2 or more cand elected by mass cut ................. " << Nzmmc_masspassed_moreZ << endl<< endl ;

  cout << "n of ZMuMu events not selected by iso cut ................................. " << Nzmmc_isopassed_0Z << endl;
  cout << "n of ZMuMu events with 1 cand selected iso cut ............................ " << Nzmmc_isopassed_1Z << endl;
  cout << "n of ZMuMu events with 2 or more cand elected iso cut ..................... " << Nzmmc_isopassed_moreZ << endl<< endl ;

  cout << "n of ZMuMu events not selected by pt and eta cut .......................... " << Nzmmc_ptetapassed_0Z << endl;
  cout << "n of ZMuMu events with 1 cand selected by pt and eta cut .................. " << Nzmmc_ptetapassed_1Z << endl;
  cout << "n of ZMuMu events with 2 or more cand elected by pt and eta cut ........... " << Nzmmc_ptetapassed_moreZ << endl<< endl ;

  cout << "n of ZMuMu events not selected by pt and eta and mass cut ................. " << Nzmmc_ptetamasspassed_0Z << endl;
  cout << "n of ZMuMu events with 1 cand selected by pt and eta and mass cut ......... " << Nzmmc_ptetamasspassed_1Z << endl;
  cout << "n of ZMuMu events with 2 or more cand elected by pt and eta and mass cut .. " << Nzmmc_ptetamasspassed_moreZ << endl<< endl ;

  cout << "................When No ZMuMu are selected.................................." << endl;
  cout << "n of ZMuSta entries ....................................................... " << Nzmsc << endl;
  cout << "n of ZMuSta events with 0 cand ............................................ " << Nzmsc_0Z << endl;
  cout << "n of ZMuSta events with 1 cand ............................................ " << Nzmsc_1Z << endl;
  cout << "n of ZMuSta events with 2 or more cand .................................... " << Nzmsc_moreZ << endl<< endl ;

  cout << "n of ZMuSta not contained events with 0 cand .............................. " << Nzmsc_notcontained_0Z << endl;
  cout << "n of ZMuSta events not contained with 1 cand .............................. " << Nzmsc_notcontained_1Z << endl;
  cout << "n of ZMuSta events no contained with 2 or more cand ....................... " << Nzmsc_notcontained_moreZ << endl<< endl ;

  cout << "n of ZMuSta cand not selectd by cuts ...................................... " << Nzmsc_passed_0Z << endl;
  cout << "n of ZMuSta events with 1 cand selected by cuts ........................... " << Nzmsc_passed_1Z << endl;
  cout << "n of ZMuSta events with 2 or more cand selected by cuts ................... " << Nzmsc_passed_moreZ << endl<< endl ;

  cout << "n of ZMuSta MCmatched cand not selectd by cuts ............................ " << Nzmsc_MCmatched_passed_0Z << endl;
  cout << "n of ZMuSta MCmatched events with 1 cand selected by cuts ................. " << Nzmsc_MCmatched_passed_1Z << endl;
  cout << "n of ZMuSta MCmatched events with 2 or more cand selected by cuts ......... " << Nzmsc_MCmatched_passed_moreZ << endl<< endl ;

  cout << "...............When no ZMuMu and ZMuSta are selcted........................." << endl;
  cout << "n of ZMuTrack entries   ................................................... " << Nzmtc << endl;
  cout << "n of ZMuTrack events with 0 cand ..........................................." << Nzmtc_0Z << endl;
  cout << "n of ZMuTrack events with 1 cand ..........................................." << Nzmtc_1Z << endl;
  cout << "n of ZMuTrack events with 2 or more cand ..................................." << Nzmtc_moreZ << endl<< endl ;

  cout << "n of ZMuTrack not contained events with 0 cand ............................ " << Nzmtc_notcontained_0Z << endl;
  cout << "n of ZMuTrack events not contained with 1 cand ............................ " << Nzmtc_notcontained_1Z << endl;
  cout << "n of ZMuTrack events no contained with 2 or more cand ..................... " << Nzmtc_notcontained_moreZ << endl<< endl ;

  cout << "n of ZMuTrack cand not selectd by cuts ....................................." << Nzmtc_passed_0Z << endl;
  cout << "n of ZMuTrack events with 1 cand selected by cuts .........................." << Nzmtc_passed_1Z << endl;
  cout << "n of ZMuTrack events with 2 or more cand selected by cuts .................." << Nzmtc_passed_moreZ << endl<< endl ;

  cout << "n of ZMuTrack MCmatched cand not selectd by cuts .......................... " << Nzmtc_MCmatched_passed_0Z << endl;
  cout << "n of ZMuTrcak MCmatched events with 1 cand selected by cuts ............... " << Nzmtc_MCmatched_passed_1Z << endl;
  cout << "n of ZMuTrack MCmatched events with 2 or more cand selected by cuts ....... " << Nzmtc_MCmatched_passed_moreZ << endl;

  cout << "------------------------------------------------------------------------------------------" << endl;

  double Nzmm = h_zMuMu_mass_->GetEntries() ;
  double Nzsm = h_zMuSingleStandAlone_mass_->GetEntries()  ;
  double Nzsnom = h_zMuSingleStandAloneOverlap_mass_->GetEntries()  ;
  double Nztm = h_zMuSingleTrack_mass_->GetEntries();

  double NzmmMatch = h_zMuMuMatched_mass_->GetEntries() ;
  double NzsmMatch = h_zMuSingleStandAloneMatched_mass_->GetEntries()  ;
  double NzsnomMatch = h_zMuSingleStandAloneOverlapMatched_mass_->GetEntries()  ;
  double NztmMatch = h_zMuSingleTrackMatched_mass_->GetEntries();

  cout<<"-- N SingleTrackMu = "<<Nztm<<endl;
  cout<<"-----N SinglStandAloneMu = "<<Nzsm<<endl;
  cout<<"-----N SingleStandAloneOverlapMu = "<<Nzsnom<<endl;
  cout<<"------- N MuMu = "<<Nzmm<<endl;

  cout<<"-- N SingleTrackMuMatched = "<<NztmMatch<<endl;
  cout<<"-----N SinglStandAloneMuMatched = "<<NzsmMatch<<endl;
  cout<<"-----N SingleStandAloneOverlapMuMatched  = "<<NzsnomMatch<<endl;
  cout<<"------- N MuMu Matched  = "<<NzmmMatch<<endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuAnalyzer_cynematics);

