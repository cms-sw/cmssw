/* \class ZMuMuAnalyzer
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
 * \author Michele de Gruttola, INFN Naples
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
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "TH1.h"
#include <iostream>
#include <iterator>
using namespace edm;
using namespace std;
using namespace reco;

typedef edm::AssociationVector<reco::CandidateRefProd, std::vector<double> > IsolationCollection;

class ZMuMuAnalyzer : public edm::EDAnalyzer {
public:
  ZMuMuAnalyzer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
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
  TH1D * h_zMuMu_mass_, * h_zMuSingleTrack_mass_, * h_zMuSingleStandAlone_mass_,* h_zMuSingleStandAloneOverlap_mass_,
    * h_zMuMuMatched_mass_,
 *h_zMuSingleTrackMatched_mass_,
    * h_zMuSingleStandAloneMatched_mass_,
    * h_zMuSingleStandAloneOverlapMatched_mass_;
};

ZMuMuAnalyzer::ZMuMuAnalyzer(const edm::ParameterSet& pset) :
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
  h_zMuMu_mass_ = fs->make<TH1D>( "ZMuMumass", "ZMuMu mass(GeV)", 200,  0., 200. );
  h_zMuSingleTrack_mass_ = fs->make<TH1D>( "ZMuSingleTrackmass", "ZMuSingleTrack mass(GeV)", 100,  0., 200. );
  h_zMuSingleStandAlone_mass_ = fs->make<TH1D>( "ZMuSingleStandAlonemass", "ZMuSingleStandAlone mass(GeV)", 50,  0., 200. );
  h_zMuSingleStandAloneOverlap_mass_ = fs->make<TH1D>( "ZMuSingleStandAloneOverlapmass", "ZMuSingleStandAloneOverlap  mass(GeV)", 50,  0., 200. );


  h_zMuMuMatched_mass_ = fs->make<TH1D>( "ZMuMuMatchedmass", "ZMuMu Matched  mass(GeV)", 200,  0., 200. );
  h_zMuSingleTrackMatched_mass_ = fs->make<TH1D>( "ZMuSingleTrackmassMatched", "ZMuSingleTrackMatched mass(GeV)", 100,  0., 200. );
  h_zMuSingleStandAloneMatched_mass_ = fs->make<TH1D>( "ZMuSingleStandAlonemassMatched", "ZMuSingleStandAloneMatched mass(GeV)", 50,  0., 200. );
  h_zMuSingleStandAloneOverlapMatched_mass_ = fs->make<TH1D>( "ZMuSingleStandAloneOverlapmassMatched", "ZMuSingleStandAloneMatched Overlap  mass(GeV)", 50,  0., 200. );
}

void ZMuMuAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  Handle<CandidateCollection> zMuMu;
  event.getByToken(zMuMuToken_, zMuMu);
  Handle<CandidateCollection> zMuTrack;
  event.getByToken( zMuTrackToken_, zMuTrack );
  Handle<CandidateCollection> zMuStandAlone;
  event.getByToken( zMuStandAloneToken_, zMuStandAlone );

  unsigned int nZMuMu = zMuMu->size();
  unsigned int nZTrackMu = zMuTrack->size();
  unsigned int nZStandAloneMu = zMuStandAlone->size();
  static const double zMass = 91.1876; // PDG Z mass

  //  cout << "nZMuMu = " << nZMuMu << endl;
  //  cout << "nZTrackMu = " << nZTrackMu << endl;
  //  cout << "nZStandAloneMu = " << nZStandAloneMu << endl;

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
    double mass = 1000000.;
    for( unsigned int i = 0; i < nZMuMu; i++ ) {
      const Candidate & zmmCand = (*zMuMu)[ i ];
      CandidateRef CandRef(zMuMu,i);
      CandidateRef lep1 = zmmCand.daughter( 0 )->masterClone().castTo<CandidateRef>();
      CandidateRef lep2 = zmmCand.daughter( 1 )->masterClone().castTo<CandidateRef>();

      const  double iso1 = muIso->value( lep1.key() );
      const  double iso2 = muIso->value( lep2.key() );

      double m = zmmCand.mass();
      if (lep1->pt()>ptcut_ && lep2->pt()>ptcut_ &&
	  fabs(lep1->eta())<etacut_ && fabs(lep2->eta())<etacut_ &&
	  m>minZmass_ && m<maxZmass_ && iso1 < isocut_ && iso2 <isocut_) {
	if ( fabs( mass - zMass ) > fabs( m - zMass ) ) {
	  mass = m;
	}

	h_zMuMu_mass_->Fill( mass );
	CandMatchMap::const_iterator m0 = zMuMuMap->find(CandRef);
	if( m0 != zMuMuMap->end()) {
	    h_zMuMuMatched_mass_->Fill( mass );
	}
      }
    }
  }

  //ZmuSingleTRack
  if (nZMuMu ==0 && nZTrackMu>0) {
    for( unsigned int j = 0; j < nZTrackMu; j++ ) {
      const Candidate & ztmCand = (*zMuTrack)[ j ];
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
	CandMatchMap::const_iterator m0 = zMuTrackMap->find(CandRef);
	if( m0 != zMuTrackMap->end()) {
	  h_zMuSingleTrackMatched_mass_->Fill( mt );
	}
      }
    }
  }

  //ZmuSingleStandAlone
  if (nZMuMu ==0 && nZStandAloneMu>0) {
    //      unsigned int index = 1000;
    for( unsigned int j = 0; j < nZStandAloneMu; j++ ) {
      const Candidate & zsmCand = (*zMuStandAlone)[ j ];
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
	CandMatchMap::const_iterator m0 = zMuStandAloneMap->find(CandRef);
	if( m0 != zMuStandAloneMap->end()) {
	  h_zMuSingleStandAloneMatched_mass_->Fill( ms );
	}

	bool noOverlap = true;
	for( unsigned int j = 0; j < zMuTrack->size(); j++ ) {
	  const Candidate & ztmCand = (*zMuTrack)[ j ];
	  CandidateRef CandReft(zMuTrack,j);

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

	    if ( overlap_( ztmCand, zsmCand ) ) {
	      noOverlap = false;
	      break;
	    }
	    if (!noOverlap ) {
	      h_zMuSingleStandAloneOverlap_mass_->Fill( ms );
	      CandMatchMap::const_iterator m1 = zMuTrackMap->find(CandReft);
	      CandMatchMap::const_iterator m2 = zMuStandAloneMap->find(CandRef);

	      if( m1 != zMuTrackMap->end()  && m2 != zMuStandAloneMap->end()  ) {
		h_zMuSingleStandAloneOverlapMatched_mass_->Fill( ms );
	      }
	    }
	  }
	}
      }
    }
  }
}

void ZMuMuAnalyzer::endJob() {
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

DEFINE_FWK_MODULE(ZMuMuAnalyzer);

