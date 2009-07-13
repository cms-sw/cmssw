/*
 * Package:  TopHLTDiMuonDQM
 *   Class:  TopHLTDiMuonDQM
 *
 * Original Author:  Muriel VANDER DONCKT *:0
 *         Created:  Wed Dec 12 09:55:42 CET 2007
 *   Original Code:  HLTMuonRecoDQMSource.cc,v 1.2 2008/10/16 16:41:29 hdyoo Exp $
 *
 */

#include "DQM/Physics/src/TopHLTDiMuonDQM.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;


//
// constructors and destructor
//

TopHLTDiMuonDQM::TopHLTDiMuonDQM( const ParameterSet& parameters_ ) : counterEvt_( 0 )
{

  verbose_           = parameters_.getUntrackedParameter<bool>("verbose", false);
  monitorName_       = parameters_.getUntrackedParameter<string>("monitorName", "Top/HLTDiMuons");
  level_             = parameters_.getUntrackedParameter<int>("Level", 1);
  prescaleEvt_       = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  candCollectionTag_ = parameters_.getUntrackedParameter<InputTag>("candCollection", edm::InputTag("hltL1extraParticles"));

  dbe_ = Service<DQMStore>().operator->();

}


TopHLTDiMuonDQM::~TopHLTDiMuonDQM() {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginJob(const EventSetup& context) {

  dbe_ = Service<DQMStore>().operator->();

  if( dbe_ ) {

    dbe_->setCurrentFolder("monitorName_");
    if( monitorName_ != "" )  monitorName_ = monitorName_+"/" ;
    if( verbose_ )  cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;

    char name[512];

    sprintf(name,"Level%i",level_);
    dbe_->setCurrentFolder(monitorName_+name);

    NMuons = dbe_->book1D("HLTDimuon_NMuons", "Number of muons", 10, 0., 10.);
    NMuons->setAxisTitle("Number of muons", 1);

    PtMuons = dbe_->book1D("HLTDimuon_Pt","P_T of muons", 100, 0., 200.);
    PtMuons->setAxisTitle("P^{muon}_{T}  (GeV)", 1);

    EtaMuons = dbe_->book1D("HLTDimuon_Eta","Pseudorapidity of muons", 100, -5., 5.);
    EtaMuons->setAxisTitle("#eta_{muon}", 1);

    PhiMuons = dbe_->book1D("HLTDimuon_Phi","Azimutal angle of muons", 70, -3.5, 3.5);
    PhiMuons->setAxisTitle("#phi_{muon}  (rad)", 1);

    DiMuonMass = dbe_->book1D("HLTDimuon_DiMuonMass","Invariant Dimuon Mass", 100, 0., 200.);
    DiMuonMass->setAxisTitle("Invariant #mu #mu mass  (GeV)", 1);

    DeltaEtaMuons = dbe_->book1D("HLTDimuon_DeltaEta","#Delta #eta of muon pair", 100, -5., 5.);
    DeltaEtaMuons->setAxisTitle("#Delta #eta_{#mu #mu}", 1);

    DeltaPhiMuons = dbe_->book1D("HLTDimuon_DeltaPhi","#Delta #phi of muon pair", 100, -5., 5.);
    DeltaPhiMuons->setAxisTitle("#Delta #phi_{#mu #mu}  (rad)", 1);
  }

} 


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginRun(const Run& r, const EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

}


// ----------------------------------------------------------
void TopHLTDiMuonDQM::analyze(const Event& iEvent, const EventSetup& iSetup ) {

  if( !dbe_ ) return;

  counterEvt_++;

  Handle<L1MuonParticleCollection> mucands;
  iEvent.getByLabel (candCollectionTag_, mucands);

  if( mucands.failedToGet() ) {

    cout << endl << "-----------------------" << endl;
    cout << "--- NO HLT MUONS !! ---" << endl;
    cout << "-----------------------" << endl << endl;

    return;

  }

  NMuons->Fill(mucands->size());

  //  cout << endl << "--------------------" << endl;
  //  cout << " Nmuons: " << mucands->size() << endl;
  //  cout << "--------------------" << endl << endl;

  if( mucands->size() > 1 ) {

    L1MuonParticleCollection::const_reference mu1 = mucands->at(0);
    L1MuonParticleCollection::const_reference mu2 = mucands->at(1);

    DeltaEtaMuons->Fill( mu1.eta()-mu2.eta() );
    DeltaPhiMuons->Fill( mu1.phi()-mu2.phi() );

    double dilepMass = sqrt( (mu1.energy() + mu2.energy())*(mu1.energy() + mu2.energy())
			     - (mu1.px() + mu2.px())*(mu1.px() + mu2.px())
			     - (mu1.py() + mu2.py())*(mu1.py() + mu2.py())
			     - (mu1.pz() + mu2.pz())*(mu1.pz() + mu2.pz()) );

    DiMuonMass->Fill( dilepMass );

  }

  L1MuonParticleCollection::const_iterator cand, cand2;

  if( !mucands.failedToGet() ) {

    for( cand = mucands->begin(); cand != mucands->end(); ++cand ) {

      PtMuons->Fill(  cand->pt()  );
      EtaMuons->Fill( cand->eta() );
      PhiMuons->Fill( cand->phi() );

    }

  }

  /*

  if( !mucands.failedToGet() ) {

    if( verbose_ )  cout << " filling Reco stuff " << endl;

    NMuons->Fill(mucands->size());

    for( cand=mucands->begin(); cand != mucands->end(); ++cand ) {

      TrackRef tk = cand->get<TrackRef>();

      // eta cut
      hpt->Fill(tk->pt());
      hcharge->Fill(tk->charge());

      if( tk->charge() != 0 ) {

	heta->Fill(tk->eta());
	hphi->Fill(tk->phi());
	hetaphi->Fill(tk->phi(), tk->eta());
	hptphi->Fill(tk->pt(), tk->phi());
	hpteta->Fill(tk->pt(), tk->eta());
	hnhit->Fill(tk->numberOfValidHits());
	hd0->Fill(tk->d0());

	//        if( !recoBeamSpotHandle.failedToGet() ) {
	//
	//	  hdr->Fill(tk->dxy(beamSpot.position()));
	//	  hdrphi->Fill(tk->phi(),tk->dxy(beamSpot.position()));
	//
	//	}

	hd0phi->Fill(tk->phi(), tk->d0());
	hdz->Fill(tk->dz());
	hdzeta->Fill(tk->eta(), tk->dz());
	herr0->Fill(tk->error(0));

	cand2 = cand;
	++cand2;

	for( ; cand2!=mucands->end(); cand2++ ) {

	  TrackRef tk2 = cand2->get<TrackRef>();

	  if( tk->charge()*tk2->charge() == -1 ) {

	    double mass = (cand->p4()+cand2->p4()).M();
	    hdimumass->Fill(mass);

	  }

	}

        if ( level_ == 3 ) {

          TrackRef l2tk = tk->seedRef().castTo<Ref<L3MuonTrajectorySeedCollection>>()->l2Track();

	  if( tk->pt()*l2tk->pt() != 0 )  hptres->Fill( 1/tk->pt()-1/l2tk->pt() );

	  hetares->Fill( tk->eta()-l2tk->eta() );
	  hetareseta->Fill( tk->eta(), tk->eta()-l2tk->eta() );
	  hphires->Fill( tk->phi()-l2tk->phi() );

	  double dphi = tk->phi()-l2tk->phi();

	  if(      dphi >  TMath::TwoPi() )  dphi -= 2*TMath::TwoPi();
	  else if( dphi < -TMath::TwoPi() )  dphi +=   TMath::TwoPi();

	  hphiresphi->Fill( tk->phi(), dphi );

	}

	else {

	  Handle<L2MuonTrajectorySeedCollection> museeds;
	  iEvent.getByLabel(l2seedscollectionTag_, museeds);

	  if( !museeds.failedToGet() ) {

	    RefToBase<TrajectorySeed> seed = tk->seedRef();
	    L1MuonParticleRef l1ref;

	    for( uint iMuSeed=0; iMuSeed != museeds->size(); ++iMuSeed ) {

	      Ref<L2MuonTrajectorySeedCollection> l2seed(museeds,iMuSeed);

	      if( l2seed.id() == seed.id() && l2seed.key() == seed.key() ) {
		l1ref = l2seed->l1Particle();
		break;
	      }

	    }

	    if( tk->pt()*l1ref->pt() != 0 )  hptres->Fill( 1/tk->pt()-1/l1ref->pt() );

	    hetares->Fill( tk->eta()-l1ref->eta() );
	    hetareseta->Fill( tk->eta(), tk->eta()-l1ref->eta() );
	    hphires->Fill( tk->phi()-l1ref->phi() );

	    double dphi = tk->phi() - l1ref->phi();

	    if(      dphi >  TMath::TwoPi() )  dphi -= 2*TMath::TwoPi();
	    else if( dphi < -TMath::TwoPi() )  dphi +=   TMath::TwoPi();

	    hphiresphi->Fill( tk->phi(), dphi );

	  }

	}

      }

      else LogWarning("HLTMonMuon")<<"stop filling candidate with update@Vtx failure";
    }

  }

  */

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::endRun(const Run& r, const EventSetup& context) {

}


//--------------------------------------------------------
void TopHLTDiMuonDQM::endJob() {

  LogInfo("HLTMonMuon") << "analyzed " << counterEvt_ << " events";
 
  if( outputFile_.size() != 0 && dbe_ )  dbe_->save(outputFile_);
  return;

}
