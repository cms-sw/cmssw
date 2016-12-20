/* \class MCEfficiencyAnalyzer
 *
 * Muon reconstruction efficiency from MC truth,
 * for global muons, tracks and standalone. Take as input the output of the
 * standard EWK skim: zToMuMu
 *
 * Produces in output the efficency number
 *
 * \author Michele de Gruttola, INFN Naples
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include <iostream>
#include <iterator>
using namespace edm;
using namespace std;
using namespace reco;

class MCEfficiencyAnalyzer : public edm::EDAnalyzer
{
public:
  MCEfficiencyAnalyzer(const edm::ParameterSet& pset) :
    zMuMuToken_( consumes<CandidateCollection>( pset.getParameter<InputTag>( "zMuMu" ) ) ),
    MuonsToken_( consumes<CandidateCollection>( pset.getParameter<InputTag>( "Muons" ) ) ),
    MuonsMapToken_( consumes<CandMatchMap>( pset.getParameter<InputTag>( "MuonsMap" ) ) ),
    TracksToken_( consumes<CandidateCollection>( pset.getParameter<InputTag>( "Tracks" ) ) ),
    TracksMapToken_( consumes<CandMatchMap>( pset.getParameter<InputTag>( "TracksMap" ) ) ),
    genParticlesToken_( consumes<GenParticleCollection>( pset.getParameter<InputTag>( "genParticles" ) ) ),
    StandAloneToken_( consumes<CandidateCollection>( pset.getParameter<InputTag>( "StandAlone" ) ) ),
    StandAloneMapToken_( consumes<CandMatchMap>( pset.getParameter<InputTag>( "StandAloneMap" ) ) ),
    etacut_( pset.getParameter<double>( "etacut" ) ),
    ptcut_( pset.getParameter<double>( "ptcut" ) ),
    deltaRStacut_( pset.getParameter<double>( "deltaRStacut" ) )

  {
    nMuMC =0; nMureco =0; nTrk=0; nSta=0; nNotMuMatching =0 ;
  }

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override
  {

    Handle<CandidateCollection> zMuMu;
    event.getByToken(zMuMuToken_, zMuMu);


    Handle<CandidateCollection> Muons;
    event.getByToken(MuonsToken_, Muons);

    Handle<CandidateCollection> Tracks;
    event.getByToken(TracksToken_, Tracks);

    Handle<CandidateCollection> StandAlone;
    event.getByToken(StandAloneToken_, StandAlone);

    Handle<CandMatchMap> MuonsMap;
    event.getByToken(MuonsMapToken_, MuonsMap);

    Handle<CandMatchMap> TracksMap;
    event.getByToken(TracksMapToken_, TracksMap);

    Handle<CandMatchMap> StandAloneMap;
    event.getByToken(StandAloneMapToken_, StandAloneMap);

    Handle<GenParticleCollection> genParticles;
    event.getByToken(genParticlesToken_, genParticles);

    //Getting muons from Z MC
    for( unsigned int k = 0; k < genParticles->size(); k++ )
      {
	const Candidate & ZCand = (*genParticles)[ k ];
	int status = ZCand.status();

	if (ZCand.pdgId()==23&& status==3 )
	  {
	    // positive muons
	    const Candidate  *  muCand1 =   ZCand.daughter(0);
	    if (muCand1->status()==3)
	      {
		for (unsigned int d =0 ; d< muCand1->numberOfDaughters(); d++)
		  {
		    const Candidate * muCandidate = muCand1->daughter(d);
		    if (muCandidate->pdgId() == muCand1->pdgId() )
		      {
			muCand1 = muCand1->daughter(d);
		      }
		  }
	      }
	    // negative muons
	    const Candidate  *  muCand2 =   ZCand.daughter(1);
	    if (muCand2->status()==3)
	      {
		for (unsigned int e =0 ; e< muCand2->numberOfDaughters(); e++)
		  {
		    const Candidate * muCandidate = muCand2->daughter(e);
		    if (muCandidate->pdgId() == muCand2->pdgId() )
		      {
			muCand2 = muCand2->daughter(e);
		      }
		  }
	      }

	    double deltaR_Mu_Sta =0;
	    int nMurecoTemp = nMureco;
	    // getting mu matched
	    CandMatchMap::const_iterator i;
	    for(i = MuonsMap->begin(); i != MuonsMap->end(); i++ )
	      {
		const Candidate/* & reco = * i -> key,*/ &  mc =  * i -> val;
		if ((muCand1 == &mc) && (mc.pt()>ptcut_) && (abs(mc.eta()<etacut_)))
		  {
		    nMuMC++;
		    nMureco++;
		    break;
		  }
	      }
	    if (nMureco == nMurecoTemp ) // I.E. MU RECO NOT FOUND!!!!
	      {
                int nTrkTemp = nTrk;
		int nStaTemp = nSta;

		// getting tracks matched and doing the same, CONTROLLING IF MU IS RECONSTRUCTED AS A TRACK AND NOT AS A MU
		CandMatchMap::const_iterator l;
		for(l = TracksMap->begin(); l != TracksMap->end(); l++ )
		  {
		    const Candidate /* & Trkreco = * l -> key, */ &  Trkmc =  * l -> val;
		    if (( muCand1 == & Trkmc) && (Trkmc.pt()>ptcut_) && (abs(Trkmc.eta()<etacut_)))
		      {
		        nMuMC++;
			nTrk++;
			break;
		      }
		  }
		// the same for standalone
		CandMatchMap::const_iterator n;
		for(n = StandAloneMap->begin(); n != StandAloneMap->end(); n++ )
		  {
		    const Candidate & Stareco = * n -> key, &  Stamc =  * n -> val;
		    if ((muCand1 == &Stamc ) && (Stamc.pt()>ptcut_) && (abs(Stamc.eta()<etacut_)))
		      {
			nMuMC++;
			nSta++;
			deltaR_Mu_Sta = deltaR(Stareco, *muCand1);

			//	cout<<"Ho trovato un sta reco "<<endl;
			break;
		      }
		  }
		// controlling if sta and trk are reconstrucetd both, and if so the get the deltaR beetween muon MC and reco sta, to controll correlation to this happening
		if ((nSta == nStaTemp + 1) && (nTrk == nTrkTemp + 1 ) )
		  {
		    nNotMuMatching ++;
		    if ((deltaR_Mu_Sta< deltaRStacut_))
		      {
		    v_.push_back(deltaR_Mu_Sta) ;
                   cout << "Not matching from trk and sta matched to MC mu, to reconstruct a recoMU" << endl;
		      }
		  }
	      }
	  }
      }
  }


  virtual void endJob() override
  {


    cout <<"--- nMuMC == "<<nMuMC<<endl;
    cout <<"--- nMureco == "<<nMureco<<endl;
    cout <<"--- nSta == "<<nSta<<endl;
    cout <<"--- nTrk == "<<nTrk<<endl;
    cout <<"--- nNotMuMatching from a trk and sta matched to a Mu MC == "<<nNotMuMatching<<endl;
    if (nMuMC!=0 )
      {
	cout<<" effMu == "<<(double) nMureco/nMuMC<<endl;
	cout<<" effTrk == "<< (double)(nTrk + nMureco) /nMuMC<<endl;
	cout<<" effSta == "<< (double)(nSta + nMureco) / nMuMC<<endl;
      }

vector< int >::const_iterator p2;
    for (unsigned int i =0 ; i < v_.size(); ++i )
      {
	cout<<" delta R Mu Sta == "<< v_[i]<<endl;
        }
  }





private:

  EDGetTokenT<CandidateCollection> zMuMuToken_;
  EDGetTokenT<CandidateCollection> MuonsToken_;
  EDGetTokenT<CandMatchMap> MuonsMapToken_;
  EDGetTokenT<CandidateCollection> TracksToken_;
  EDGetTokenT<CandMatchMap> TracksMapToken_;
  EDGetTokenT<GenParticleCollection> genParticlesToken_;
  EDGetTokenT<CandidateCollection> StandAloneToken_;
  EDGetTokenT<CandMatchMap> StandAloneMapToken_;
  double etacut_, ptcut_, deltaRStacut_;
  int nMuMC, nMureco, nTrk, nSta, nNotMuMatching;
  vector<double> v_;
};


DEFINE_FWK_MODULE(MCEfficiencyAnalyzer);
