#include <TMath.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeavyFlavorAnalysis/Skimming/interface/Combinatorics.h"

#include "HeavyFlavorAnalysis/Skimming/interface/Tau3MuReco.h"

Tau3MuReco::Tau3MuReco(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC):m_kMatchingDeltaR(iConfig.getParameter<double>("RecoAnalysisMatchingDeltaR")),
							 m_kMatchingPt(iConfig.getParameter<double>("RecoAnalysisMatchingPt")),
							 m_kTauMassCut(iConfig.getParameter<double>("RecoAnalysisTauMassCut")),
							 m_kTauMass(iConfig.getParameter<double>("RecoAnalysisTauMass")),
							 m_kMuonMass(iConfig.getParameter<double>("RecoAnalysisMuonMass")),
                                                         m_kMuonSourceToken(iC.consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("MuonSourceTag"))),
                                                         m_kTrackSourceToken(iC.consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("TrackSourceTag")))
{
}

Tau3MuReco::~Tau3MuReco()
{

}

bool Tau3MuReco::doTau3MuReco(const edm::Event& iEvent, const edm::EventSetup& iSetup, reco::MuonCollection* muonCollection, reco::TrackCollection* trackCollection)
{
    m_MuonCollection = muonCollection;
    m_TrackCollection = trackCollection;

    edm::Handle<reco::MuonCollection> muons;
    edm::Handle<reco::TrackCollection> tracks;

    reco::MuonCollection::const_iterator muon;

    iEvent.getByToken(m_kMuonSourceToken, muons);
    iEvent.getByToken(m_kTrackSourceToken, tracks);

    for( muon = muons->begin(); muon != muons->end(); ++muon )
    {
	m_TrackCollection->push_back(*(muon->track().get()));
        m_MuonCollection->push_back(*muon);
    }

    if( m_TrackCollection->size() > 3 )
    {
        //find the right three ones coming from tau
        if(findCorrectPairing()==false)
        {  //maybe implement something like in ==3 (throw away ....)
	    LogDebug("Tau3MuReco") << "Could not find correct combination!" << std::endl;
	    return false;
        }

	return true;
    }

    if( m_TrackCollection->size() == 3 )
    {
        if(fabs(getInvariantMass(m_TrackCollection, m_kMuonMass)-m_kTauMass)<=m_kTauMassCut)
            return true;
        else//throw away one muon which don't match
            removeIncorrectMuon();

    }

    if( m_TrackCollection->size() == 2)
    {
        //search the third track

        //get 3rd muon canidate from tracks
        if(find3rdTrack(iEvent, iSetup, *(tracks.product()))==false)
        {
            LogDebug("Tau3MuReco") << "A 3rd Track can not be assigned!" << std::endl;
            return false;
        }
        else
            return true;
    }

    // cannot use this event, because less than 2 muons have been found

    LogDebug("Tau3MuReco") << "Not enough (" << m_TrackCollection->size() << ") muons found! Event skipped!" << std::endl;

    return false;
}

//private
bool Tau3MuReco::check4MuonTrack(const reco::Track& track)
{
    reco::TrackCollection::const_iterator iter;

    for(iter = m_TrackCollection->begin(); iter!=m_TrackCollection->end(); iter++)
    {
        //check if the track has the right charge
        //and check if dR is smaller than fMatchingDeltaR
        if((*iter).charge() == track.charge()
           && getDeltaR(*iter,track)<m_kMatchingDeltaR
           && fabs(((*iter).pt())-(track.pt()))<=(track.pt()*m_kMatchingPt))
        {
            LogDebug("Tau3MuReco") << "Found muon track in Tracks with DeltaR: " << getDeltaR(*iter,track)
				       << " Pt: " << track.pt()
				       << " Muons Pt is: " << (*iter).pt() << std::endl;
            return true;
        }
    }
    return false;
}


bool Tau3MuReco::find3rdTrack(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::TrackCollection& Tracks)
{
    //check size of TrackVector (should be two!)
    if(m_TrackCollection->size()!=2)
        return false;

    //more then two tracks should be in the event
    if(Tracks.size()<=2)
        return false;

    double SumDeltaR = 0;

    double MuonDeltaR = getDeltaR(m_TrackCollection->at(0),m_TrackCollection->at(1));

    //Loop overall tracks

    LogDebug("Tau3MuReco") << "Number of tracks found: " << Tracks.size() << std::endl;

    std::multimap<double, const reco::Track> TrackMultiMap;

    unsigned short muonCounter = 0;

    reco::TrackCollection::const_iterator track;

    for(track=Tracks.begin(); track!=Tracks.end(); track++)
    {
	if(check4MuonTrack(*track))
        {
            muonCounter++;
            continue;
        }

        SumDeltaR = MuonDeltaR;

        SumDeltaR += getDeltaR(m_TrackCollection->at(1), *track);
        SumDeltaR += getDeltaR(*track,m_TrackCollection->at(0));

        std::pair<double, const reco::Track> actTrack(SumDeltaR,*track);

        TrackMultiMap.insert(actTrack);
    }

    //two tracks should be clearly identified as muons by check4MuonTrack
    //else event is not useable
    if(muonCounter<2)
    {
        LogDebug("Tau3MuReco") << "Not enough muons (" << muonCounter << ") assigned to a track! Event skipped!" << std::endl;
        return false;
    }

    std::multimap<double, const reco::Track>::iterator it = TrackMultiMap.begin();

    if(it==TrackMultiMap.end())
    {
        LogDebug("Tau3MuReco") << "Not enough tracks (0) left! Event skipped!" << std::endl;
        return false;
    }

    //get 2mu+track with minimal DeltaR Sum (MultiMaps are sorted)
    m_TrackCollection->push_back((*it).second);

    //and check charge of this track
    //and check invariant mass of this combination
    //and make a vertex fit

    char Charge = m_TrackCollection->at(0).charge() * m_TrackCollection->at(1).charge();

    unsigned int  count = 0;

    //Charge > 0 means the two muons have same charge, so the third track has to have the opposit charge
    while((Charge > 0 && ((*it).second).charge()==(m_TrackCollection->at(0)).charge())
          || fabs(getInvariantMass(m_TrackCollection)-m_kTauMass) > m_kTauMassCut
        )
    {

        count++;

        LogDebug("Tau3MuReco") << "Track canidate: " << count << std::endl;

        if(Charge > 0 && ((*it).second).charge()!=(m_TrackCollection->at(0)).charge())
	LogDebug("Tau3MuReco") << "\tWrong charge!" << std::endl;
        LogDebug("Tau3MuReco") << "\tInvariant Mass deviation! " << fabs(getInvariantMass(m_TrackCollection)-m_kTauMass) << std::endl;
        LogDebug("Tau3MuReco") << "\tTrack Pt: " << (*it).second.pt() << std::endl;
        LogDebug("Tau3MuReco") << "\tDelta R: " << (*it).first << std::endl;
        LogDebug("Tau3MuReco") << "\tChi2: " << ((*it).second).normalizedChi2() << std::endl;

        ++it;

        //was not the best canidate
        m_TrackCollection->pop_back();

        if(it==TrackMultiMap.end())
            return false;

        //get next to minimal (Delta R Sum) track
        m_TrackCollection->push_back((*it).second);
    }

    LogDebug("Tau3MuReco") << "Found corresponding 3rd track: " << std::endl;
    LogDebug("Tau3MuReco") << "Track canidate: " << count << std::endl;
    LogDebug("Tau3MuReco") << "\tInvariant Mass deviation! " << fabs(getInvariantMass(m_TrackCollection)-m_kTauMass) << std::endl;
    LogDebug("Tau3MuReco") << "\tDelta R: " << (*it).first << std::endl;
    LogDebug("Tau3MuReco") << "\tNormChi2: " << ((*it).second).normalizedChi2() << std::endl;

    //choose this track, because it is the best canidate
    return true;
}

bool Tau3MuReco::findCorrectPairing()
{
    Combinatorics myCombinatorics(m_TrackCollection->size(), 3);

    std::vector < std::vector<UInt_t> > CombinationVec = myCombinatorics.GetCombinations();

    std::vector< std::vector<UInt_t> >::iterator it = CombinationVec.begin();

    char Charge = 0;

    reco::TrackCollection tempTrackCollection;
    reco::MuonCollection tempMuonCollection;

    do
    {
        if(it==CombinationVec.end())
            return false;

        Charge = 0;

        tempMuonCollection.clear();
	tempTrackCollection.clear();

        for(UInt_t i=0; i< (*it).size(); i++)
        {
            Charge += m_TrackCollection->at((*it).at(i)).charge();
            tempTrackCollection.push_back(m_TrackCollection->at((*it).at(i)));
	    tempMuonCollection.push_back(m_MuonCollection->at((*it).at(i)));
        }

        LogDebug("Tau3MuReco") << "Charge is: " << (int)Charge << " Have to be -1 or 1!!!" << std::endl;
        LogDebug("Tau3MuReco") << "Invariant mass is: " << fabs(getInvariantMass(&tempTrackCollection)-m_kTauMass) << " Have to be smaller than " << m_kTauMassCut << std::endl;

        it++;
    }
    while(abs(Charge)!=1 || fabs(getInvariantMass(&tempTrackCollection)-m_kTauMass)>m_kTauMassCut);

    *m_MuonCollection = tempMuonCollection;
    *m_TrackCollection = tempTrackCollection;

    return true;
}

bool Tau3MuReco::removeIncorrectMuon()
{

    double deltaR12 = getDeltaR(m_TrackCollection->at(0),m_TrackCollection->at(1));
    double deltaR23 = getDeltaR(m_TrackCollection->at(1),m_TrackCollection->at(2));
    double deltaR31 = getDeltaR(m_TrackCollection->at(2),m_TrackCollection->at(0));

    //if DeltaR12 is the smallest, than the 3rd one seems to be wrong
    //if DeltaR23 is the smallest, than the 2nd one seems to be wrong
    //if DeltaR31 is the smallest, than the 1st one seems to be wrong

    unsigned char temp;
    double junk;

    deltaR12 < deltaR23 ? temp=3 : temp=1;
    deltaR12 < deltaR23 ? junk=deltaR12 : junk=deltaR23;

    if(deltaR31 < junk)
        temp=2;

    m_TrackCollection->erase(m_TrackCollection->begin()+temp-1);

    return true;
}

double Tau3MuReco::getInvariantMass(const reco::TrackCollection* tracks, const double MuonMass)
{
    unsigned int numOfParticles = tracks->size();

    double SumPx = 0;
    double SumPy = 0;
    double SumPz = 0;

    double SumE = 0;

    for(unsigned int i=0; i<numOfParticles; i++)
    {
	SumPx += tracks->at(i).px();
	SumPy += tracks->at(i).py();
	SumPz += tracks->at(i).pz();

	SumE += sqrt(pow(tracks->at(i).p(),2)+pow(MuonMass,2));
    }

    double invmass = sqrt(pow(SumE,2)-pow(SumPx,2)-pow(SumPy,2)-pow(SumPz,2));

    return invmass;
}

double Tau3MuReco::getDeltaR(const reco::Track& track1, const reco::Track& track2)
{
    double dEta = track1.eta() - track2.eta();
    double dPhi = track1.phi() - track2.phi();

    while(dPhi >= TMath::Pi())       dPhi -= (2.0*TMath::Pi());
    while(dPhi < (-1.0*TMath::Pi())) dPhi += (2.0*TMath::Pi());

    return sqrt(pow(dEta,2)+pow(dPhi,2));
}
