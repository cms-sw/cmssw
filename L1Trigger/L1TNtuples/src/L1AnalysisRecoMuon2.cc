#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMuon2.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include <DataFormats/PatCandidates/interface/Muon.h>
#include "L1Trigger/L1TNtuples/interface/MuonID.h"

using namespace std;
using namespace muon;

L1Analysis::L1AnalysisRecoMuon2::L1AnalysisRecoMuon2()
{
}


L1Analysis::L1AnalysisRecoMuon2::~L1AnalysisRecoMuon2()
{
}

void L1Analysis::L1AnalysisRecoMuon2::SetMuon(const edm::Event& event,
					   const edm::EventSetup& setup,
					   edm::Handle<reco::MuonCollection> muons, unsigned maxMuon)
{

  recoMuon_.nMuons=0;

  for(reco::MuonCollection::const_iterator it=muons->begin();
      it!=muons->end() && recoMuon_.nMuons < maxMuon;
      ++it) {

    recoMuon_.e.push_back(it->energy());    
    recoMuon_.pt.push_back(it->pt());    
    recoMuon_.et.push_back(it->et());    
    recoMuon_.eta.push_back(it->eta());
    recoMuon_.phi.push_back(it->phi());

    //check isLooseMuon
    bool flagLoose = isLooseMuonCustom(*it);
    recoMuon_.isLooseMuon.push_back(flagLoose);

    //check isMediumMuon
    bool flagMedium = isMediumMuonCustom(*it);
    recoMuon_.isMediumMuon.push_back(flagMedium);

    double iso = (it->pfIsolationR03().sumChargedHadronPt + max(
           it->pfIsolationR03().sumNeutralHadronEt +
           it->pfIsolationR03().sumPhotonEt - 
           0.5 * it->pfIsolationR03().sumPUPt, 0.0)) / it->pt();
    recoMuon_.iso.push_back(iso);

    recoMuon_.nMuons++;

  }
}


