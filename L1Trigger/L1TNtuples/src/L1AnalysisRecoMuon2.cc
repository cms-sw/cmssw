#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMuon2.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include <DataFormats/PatCandidates/interface/Muon.h>
#include "L1Trigger/L1TNtuples/interface/MuonID.h"
#include <TLorentzVector.h>

using namespace std;
using namespace muon;

L1Analysis::L1AnalysisRecoMuon2::L1AnalysisRecoMuon2(const edm::ParameterSet& pset) :
  muPropagator1st_(pset.getParameter<edm::ParameterSet>("muProp1st")),
  muPropagator2nd_(pset.getParameter<edm::ParameterSet>("muProp2nd"))
{
}


L1Analysis::L1AnalysisRecoMuon2::~L1AnalysisRecoMuon2()
{
}

void L1Analysis::L1AnalysisRecoMuon2::SetMuon(const edm::Event& event,
					      const edm::EventSetup& setup,
					      edm::Handle<reco::MuonCollection> muons,
					      edm::Handle<reco::VertexCollection> vertices, 
					      double METx, double METy,
                                              unsigned maxMuon)
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
    recoMuon_.charge.push_back(it->charge());

    //check isLooseMuon
    bool flagLoose = isLooseMuonCustom(*it);
    recoMuon_.isLooseMuon.push_back(flagLoose);

    //check isMediumMuon
     bool flagMedium = isMediumMuonCustom(*it);
    recoMuon_.isMediumMuon.push_back(flagMedium);

    //check isTightMuon
    bool flagTight = false;
    if (vertices.isValid())
      flagTight = isTightMuonCustom(*it, (*vertices)[0]);
    recoMuon_.isTightMuon.push_back(flagTight);

    double iso = (it->pfIsolationR04().sumChargedHadronPt + max(0.,
           it->pfIsolationR04().sumNeutralHadronEt +
           it->pfIsolationR04().sumPhotonEt -
           0.5*it->pfIsolationR04().sumPUPt)) / it->pt();
    recoMuon_.iso.push_back(iso);

   double MET_local = TMath::Sqrt (METx*METx + METy*METy);
    recoMuon_.met.push_back(MET_local);

    TLorentzVector METP4;
    METP4.SetPxPyPzE(METx, METy, 0, MET_local);

    TLorentzVector Muon;
    Muon.SetPtEtaPhiE(it->pt(),it->eta(),it->phi(),it->energy());
    
    double scalSum = MET_local + Muon.Pt();
    TLorentzVector vecSum (Muon);
    vecSum += METP4;
    double vecSumPt = vecSum.Pt();
    
    recoMuon_.mt.push_back(TMath::Sqrt (scalSum*scalSum - vecSumPt*vecSumPt));

    recoMuon_.nMuons++;

    // extrapolation of track coordinates
    TrajectoryStateOnSurface stateAtMuSt1 = muPropagator1st_.extrapolate(*it);
    if (stateAtMuSt1.isValid()) {
      recoMuon_.etaSt1.push_back(stateAtMuSt1.globalPosition().eta());
      recoMuon_.phiSt1.push_back(stateAtMuSt1.globalPosition().phi());
    } else {
      recoMuon_.etaSt1.push_back(-9999);
      recoMuon_.phiSt1.push_back(-9999);
    }

    TrajectoryStateOnSurface stateAtMuSt2 = muPropagator2nd_.extrapolate(*it);
    if (stateAtMuSt2.isValid()) {
      recoMuon_.etaSt2.push_back(stateAtMuSt2.globalPosition().eta());
      recoMuon_.phiSt2.push_back(stateAtMuSt2.globalPosition().phi());
    } else {
      recoMuon_.etaSt2.push_back(-9999);
      recoMuon_.phiSt2.push_back(-9999);
    }
  }
}

void L1Analysis::L1AnalysisRecoMuon2::init(const edm::EventSetup &eventSetup)
{
  muPropagator1st_.init(eventSetup);
  muPropagator2nd_.init(eventSetup);
}
