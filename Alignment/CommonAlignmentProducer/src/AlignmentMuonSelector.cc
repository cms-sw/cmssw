#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentProducer/interface/AlignmentMuonSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TLorentzVector.h"

// constructor ----------------------------------------------------------------

AlignmentMuonSelector::AlignmentMuonSelector(const edm::ParameterSet& cfg)
    : applyBasicCuts(cfg.getParameter<bool>("applyBasicCuts")),
      applyNHighestPt(cfg.getParameter<bool>("applyNHighestPt")),
      applyMultiplicityFilter(cfg.getParameter<bool>("applyMultiplicityFilter")),
      applyMassPairFilter(cfg.getParameter<bool>("applyMassPairFilter")),
      nHighestPt(cfg.getParameter<int>("nHighestPt")),
      minMultiplicity(cfg.getParameter<int>("minMultiplicity")),
      pMin(cfg.getParameter<double>("pMin")),
      pMax(cfg.getParameter<double>("pMax")),
      ptMin(cfg.getParameter<double>("ptMin")),
      ptMax(cfg.getParameter<double>("ptMax")),
      etaMin(cfg.getParameter<double>("etaMin")),
      etaMax(cfg.getParameter<double>("etaMax")),
      phiMin(cfg.getParameter<double>("phiMin")),
      phiMax(cfg.getParameter<double>("phiMax")),
      nHitMinSA(cfg.getParameter<double>("nHitMinSA")),
      nHitMaxSA(cfg.getParameter<double>("nHitMaxSA")),
      chi2nMaxSA(cfg.getParameter<double>("chi2nMaxSA")),
      nHitMinGB(cfg.getParameter<double>("nHitMinGB")),
      nHitMaxGB(cfg.getParameter<double>("nHitMaxGB")),
      chi2nMaxGB(cfg.getParameter<double>("chi2nMaxGB")),
      nHitMinTO(cfg.getParameter<double>("nHitMinTO")),
      nHitMaxTO(cfg.getParameter<double>("nHitMaxTO")),
      chi2nMaxTO(cfg.getParameter<double>("chi2nMaxTO")),
      minMassPair(cfg.getParameter<double>("minMassPair")),
      maxMassPair(cfg.getParameter<double>("maxMassPair")) {
  if (applyBasicCuts)
    edm::LogInfo("AlignmentMuonSelector")
        << "applying basic muon cuts ..."
        << "\npmin,pmax:           " << pMin << "," << pMax << "\nptmin,ptmax:         " << ptMin << "," << ptMax
        << "\netamin,etamax:       " << etaMin << "," << etaMax << "\nphimin,phimax:       " << phiMin << "," << phiMax
        << "\nnhitminSA,nhitmaxSA: " << nHitMinSA << "," << nHitMaxSA << "\nchi2nmaxSA:          " << chi2nMaxSA << ","
        << "\nnhitminGB,nhitmaxGB: " << nHitMinGB << "," << nHitMaxGB << "\nchi2nmaxGB:          " << chi2nMaxGB << ","
        << "\nnhitminTO,nhitmaxTO: " << nHitMinTO << "," << nHitMaxTO << "\nchi2nmaxTO:          " << chi2nMaxTO;

  if (applyNHighestPt)
    edm::LogInfo("AlignmentMuonSelector") << "filter N muons with highest Pt N=" << nHighestPt;

  if (applyMultiplicityFilter)
    edm::LogInfo("AlignmentMuonSelector") << "apply multiplicity filter N>=" << minMultiplicity;

  if (applyMassPairFilter)
    edm::LogInfo("AlignmentMuonSelector")
        << "apply Mass Pair filter minMassPair=" << minMassPair << " maxMassPair=" << maxMassPair;
}

// destructor -----------------------------------------------------------------

AlignmentMuonSelector::~AlignmentMuonSelector() {}

// do selection ---------------------------------------------------------------

AlignmentMuonSelector::Muons AlignmentMuonSelector::select(const Muons& muons, const edm::Event& evt) const {
  Muons result = muons;

  // apply basic muon cuts (if selected)
  if (applyBasicCuts)
    result = this->basicCuts(result);

  // filter N muons with highest Pt (if selected)
  if (applyNHighestPt)
    result = this->theNHighestPtMuons(result);

  // apply minimum multiplicity requirement (if selected)
  if (applyMultiplicityFilter) {
    if (result.size() < (unsigned int)minMultiplicity)
      result.clear();
  }

  // apply mass pair requirement (if selected)
  if (applyMassPairFilter) {
    if (result.size() < 2)
      result.clear();  // at least 2 muons are require for a mass pair...
    else
      result = this->theBestMassPairCombinationMuons(result);
  }

  edm::LogInfo("AlignmentMuonSelector") << "muons all,kept: " << muons.size() << "," << result.size();

  return result;
}

// make basic cuts ------------------------------------------------------------

AlignmentMuonSelector::Muons AlignmentMuonSelector::basicCuts(const Muons& muons) const {
  Muons result;

  for (Muons::const_iterator it = muons.begin(); it != muons.end(); ++it) {
    const reco::Muon* muonp = *it;
    float p = muonp->p();
    float pt = muonp->pt();
    float eta = muonp->eta();
    float phi = muonp->phi();

    int nhitSA = 0;
    float chi2nSA = 9999.;
    if (muonp->isStandAloneMuon()) {
      nhitSA = muonp->standAloneMuon()->numberOfValidHits();  // standAlone Muon
      chi2nSA = muonp->standAloneMuon()->normalizedChi2();    // standAlone Muon
    }
    int nhitGB = 0;
    float chi2nGB = 9999.;
    if (muonp->isGlobalMuon()) {
      nhitGB = muonp->combinedMuon()->numberOfValidHits();  // global Muon
      chi2nGB = muonp->combinedMuon()->normalizedChi2();    // global Muon
    }
    int nhitTO = 0;
    float chi2nTO = 9999.;
    if (muonp->isTrackerMuon()) {
      nhitTO = muonp->track()->numberOfValidHits();  // Tracker Only
      chi2nTO = muonp->track()->normalizedChi2();    // Tracker Only
    }
    edm::LogInfo("AlignmentMuonSelector")
        << " pt,eta,phi,nhitSA,chi2nSA,nhitGB,chi2nGB,nhitTO,chi2nTO: " << pt << "," << eta << "," << phi << ","
        << nhitSA << "," << chi2nSA << "," << nhitGB << "," << chi2nGB << "," << nhitTO << "," << chi2nTO;

    if (p > pMin && p < pMax && pt > ptMin && pt < ptMax && eta > etaMin && eta < etaMax && phi > phiMin &&
        phi < phiMax && nhitSA >= nHitMinSA && nhitSA <= nHitMaxSA && chi2nSA < chi2nMaxSA && nhitGB >= nHitMinGB &&
        nhitGB <= nHitMaxGB && chi2nGB < chi2nMaxGB && nhitTO >= nHitMinTO && nhitTO <= nHitMaxTO &&
        chi2nTO < chi2nMaxTO) {
      result.push_back(muonp);
    }
  }

  return result;
}

//-----------------------------------------------------------------------------

AlignmentMuonSelector::Muons AlignmentMuonSelector::theNHighestPtMuons(const Muons& muons) const {
  Muons sortedMuons = muons;
  Muons result;

  // sort in pt
  std::sort(sortedMuons.begin(), sortedMuons.end(), ptComparator);

  // copy theMuonMult highest pt muons to result vector
  int n = 0;
  for (Muons::const_iterator it = sortedMuons.begin(); it != sortedMuons.end(); ++it) {
    if (n < nHighestPt) {
      result.push_back(*it);
      n++;
    }
  }

  return result;
}

//-----------------------------------------------------------------------------

AlignmentMuonSelector::Muons AlignmentMuonSelector::theBestMassPairCombinationMuons(const Muons& muons) const {
  Muons sortedMuons = muons;
  Muons result;
  TLorentzVector mu1, mu2, pair;
  double mass = 0, minDiff = 999999.;

  // sort in pt
  std::sort(sortedMuons.begin(), sortedMuons.end(), ptComparator);

  // copy best mass pair combination muons to result vector
  // Criteria:
  // a) maxMassPair !=    minMassPair: the two highest pt muons with mass pair inside the given mass window
  // b) maxMassPair ==    minMassPair: the muon pair with massPair closest to given mass value
  for (Muons::const_iterator it1 = sortedMuons.begin(); it1 != sortedMuons.end(); ++it1) {
    for (Muons::const_iterator it2 = it1 + 1; it2 != sortedMuons.end(); ++it2) {
      mu1 = TLorentzVector((*it1)->momentum().x(), (*it1)->momentum().y(), (*it1)->momentum().z(), (*it1)->p());
      mu2 = TLorentzVector((*it2)->momentum().x(), (*it2)->momentum().y(), (*it2)->momentum().z(), (*it2)->p());
      pair = mu1 + mu2;
      mass = pair.M();

      if (maxMassPair != minMassPair) {
        if (mass < maxMassPair && mass > minMassPair) {
          result.push_back(*it1);
          result.push_back(*it2);
          break;
        }
      } else {
        if (fabs(mass - maxMassPair) < minDiff) {
          minDiff = fabs(mass - maxMassPair);
          result.clear();
          result.push_back(*it1);
          result.push_back(*it2);
        }
      }
    }
  }

  return result;
}
