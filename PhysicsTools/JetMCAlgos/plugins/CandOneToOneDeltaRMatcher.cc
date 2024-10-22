/* \class CandOneToOneDeltaRMatcher
 *
 * Producer for simple match map
 * to match two collections of candidate
 * with one-to-One matching
 * minimizing Sum(DeltaR)
 *
 */

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include <vector>
#include <iostream>

class CandOneToOneDeltaRMatcher : public edm::global::EDProducer<> {
public:
  explicit CandOneToOneDeltaRMatcher(const edm::ParameterSet&);

  using AllDist = std::vector<std::vector<float>>;

  enum class Algo { kBruteForce, kSwitchMode, kMixMode };

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  double length(const std::vector<int>&, const AllDist&) const;
  std::vector<int> AlgoBruteForce(int, int, const AllDist&) const;
  std::vector<int> AlgoSwitchMethod(int, int, AllDist&) const;
  static Algo algo(std::string const&);
  static const char* algoName(Algo);

  const edm::EDGetTokenT<reco::CandidateView> sourceToken_;
  const edm::EDGetTokenT<reco::CandidateView> matchedToken_;
  const Algo algoMethod_;
};

#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include <Math/VectorUtil.h>
#include <TMath.h>

using namespace edm;
using namespace std;
using namespace reco;
using namespace ROOT::Math::VectorUtil;
using namespace stdcomb;

CandOneToOneDeltaRMatcher::Algo CandOneToOneDeltaRMatcher::algo(const std::string& algoName) {
  if (algoName == "BruteForce") {
    return Algo::kBruteForce;
  } else if (algoName == "SwitchMode") {
    return Algo::kSwitchMode;
  } else if (algoName == "MixMode") {
    return Algo::kMixMode;
  } else {
    throw cms::Exception("OneToOne Constructor") << "wrong matching method in ParameterSet";
  }
}

const char* CandOneToOneDeltaRMatcher::algoName(Algo iAlgo) {
  switch (iAlgo) {
    case Algo::kBruteForce:
      return "BruteForce";
    case Algo::kSwitchMode:
      return "SwitchMode";
    case Algo::kMixMode:
      return "MixMode";
  }
  //can not get here
  return "";
}

CandOneToOneDeltaRMatcher::CandOneToOneDeltaRMatcher(const ParameterSet& cfg)
    : sourceToken_(consumes<CandidateView>(cfg.getParameter<InputTag>("src"))),
      matchedToken_(consumes<CandidateView>(cfg.getParameter<InputTag>("matched"))),
      algoMethod_(algo(cfg.getParameter<string>("algoMethod"))) {
  produces<CandViewMatchMap>("src2mtc");
  produces<CandViewMatchMap>("mtc2src");
}

void CandOneToOneDeltaRMatcher::produce(StreamID, Event& evt, const EventSetup& es) const {
  Handle<CandidateView> source;
  Handle<CandidateView> matched;
  evt.getByToken(sourceToken_, source);
  evt.getByToken(matchedToken_, matched);

  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "======== Source Collection =======";
  for (CandidateView::const_iterator c = source->begin(); c != source->end(); ++c) {
    edm::LogVerbatim("CandOneToOneDeltaRMatcher")
        << " pt source  " << c->pt() << " " << c->eta() << " " << c->phi() << endl;
  }
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "======== Matched Collection =======";
  for (CandidateView::const_iterator c = matched->begin(); c != matched->end(); ++c) {
    edm::LogVerbatim("CandOneToOneDeltaRMatcher")
        << " pt source  " << c->pt() << " " << c->eta() << " " << c->phi() << endl;
  }

  const int nSrc = source->size();
  const int nMtc = matched->size();

  const int nMin = min(source->size(), matched->size());
  const int nMax = max(source->size(), matched->size());
  if (nMin < 1)
    return;

  std::vector<std::vector<float>> allDist;

  if (nSrc <= nMtc) {
    allDist.reserve(source->size());
    for (CandidateView::const_iterator iSr = source->begin(); iSr != source->end(); iSr++) {
      vector<float> tempAllDist;
      tempAllDist.reserve(matched->size());
      for (CandidateView::const_iterator iMt = matched->begin(); iMt != matched->end(); iMt++) {
        tempAllDist.push_back(DeltaR(iSr->p4(), iMt->p4()));
      }
      allDist.emplace_back(std::move(tempAllDist));
    }
  } else {
    allDist.reserve(matched->size());
    for (CandidateView::const_iterator iMt = matched->begin(); iMt != matched->end(); iMt++) {
      vector<float> tempAllDist;
      tempAllDist.reserve(source->size());
      for (CandidateView::const_iterator iSr = source->begin(); iSr != source->end(); iSr++) {
        tempAllDist.push_back(DeltaR(iSr->p4(), iMt->p4()));
      }
      allDist.emplace_back(std::move(tempAllDist));
    }
  }

  /*
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "======== The DeltaR Matrix =======";
  for(int m0=0; m0<nMin; m0++) {
    //    for(int m1=0; m1<nMax; m1++) {
      edm::LogVerbatim("CandOneToOneDeltaRMatcher") << setprecision(2) << fixed << (m1 AllDist[m0][m1] ;
    //}
    edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "\n";
  }
  */

  // Loop size if Brute Force
  int nLoopToDo = (int)(TMath::Factorial(nMax) / TMath::Factorial(nMax - nMin));
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "nLoop:" << nLoopToDo << endl;
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "Choosen Algo is:" << algoName(algoMethod_);
  vector<int> bestCB;

  // Algo is Brute Force
  if (algoMethod_ == Algo::kBruteForce) {
    bestCB = AlgoBruteForce(nMin, nMax, allDist);

    // Algo is Switch Method
  } else if (algoMethod_ == Algo::kSwitchMode) {
    bestCB = AlgoSwitchMethod(nMin, nMax, allDist);

    // Algo is Brute Force if nLoop < 10000
  } else if (algoMethod_ == Algo::kMixMode) {
    if (nLoopToDo < 10000) {
      bestCB = AlgoBruteForce(nMin, nMax, allDist);
    } else {
      bestCB = AlgoSwitchMethod(nMin, nMax, allDist);
    }
  }

  for (int i1 = 0; i1 < nMin; i1++)
    edm::LogVerbatim("CandOneToOneDeltaRMatcher")
        << "min: " << i1 << " " << bestCB[i1] << " " << allDist[i1][bestCB[i1]];

  /*
  auto matchMapSrMt = std::make_unique<CandViewMatchMap>(CandViewMatchMap::ref_type( CandidateRefProd( source  ),
                                                                                             CandidateRefProd( matched ) ) );
  auto matchMapMtSr = std::make_unique<CandViewMatchMap>(CandViewMatchMap::ref_type( CandidateRefProd( matched ),
                                                                                             CandidateRefProd( source ) ) );
*/

  auto matchMapSrMt = std::make_unique<CandViewMatchMap>();
  auto matchMapMtSr = std::make_unique<CandViewMatchMap>();

  for (int c = 0; c != nMin; c++) {
    if (source->size() <= matched->size()) {
      matchMapSrMt->insert(source->refAt(c), matched->refAt(bestCB[c]));
      matchMapMtSr->insert(matched->refAt(bestCB[c]), source->refAt(c));
    } else {
      matchMapSrMt->insert(source->refAt(bestCB[c]), matched->refAt(c));
      matchMapMtSr->insert(matched->refAt(c), source->refAt(bestCB[c]));
    }
  }

  /*
  for( int c = 0; c != nMin; c ++ ) {
    if( source->size() <= matched->size() ) {
      matchMapSrMt->insert( CandidateRef( source,  c         ), CandidateRef( matched, bestCB[c] ) );
      matchMapMtSr->insert( CandidateRef( matched, bestCB[c] ), CandidateRef( source, c          ) );
    } else {
      matchMapSrMt->insert( CandidateRef( source,  bestCB[c] ), CandidateRef( matched, c         ) );
      matchMapMtSr->insert( CandidateRef( matched, c         ), CandidateRef( source,  bestCB[c] ) );
    }
  }
*/
  evt.put(std::move(matchMapSrMt), "src2mtc");
  evt.put(std::move(matchMapMtSr), "mtc2src");
}

double CandOneToOneDeltaRMatcher::length(const vector<int>& best, const AllDist& allDist) const {
  double myLength = 0;
  int row = 0;
  for (vector<int>::const_iterator it = best.begin(); it != best.end(); it++) {
    myLength += allDist[row][*it];
    row++;
  }
  return myLength;
}

// this is the Brute Force Algorithm
// All the possible combination are checked
// The best one is always found
// Be carefull when you have high values for nMin and nMax --> the combinatorial could explode!
// Sum(DeltaR) is minimized -->
// 0.1 - 0.2 - 1.0 - 1.5 is lower than
// 0.1 - 0.2 - 0.3 - 3.0
// Which one do you prefer? --> BruteForce select always the first

vector<int> CandOneToOneDeltaRMatcher::AlgoBruteForce(int nMin, int nMax, const AllDist& allDist) const {
  vector<int> ca;
  vector<int> cb;
  vector<int> bestCB;
  float totalDeltaR = 0;
  float BestTotalDeltaR = 1000;

  ca.reserve(nMax);
  for (int i1 = 0; i1 < nMax; i1++)
    ca.push_back(i1);
  cb.reserve(nMin);
  for (int i1 = 0; i1 < nMin; i1++)
    cb.push_back(i1);

  do {
    //do your processing on the new combination here
    for (int cnt = 0; cnt < TMath::Factorial(nMin); cnt++) {
      totalDeltaR = length(cb, allDist);
      if (totalDeltaR < BestTotalDeltaR) {
        BestTotalDeltaR = totalDeltaR;
        bestCB = cb;
      }
      next_permutation(cb.begin(), cb.end());
    }
  } while (next_combination(ca.begin(), ca.end(), cb.begin(), cb.end()));

  return bestCB;
}

// This method (Developed originally by Daniele Benedetti) check for the best combination
// choosing the minimum DeltaR for each line in AllDist matrix
// If no repeated row is found: ie (line,col)=(1,3) and (2,3) --> same as BruteForce
// If repetition --> set the higher DeltaR between  the 2 repetition to 1000 and re-check best combination
// Iterate until no repetition
// No guaranted minimum for Sum(DeltaR)
// If you have:
// 0.1 - 0.2 - 1.0 - 1.5 is lower than
// 0.1 - 0.2 - 0.3 - 3.0
// SwitchMethod normally select the second solution

vector<int> CandOneToOneDeltaRMatcher::AlgoSwitchMethod(int nMin, int nMax, AllDist& allDist) const {
  vector<int> bestCB;
  for (int i1 = 0; i1 < nMin; i1++) {
    int minInd = 0;
    for (int i2 = 1; i2 < nMax; i2++)
      if (allDist[i1][i2] < allDist[i1][minInd])
        minInd = i2;
    bestCB.push_back(minInd);
  }

  bool inside = true;
  while (inside) {
    inside = false;
    for (int i1 = 0; i1 < nMin; i1++) {
      for (int i2 = i1 + 1; i2 < nMin; i2++) {
        if (bestCB[i1] == bestCB[i2]) {
          inside = true;
          if (allDist[i1][(bestCB[i1])] <= allDist[i2][(bestCB[i2])]) {
            allDist[i2][(bestCB[i2])] = 1000;
            int minInd = 0;
            for (int i3 = 1; i3 < nMax; i3++)
              if (allDist[i2][i3] < allDist[i2][minInd])
                minInd = i3;
            bestCB[i2] = minInd;
          } else {
            allDist[i1][(bestCB[i1])] = 1000;
            int minInd = 0;
            for (int i3 = 1; i3 < nMax; i3++)
              if (allDist[i1][i3] < allDist[i1][minInd])
                minInd = i3;
            bestCB[i1] = minInd;
          }
        }  // End if
      }
    }
  }  // End while

  return bestCB;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CandOneToOneDeltaRMatcher);
