// Migrated to use DQMEDAnalyzer by: Jyothsna Rani Komaragiri, Oct 2014

#include "FWCore/Common/interface/TriggerNames.h"
#include "HLTriggerOffline/JetMET/interface/HLTJetMETValidation.h"
#include "Math/GenVector/VectorUtil.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
using namespace trigger;

HLTJetMETValidation::HLTJetMETValidation(const edm::ParameterSet &ps)
    : triggerEventObject_(
          consumes<TriggerEventWithRefs>(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject"))),
      PFJetAlgorithm(consumes<PFJetCollection>(ps.getUntrackedParameter<edm::InputTag>("PFJetAlgorithm"))),
      GenJetAlgorithm(consumes<GenJetCollection>(ps.getUntrackedParameter<edm::InputTag>("GenJetAlgorithm"))),
      CaloMETColl(consumes<CaloMETCollection>(ps.getUntrackedParameter<edm::InputTag>("CaloMETCollection"))),
      GenMETColl(consumes<GenMETCollection>(ps.getUntrackedParameter<edm::InputTag>("GenMETCollection"))),
      HLTriggerResults(consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("HLTriggerResults"))),
      triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder", "SingleJet")),
      patternJetTrg_(ps.getUntrackedParameter<std::string>("PatternJetTrg", "")),
      patternMetTrg_(ps.getUntrackedParameter<std::string>("PatternMetTrg", "")),
      patternMuTrg_(ps.getUntrackedParameter<std::string>("PatternMuTrg", "")),
      HLTinit_(false) {
  evtCnt = 0;
}

HLTJetMETValidation::~HLTJetMETValidation() {}

//
// member functions
//

// ------------ method called when starting to processes a run ------------
void HLTJetMETValidation::dqmBeginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  bool foundMuTrg = false;
  std::string trgMuNm;
  bool changedConfig = true;

  //--define search patterns
  TPRegexp patternJet(patternJetTrg_);
  TPRegexp patternMet(patternMetTrg_);
  TPRegexp patternMu(patternMuTrg_);

  if (!hltConfig_.init(iRun, iSetup, "HLT", changedConfig)) {
    edm::LogError("HLTJetMETValidation") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  std::vector<std::string> validTriggerNames = hltConfig_.triggerNames();
  for (size_t j = 0; j < validTriggerNames.size(); j++) {
    //---find the muon path
    if (TString(validTriggerNames[j]).Contains(patternMu)) {
      // std::cout <<validTriggerNames[j].c_str()<<std::endl;
      if (!foundMuTrg)
        trgMuNm = validTriggerNames[j];
      foundMuTrg = true;
    }
    //---find the jet paths
    if (TString(validTriggerNames[j]).Contains(patternJet)) {
      hltTrgJet.push_back(validTriggerNames[j]);
    }
    //---find the met paths
    if (TString(validTriggerNames[j]).Contains(patternMet)) {
      hltTrgMet.push_back(validTriggerNames[j]);
    }
  }

  //----set the denominator paths
  for (size_t it = 0; it < hltTrgJet.size(); it++) {
    if (it == 0 && foundMuTrg)
      hltTrgJetLow.push_back(trgMuNm);  //--lowest threshold uses muon path
    if (it == 0 && !foundMuTrg)
      hltTrgJetLow.push_back(hltTrgJet[it]);  //---if no muon then itself
    if (it != 0)
      hltTrgJetLow.push_back(hltTrgJet[it - 1]);
  }
  int itm(0), itpm(0), itmh(0), itpmh(0);
  for (size_t it = 0; it < hltTrgMet.size(); it++) {
    if (TString(hltTrgMet[it]).Contains("PF")) {
      if (TString(hltTrgMet[it]).Contains("MHT")) {
        if (0 == itpmh) {
          if (foundMuTrg)
            hltTrgMetLow.push_back(trgMuNm);
          else
            hltTrgMetLow.push_back(hltTrgMet[it]);
        } else
          hltTrgMetLow.push_back(hltTrgMet[it - 1]);
        itpmh++;
      }
      if (TString(hltTrgMet[it]).Contains("MET")) {
        if (0 == itpm) {
          if (foundMuTrg)
            hltTrgMetLow.push_back(trgMuNm);
          else
            hltTrgMetLow.push_back(hltTrgMet[it]);
        } else
          hltTrgMetLow.push_back(hltTrgMet[it - 1]);
        itpm++;
      }
    } else {
      if (TString(hltTrgMet[it]).Contains("MHT")) {
        if (0 == itmh) {
          if (foundMuTrg)
            hltTrgMetLow.push_back(trgMuNm);
          else
            hltTrgMetLow.push_back(hltTrgMet[it]);
        } else
          hltTrgMetLow.push_back(hltTrgMet[it - 1]);
        itmh++;
      }
      if (TString(hltTrgMet[it]).Contains("MET")) {
        if (0 == itm) {
          if (foundMuTrg)
            hltTrgMetLow.push_back(trgMuNm);
          else
            hltTrgMetLow.push_back(hltTrgMet[it]);
        } else
          hltTrgMetLow.push_back(hltTrgMet[it - 1]);
        itm++;
      }
    }
  }
}

// ------------ method called to book histograms before starting event loop
// ------------
void HLTJetMETValidation::bookHistograms(DQMStore::IBooker &iBooker,
                                         edm::Run const &iRun,
                                         edm::EventSetup const &iSetup) {
  //----define DQM folders and elements
  for (size_t it = 0; it < hltTrgJet.size(); it++) {
    std::string trgPathName = HLTConfigProvider::removeVersion(triggerTag_ + hltTrgJet[it]);
    iBooker.setCurrentFolder(trgPathName);
    _meHLTJetPt.push_back(iBooker.book1D("HLTJetPt", "Single HLT Jet Pt", 100, 0, 1000));
    _meHLTJetPtTrgMC.push_back(iBooker.book1D("HLTJetPtTrgMC", "Single HLT Jet Pt - HLT Triggered", 100, 0, 1000));
    _meHLTJetPtTrg.push_back(iBooker.book1D("HLTJetPtTrg", "Single HLT Jet Pt - HLT Triggered", 100, 0, 1000));
    _meHLTJetPtTrgLow.push_back(
        iBooker.book1D("HLTJetPtTrgLow", "Single HLT Jet Pt - HLT Triggered Low", 100, 0, 1000));

    _meHLTJetEta.push_back(iBooker.book1D("HLTJetEta", "Single HLT Jet Eta", 100, -10, 10));
    _meHLTJetEtaTrgMC.push_back(iBooker.book1D("HLTJetEtaTrgMC", "Single HLT Jet Eta - HLT Triggered", 100, -10, 10));
    _meHLTJetEtaTrg.push_back(iBooker.book1D("HLTJetEtaTrg", "Single HLT Jet Eta - HLT Triggered", 100, -10, 10));
    _meHLTJetEtaTrgLow.push_back(
        iBooker.book1D("HLTJetEtaTrgLow", "Single HLT Jet Eta - HLT Triggered Low", 100, -10, 10));

    _meHLTJetPhi.push_back(iBooker.book1D("HLTJetPhi", "Single HLT Jet Phi", 100, -4., 4.));
    _meHLTJetPhiTrgMC.push_back(iBooker.book1D("HLTJetPhiTrgMC", "Single HLT Jet Phi - HLT Triggered", 100, -4., 4.));
    _meHLTJetPhiTrg.push_back(iBooker.book1D("HLTJetPhiTrg", "Single HLT Jet Phi - HLT Triggered", 100, -4., 4.));
    _meHLTJetPhiTrgLow.push_back(
        iBooker.book1D("HLTJetPhiTrgLow", "Single HLT Jet Phi - HLT Triggered Low", 100, -4., 4.));

    _meGenJetPt.push_back(iBooker.book1D("GenJetPt", "Single Generated Jet Pt", 100, 0, 1000));
    _meGenJetPtTrgMC.push_back(
        iBooker.book1D("GenJetPtTrgMC", "Single Generated Jet Pt - HLT Triggered", 100, 0, 1000));
    _meGenJetPtTrg.push_back(iBooker.book1D("GenJetPtTrg", "Single Generated Jet Pt - HLT Triggered", 100, 0, 1000));
    _meGenJetPtTrgLow.push_back(
        iBooker.book1D("GenJetPtTrgLow", "Single Generated Jet Pt - HLT Triggered Low", 100, 0, 1000));

    _meGenJetEta.push_back(iBooker.book1D("GenJetEta", "Single Generated Jet Eta", 100, -10, 10));
    _meGenJetEtaTrgMC.push_back(
        iBooker.book1D("GenJetEtaTrgMC", "Single Generated Jet Eta - HLT Triggered", 100, -10, 10));
    _meGenJetEtaTrg.push_back(iBooker.book1D("GenJetEtaTrg", "Single Generated Jet Eta - HLT Triggered", 100, -10, 10));
    _meGenJetEtaTrgLow.push_back(
        iBooker.book1D("GenJetEtaTrgLow", "Single Generated Jet Eta - HLT Triggered Low", 100, -10, 10));

    _meGenJetPhi.push_back(iBooker.book1D("GenJetPhi", "Single Generated Jet Phi", 100, -4., 4.));
    _meGenJetPhiTrgMC.push_back(
        iBooker.book1D("GenJetPhiTrgMC", "Single Generated Jet Phi - HLT Triggered", 100, -4., 4.));
    _meGenJetPhiTrg.push_back(iBooker.book1D("GenJetPhiTrg", "Single Generated Jet Phi - HLT Triggered", 100, -4., 4.));
    _meGenJetPhiTrgLow.push_back(
        iBooker.book1D("GenJetPhiTrgLow", "Single Generated Jet Phi - HLT Triggered Low", 100, -4., 4.));
  }
  for (size_t it = 0; it < hltTrgMet.size(); it++) {
    // std::cout<<hltTrgMet[it].c_str()<<"
    // "<<hltTrgMetLow[it].c_str()<<std::endl;
    std::string trgPathName = HLTConfigProvider::removeVersion(triggerTag_ + hltTrgMet[it]);
    iBooker.setCurrentFolder(trgPathName);
    _meHLTMET.push_back(iBooker.book1D("HLTMET", "HLT Missing ET", 100, 0, 1000));
    _meHLTMETTrgMC.push_back(iBooker.book1D("HLTMETTrgMC", "HLT Missing ET - HLT Triggered", 100, 0, 1000));
    _meHLTMETTrg.push_back(iBooker.book1D("HLTMETTrg", "HLT Missing ET - HLT Triggered", 100, 0, 1000));
    _meHLTMETTrgLow.push_back(iBooker.book1D("HLTMETTrgLow", "HLT Missing ET - HLT Triggered Low", 100, 0, 1000));

    _meGenMET.push_back(iBooker.book1D("GenMET", "Generated Missing ET", 100, 0, 1000));
    _meGenMETTrgMC.push_back(iBooker.book1D("GenMETTrgMC", "Generated Missing ET - HLT Triggered", 100, 0, 1000));
    _meGenMETTrg.push_back(iBooker.book1D("GenMETTrg", "Generated Missing ET - HLT Triggered", 100, 0, 1000));
    _meGenMETTrgLow.push_back(iBooker.book1D("GenMETTrgLow", "Generated Missing ET - HLT Triggered Low", 100, 0, 1000));
  }
}

// ------------ method called for each event ------------
void HLTJetMETValidation::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;

  evtCnt++;

  // get The triggerEvent
  Handle<TriggerEventWithRefs> trigEv;
  iEvent.getByToken(triggerEventObject_, trigEv);

  // get TriggerResults object
  bool gotHLT = true;
  std::vector<bool> myTrigJ;
  myTrigJ.reserve(hltTrgJet.size());
  for (size_t it = 0; it < hltTrgJet.size(); it++)
    myTrigJ.push_back(false);
  std::vector<bool> myTrigJLow;
  myTrigJLow.reserve(hltTrgJetLow.size());
  for (size_t it = 0; it < hltTrgJetLow.size(); it++)
    myTrigJLow.push_back(false);
  std::vector<bool> myTrigM;
  myTrigM.reserve(hltTrgMet.size());
  for (size_t it = 0; it < hltTrgMet.size(); it++)
    myTrigM.push_back(false);
  std::vector<bool> myTrigMLow;
  myTrigMLow.reserve(hltTrgMetLow.size());
  for (size_t it = 0; it < hltTrgMetLow.size(); it++)
    myTrigMLow.push_back(false);

  Handle<TriggerResults> hltresults;
  iEvent.getByToken(HLTriggerResults, hltresults);
  if (!hltresults.isValid()) {
    gotHLT = false;
  }

  if (gotHLT) {
    const edm::TriggerNames &triggerNames = iEvent.triggerNames(*hltresults);
    getHLTResults(*hltresults, triggerNames);

    //---pick-up the jet trigger decisions
    for (size_t it = 0; it < hltTrgJet.size(); it++) {
      trig_iter = hltTriggerMap.find(hltTrgJet[it]);
      if (trig_iter != hltTriggerMap.end()) {
        myTrigJ[it] = trig_iter->second;
      }
    }
    for (size_t it = 0; it < hltTrgJetLow.size(); it++) {
      trig_iter = hltTriggerMap.find(hltTrgJetLow[it]);
      if (trig_iter != hltTriggerMap.end()) {
        myTrigJLow[it] = trig_iter->second;
      }
    }

    //---pick-up the met trigger decisions
    for (size_t it = 0; it < hltTrgMet.size(); it++) {
      trig_iter = hltTriggerMap.find(hltTrgMet[it]);
      if (trig_iter != hltTriggerMap.end()) {
        myTrigM[it] = trig_iter->second;
      }
    }
    for (size_t it = 0; it < hltTrgMetLow.size(); it++) {
      trig_iter = hltTriggerMap.find(hltTrgMetLow[it]);
      if (trig_iter != hltTriggerMap.end()) {
        myTrigMLow[it] = trig_iter->second;
      }
    }
  }

  // --- Fill histos for PFJet paths ---
  // HLT jets namely hltAK4PFJets
  Handle<PFJetCollection> pfJets;
  iEvent.getByToken(PFJetAlgorithm, pfJets);
  double pfJetPt = -1.;
  double pfJetEta = -999.;
  double pfJetPhi = -999.;

  if (pfJets.isValid()) {
    // Loop over the PFJets and fill some histograms
    int jetInd = 0;
    for (PFJetCollection::const_iterator pf = pfJets->begin(); pf != pfJets->end(); ++pf) {
      // std::cout << "PF JET #" << jetInd << std::endl << pf->print() <<
      // std::endl;
      if (jetInd == 0) {
        pfJetPt = pf->pt();
        pfJetEta = pf->eta();
        pfJetPhi = pf->phi();
        for (size_t it = 0; it < hltTrgJet.size(); it++) {
          _meHLTJetPt[it]->Fill(pfJetPt);
          _meHLTJetEta[it]->Fill(pfJetEta);
          _meHLTJetPhi[it]->Fill(pfJetPhi);
          if (myTrigJ[it])
            _meHLTJetPtTrgMC[it]->Fill(pfJetPt);
          if (myTrigJ[it])
            _meHLTJetEtaTrgMC[it]->Fill(pfJetEta);
          if (myTrigJ[it])
            _meHLTJetPhiTrgMC[it]->Fill(pfJetPhi);
          if (myTrigJ[it] && myTrigJLow[it])
            _meHLTJetPtTrg[it]->Fill(pfJetPt);
          if (myTrigJ[it] && myTrigJLow[it])
            _meHLTJetEtaTrg[it]->Fill(pfJetEta);
          if (myTrigJ[it] && myTrigJLow[it])
            _meHLTJetPhiTrg[it]->Fill(pfJetPhi);
          if (myTrigJLow[it])
            _meHLTJetPtTrgLow[it]->Fill(pfJetPt);
          if (myTrigJLow[it])
            _meHLTJetEtaTrgLow[it]->Fill(pfJetEta);
          if (myTrigJLow[it])
            _meHLTJetPhiTrgLow[it]->Fill(pfJetPhi);
        }
        jetInd++;
      }
    }  // loop over pfjets
  }

  // GenJets
  Handle<GenJetCollection> genJets;
  iEvent.getByToken(GenJetAlgorithm, genJets);
  double genJetPt = -1.;
  double genJetEta = -999.;
  double genJetPhi = -999.;

  if (genJets.isValid()) {
    // Loop over the GenJets and fill some histograms
    int jetInd = 0;
    for (GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end(); ++gen) {
      if (jetInd == 0) {
        genJetPt = gen->pt();
        genJetEta = gen->eta();
        genJetPhi = gen->phi();
        for (size_t it = 0; it < hltTrgJet.size(); it++) {
          _meGenJetPt[it]->Fill(genJetPt);
          _meGenJetEta[it]->Fill(genJetEta);
          _meGenJetPhi[it]->Fill(genJetPhi);
          if (myTrigJ[it])
            _meGenJetPtTrgMC[it]->Fill(genJetPt);
          if (myTrigJ[it])
            _meGenJetEtaTrgMC[it]->Fill(genJetEta);
          if (myTrigJ[it])
            _meGenJetPhiTrgMC[it]->Fill(genJetPhi);
          if (myTrigJ[it] && myTrigJLow[it])
            _meGenJetPtTrg[it]->Fill(genJetPt);
          if (myTrigJ[it] && myTrigJLow[it])
            _meGenJetEtaTrg[it]->Fill(genJetEta);
          if (myTrigJ[it] && myTrigJLow[it])
            _meGenJetPhiTrg[it]->Fill(genJetPhi);
          if (myTrigJLow[it])
            _meGenJetPtTrgLow[it]->Fill(genJetPt);
          if (myTrigJLow[it])
            _meGenJetEtaTrgLow[it]->Fill(genJetEta);
          if (myTrigJLow[it])
            _meGenJetPhiTrgLow[it]->Fill(genJetPhi);
        }
        jetInd++;
      }
    }
  }

  // --- Fill histos for PFMET paths ---
  // HLT MET namely hltmet
  edm::Handle<CaloMETCollection> recmet;
  iEvent.getByToken(CaloMETColl, recmet);

  double calMet = -1;
  if (recmet.isValid()) {
    typedef CaloMETCollection::const_iterator cmiter;
    // std::cout << "Size of MET collection" <<  recmet.size() << std::endl;
    for (cmiter i = recmet->begin(); i != recmet->end(); i++) {
      calMet = i->pt();
      for (size_t it = 0; it < hltTrgMet.size(); it++) {
        _meHLTMET[it]->Fill(calMet);
        if (myTrigM.size() > it && myTrigM[it])
          _meHLTMETTrgMC[it]->Fill(calMet);
        if (myTrigM.size() > it && myTrigMLow.size() > it && myTrigM[it] && myTrigMLow[it])
          _meHLTMETTrg[it]->Fill(calMet);
        if (myTrigMLow.size() > it && myTrigMLow[it])
          _meHLTMETTrgLow[it]->Fill(calMet);
      }
    }
  }

  edm::Handle<GenMETCollection> genmet;
  iEvent.getByToken(GenMETColl, genmet);

  double genMet = -1;
  if (genmet.isValid()) {
    typedef GenMETCollection::const_iterator cmiter;
    for (cmiter i = genmet->begin(); i != genmet->end(); i++) {
      genMet = i->pt();
      for (size_t it = 0; it < hltTrgMet.size(); it++) {
        _meGenMET[it]->Fill(genMet);
        if (myTrigM.size() > it && myTrigM[it])
          _meGenMETTrgMC[it]->Fill(genMet);
        if (myTrigM.size() > it && myTrigMLow.size() > it && myTrigM[it] && myTrigMLow[it])
          _meGenMETTrg[it]->Fill(genMet);
        if (myTrigMLow.size() > it && myTrigMLow[it])
          _meGenMETTrgLow[it]->Fill(genMet);
      }
    }
  }
}

void HLTJetMETValidation::getHLTResults(const edm::TriggerResults &hltresults, const edm::TriggerNames &triggerNames) {
  int ntrigs = hltresults.size();
  if (!HLTinit_) {
    HLTinit_ = true;
  }

  for (int itrig = 0; itrig != ntrigs; ++itrig) {
    std::string trigName = triggerNames.triggerName(itrig);
    bool accept = hltresults.accept(itrig);

    // fill the trigger map
    typedef std::map<std::string, bool>::value_type valType;
    trig_iter = hltTriggerMap.find(trigName);
    if (trig_iter == hltTriggerMap.end())
      hltTriggerMap.insert(valType(trigName, accept));
    else
      trig_iter->second = accept;
  }
}
