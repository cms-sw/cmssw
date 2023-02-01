// Implementation of the new JetMatching plugin
// author: Carlos Vico (U. Oviedo)
// taken from: https://amcatnlo.web.cern.ch/amcatnlo/JetMatching.h

#include "JetMatchingEWKFxFx.h"
using namespace Pythia8;

JetMatchingEWKFxFx::JetMatchingEWKFxFx(const edm::ParameterSet& iConfig) {}

bool JetMatchingEWKFxFx::initAfterBeams() {
  // This method is automatically called by Pythia8::init
  // and it can be used to change any of the parameter given
  // in the fragment.

  // Initialise values for stored jet matching veto inputs.
  pTfirstSave = -1.;
  processSubsetSave.init("(eventProcess)", particleDataPtr);
  workEventJetSave.init("(workEventJet)", particleDataPtr);

  // Read Madgraph specific configuration variables
  bool setMad = settingsPtr->flag("JetMatching:setMad");

  // Parse in MadgraphPar object
  MadgraphPar par;

  string parStr = infoPtr->header("MGRunCard");
  if (!parStr.empty()) {
    par.parse(parStr);
    par.printParams();
  }

  // Set Madgraph merging parameters from the file if requested
  if (setMad) {
    if (par.haveParam("xqcut") && par.haveParam("maxjetflavor") && par.haveParam("alpsfact") &&
        par.haveParam("ickkw")) {
      settingsPtr->flag("JetMatching:merge", par.getParam("ickkw"));
      settingsPtr->parm("JetMatching:qCut", par.getParam("xqcut"));
      settingsPtr->mode("JetMatching:nQmatch", par.getParamAsInt("maxjetflavor"));
      settingsPtr->parm("JetMatching:clFact", clFact = par.getParam("alpsfact"));
      if (par.getParamAsInt("ickkw") == 0)
        errorMsg(
            "Error in JetMatchingMadgraph:init: "
            "Madgraph file parameters are not set for merging");

      // Warn if setMad requested, but one or more parameters not present
    } else {
      errorMsg(
          "Warning in JetMatchingMadgraph:init: "
          "Madgraph merging parameters not found");
      if (!par.haveParam("xqcut"))
        errorMsg(
            "Warning in "
            "JetMatchingMadgraph:init: No xqcut");
      if (!par.haveParam("ickkw"))
        errorMsg(
            "Warning in "
            "JetMatchingMadgraph:init: No ickkw");
      if (!par.haveParam("maxjetflavor"))
        errorMsg(
            "Warning in "
            "JetMatchingMadgraph:init: No maxjetflavor");
      if (!par.haveParam("alpsfact"))
        errorMsg(
            "Warning in "
            "JetMatchingMadgraph:init: No alpsfact");
    }
  }

  // Read in FxFx matching parameters
  doFxFx = settingsPtr->flag("JetMatching:doFxFx");
  nPartonsNow = settingsPtr->mode("JetMatching:nPartonsNow");
  qCutME = settingsPtr->parm("JetMatching:qCutME");
  qCutMESq = pow(qCutME, 2);

  // Read in Madgraph merging parameters
  doMerge = settingsPtr->flag("JetMatching:merge");
  doShowerKt = settingsPtr->flag("JetMatching:doShowerKt");
  qCut = settingsPtr->parm("JetMatching:qCut");
  nQmatch = settingsPtr->mode("JetMatching:nQmatch");
  clFact = settingsPtr->parm("JetMatching:clFact");

  // Read in jet algorithm parameters
  jetAlgorithm = settingsPtr->mode("JetMatching:jetAlgorithm");
  nJetMax = settingsPtr->mode("JetMatching:nJetMax");
  eTjetMin = settingsPtr->parm("JetMatching:eTjetMin");
  coneRadius = settingsPtr->parm("JetMatching:coneRadius");
  etaJetMax = settingsPtr->parm("JetMatching:etaJetMax");
  slowJetPower = settingsPtr->mode("JetMatching:slowJetPower");

  // Matching procedure
  jetAllow = settingsPtr->mode("JetMatching:jetAllow");
  exclusiveMode = settingsPtr->mode("JetMatching:exclusive");
  qCutSq = pow(qCut, 2);
  etaJetMaxAlgo = etaJetMax;
  //performVeto    = true;

  // If not merging is applied, then we are done
  if (!doMerge)
    return true;

  // Exclusive mode; if set to 2, then set based on nJet/nJetMax
  if (exclusiveMode == 2) {
    // No nJet or nJetMax, so default to exclusive mode
    if (nJetMax < 0) {
      errorMsg(
          "Warning in JetMatchingMadgraph:init: "
          "missing jet multiplicity information; running in exclusive mode");
      exclusiveMode = 1;
    }
  }

  // Initialise chosen jet algorithm.
  // Currently, this only supports the kT-algorithm in SlowJet.
  // Use the QCD distance measure by default.
  jetAlgorithm = 2;

  slowJet = new SlowJet(slowJetPower, coneRadius, eTjetMin, etaJetMaxAlgo, 2, 2, nullptr, false);

  // For FxFx, also initialise jet algorithm to define matrix element jets.
  slowJetHard = new SlowJet(slowJetPower, coneRadius, qCutME, etaJetMaxAlgo, 2, 2, nullptr, false);

  // To access the DJR's
  slowJetDJR = new SlowJet(slowJetPower, coneRadius, qCutME, etaJetMaxAlgo, 2, 2, nullptr, false);

  // A special version of SlowJet to handle heavy and other partons
  hjSlowJet = new HJSlowJet(slowJetPower, coneRadius, 0.0, 100.0, 1, 2, nullptr, false, true);

  // Setup local event records
  eventProcessOrig.init("(eventProcessOrig)", particleDataPtr);
  eventProcess.init("(eventProcess)", particleDataPtr);
  workEventJet.init("(workEventJet)", particleDataPtr);

  // Print information
  if (MATCHINGDEBUG) {
    string jetStr = (jetAlgorithm == 1)    ? "CellJet"
                    : (slowJetPower == -1) ? "anti-kT"
                    : (slowJetPower == 0)  ? "C/A"
                    : (slowJetPower == 1)  ? "kT"
                                           : "unknown";
    string modeStr = (exclusiveMode) ? "exclusive" : "inclusive";
    cout << endl
         << " *-----  Madgraph matching parameters  -----*" << endl
         << " |  qCut                |  " << setw(14) << qCut << "  |" << endl
         << " |  nQmatch             |  " << setw(14) << nQmatch << "  |" << endl
         << " |  clFact              |  " << setw(14) << clFact << "  |" << endl
         << " |  Jet algorithm       |  " << setw(14) << jetStr << "  |" << endl
         << " |  eTjetMin            |  " << setw(14) << eTjetMin << "  |" << endl
         << " |  etaJetMax           |  " << setw(14) << etaJetMax << "  |" << endl
         << " |  jetAllow            |  " << setw(14) << jetAllow << "  |" << endl
         << " |  Mode                |  " << setw(14) << modeStr << "  |" << endl
         << " *-----------------------------------------*" << endl;
  }
  return true;
}

bool JetMatchingEWKFxFx::doVetoPartonLevelEarly(const Event& event) {
  // Sort event using the procedure required at parton level
  sortIncomingProcess(event);

  // For the shower-kT scheme, do not perform any veto here, as any vetoing
  // will already have taken place in doVetoStep.
  if (doShowerKt)
    return false;

  // Debug printout.
  if (MATCHINGDEBUG) {
    // Begin
    cout << endl << "-------- Begin Madgraph Debug --------" << endl;
    // Original incoming process
    cout << endl << "Original incoming process:";
    eventProcessOrig.list();
    // Final-state of original incoming process
    cout << endl << "Final-state incoming process:";
    eventProcess.list();
    // List categories of sorted particles
    for (size_t i = 0; i < typeIdx[0].size(); i++)
      cout << ((i == 0) ? "Light jets: " : ", ") << setw(3) << typeIdx[0][i];
    if (typeIdx[0].empty())
      cout << "Light jets: None";

    for (size_t i = 0; i < typeIdx[1].size(); i++)
      cout << ((i == 0) ? "\nHeavy jets: " : ", ") << setw(3) << typeIdx[1][i];
    for (size_t i = 0; i < typeIdx[2].size(); i++)
      cout << ((i == 0) ? "\nOther:      " : ", ") << setw(3) << typeIdx[2][i];
    // Full event at this stage
    cout << endl << endl << "Event:";
    event.list();
    // Work event (partons from hardest subsystem + ISR + FSR)
    cout << endl << "Work event:";
    workEvent.list();
  }

  // 2) Light/heavy jets: iType = 0 (light jets), 1 (heavy jets)
  int iTypeEnd = (typeIdx[2].empty()) ? 2 : 3;
  for (int iType = 0; iType < iTypeEnd; iType++) {
    // 2a) Find particles which will be passed from the jet algorithm.
    //     Input from 'workEvent' and output in 'workEventJet'.
    jetAlgorithmInput(event, iType);

    // Debug printout.
    if (MATCHINGDEBUG) {
      // Jet algorithm event
      cout << endl << "Jet algorithm event (iType = " << iType << "):";
      workEventJet.list();
    }

    // 2b) Run jet algorithm on 'workEventJet'.
    //     Output is stored in jetMomenta.
    runJetAlgorithm();

    // 2c) Match partons to jets and decide if veto is necessary
    if (matchPartonsToJets(iType) == true) {
      // Debug printout.
      if (MATCHINGDEBUG) {
        cout << endl
             << "Event vetoed"
             << "----------  End MLM Debug  ----------" << endl;
      }
      return true;
    }
  }

  // Debug printout.
  if (MATCHINGDEBUG) {
    cout << endl
         << "Event accepted"
         << "----------  End MLM Debug  ----------" << endl;
  }
  // If we reached here, then no veto
  return false;
}

void JetMatchingEWKFxFx::sortIncomingProcess(const Event& event) {
  omitResonanceDecays(eventProcessOrig, true);
  clearDJR();
  clear_nMEpartons();

  // Step-FxFx-1: remove preclustering from FxFx
  eventProcess = workEvent;

  for (int i = 0; i < 3; i++) {
    typeIdx[i].clear();
    typeSet[i].clear();
    origTypeIdx[i].clear();
  }

  for (int i = 0; i < eventProcess.size(); i++) {
    // Ignore non-final state and default to 'other'
    if (!eventProcess[i].isFinal())
      continue;
    int idx = -1;
    int orig_idx = -1;

    // Light jets: all gluons and quarks with id less than or equal to nQmatch
    if (eventProcess[i].isGluon() || (eventProcess[i].idAbs() <= nQmatch)) {
      orig_idx = 0;
      if (doFxFx) {
        // Crucial point FxFx: MG5 puts the scale of a not-to-be-matched quark 1 MeV lower than scalup. For
        // such particles, we should keep the default "2"
        idx = (trunc(1000. * eventProcess[i].scale()) == trunc(1000. * infoPtr->scalup())) ? 0 : 2;
      } else {
        // Crucial point: MG puts the scale of a non-QCD particle to eCM. For
        // such particles, we should keep the default "2"
        idx = (eventProcess[i].scale() < 1.999 * sqrt(infoPtr->eA() * infoPtr->eB())) ? 0 : 2;
      }
    }

    // Heavy jets:  all quarks with id greater than nQmatch
    else if (eventProcess[i].idAbs() > nQmatch && eventProcess[i].idAbs() <= ID_TOP) {
      idx = 1;
      orig_idx = 1;
      // Update to include non-SM colored particles
    } else if (eventProcess[i].colType() != 0 && eventProcess[i].idAbs() > ID_TOP) {
      idx = 1;
      orig_idx = 1;
    }
    if (idx < 0)
      continue;
    // Store
    typeIdx[idx].push_back(i);
    typeSet[idx].insert(eventProcess[i].daughter1());
    origTypeIdx[orig_idx].push_back(i);
  }
  // Exclusive mode; if set to 2, then set based on nJet/nJetMax
  if (exclusiveMode == 2) {
    // Inclusive if nJet == nJetMax, exclusive otherwise
    int nParton = origTypeIdx[0].size();
    exclusive = (nParton == nJetMax) ? false : true;

    // Otherwise, just set as given
  } else {
    exclusive = (exclusiveMode == 0) ? false : true;
  }

  // Extract partons from hardest subsystem + ISR + FSR only into
  // workEvent. Note no resonance showers or MPIs.
  subEvent(event);

  // Store things that are necessary to perform the kT-MLM veto externally.
  int nParton = typeIdx[0].size();
  processSubsetSave.clear();
  for (int i = 0; i < nParton; ++i)
    processSubsetSave.append(eventProcess[typeIdx[0][i]]);
}

bool JetMatchingEWKFxFx::doVetoProcessLevel(Event& process) {
  eventProcessOrig = process;

  // Setup for veto if hard ME has too many partons.
  // This is done to achieve consistency with the Pythia6 implementation.

  // Clear the event of MPI systems and resonace decay products. Store trimmed
  // event in workEvent.
  sortIncomingProcess(process);

  // Veto in case the hard input matrix element already has too many partons.
  if (!doFxFx && int(typeIdx[0].size()) > nJetMax)
    return true;
  if (doFxFx && npNLO() < nJetMax && int(typeIdx[0].size()) > nJetMax)
    return true;

  // Done
  return false;
}

void JetMatchingEWKFxFx::jetAlgorithmInput(const Event& event, int iType) {
  // Take input from 'workEvent' and put output in  'workEventJet'
  workEventJet = workEvent;

  // Loop over particles and decide what to pass to the jet algorithm
  for (int i = 0; i < workEventJet.size(); ++i) {
    if (!workEventJet[i].isFinal())
      continue;

    // jetAllow option to disallow certain particle types
    if (jetAllow == 1) {
      // Remove all non-QCD partons from veto list
      if (workEventJet[i].colType() == 0) {
        workEventJet[i].statusNeg();
        continue;
      }
    }
    // Get the index of this particle in original event
    int idx = workEventJet[i].daughter1();

    // Start with particle idx, and afterwards track mothers
    while (true) {
      // Light jets
      if (iType == 0) {
        // Do not include if originates from heavy jet or 'other'
        if (typeSet[1].find(idx) != typeSet[1].end() || typeSet[2].find(idx) != typeSet[2].end()) {
          workEventJet[i].statusNeg();
          break;
        }

        // Made it to start of event record so done
        if (idx == 0)
          break;
        // Otherwise next mother and continue
        idx = event[idx].mother1();

        // Heavy jets
      } else if (iType == 1) {
        // Only include if originates from heavy jet
        if (typeSet[1].find(idx) != typeSet[1].end())
          break;

        // Made it to start of event record with no heavy jet mother,
        // so DO NOT include particle
        if (idx == 0) {
          workEventJet[i].statusNeg();
          break;
        }

        // Otherwise next mother and continue
        idx = event[idx].mother1();

        // Other jets
      } else if (iType == 2) {
        // Only include if originates from other jet
        if (typeSet[2].find(idx) != typeSet[2].end())
          break;

        // Made it to start of event record with no heavy jet mother,
        // so DO NOT include particle
        if (idx == 0) {
          workEventJet[i].statusNeg();
          break;
        }

        // Otherwise next mother and continue
        idx = event[idx].mother1();
      }
    }
  }
}

void JetMatchingEWKFxFx::runJetAlgorithm() { ; }

bool JetMatchingEWKFxFx::doVetoStep(int iPos, int nISR, int nFSR, const Event& event) {
  // Do not perform any veto if not in the Shower-kT scheme.
  if (!doShowerKt)
    return false;

  // Do nothing for emissions after the first one.
  if (nISR + nFSR > 1)
    return false;

  // Do nothing in resonance decay showers.
  if (iPos == 5)
    return false;

  // Clear the event of MPI systems and resonace decay products. Store trimmed
  // event in workEvent.
  sortIncomingProcess(event);

  // Get (kinematical) pT of first emission
  double pTfirst = 0.;

  // Get weak bosons, for later checks if the emission is a "QCD emission".
  vector<int> weakBosons;
  for (int i = 0; i < event.size(); i++) {
    if (event[i].id() == 22 && event[i].id() == 23 && event[i].idAbs() == 24)
      weakBosons.push_back(i);
  }

  for (int i = workEvent.size() - 1; i > 0; --i) {
    if (workEvent[i].isFinal() && workEvent[i].colType() != 0 &&
        (workEvent[i].statusAbs() == 43 || workEvent[i].statusAbs() == 51)) {
      // Check if any of the EW bosons are ancestors of this parton. This
      // should never happen for the first non-resonance shower emission.
      // Check just to be sure.
      bool QCDemission = true;
      // Get position of this parton in the actual event (workEvent does
      // not contain right mother-daughter relations). Stored in daughters.
      int iPosOld = workEvent[i].daughter1();
      for (int j = 0; i < int(weakBosons.size()); ++i)
        if (event[iPosOld].isAncestor(j)) {
          QCDemission = false;
          break;
        }
      // Done for a QCD emission.
      if (QCDemission) {
        pTfirst = workEvent[i].pT();
        break;
      }
    }
  }

  // Store things that are necessary to perform the shower-kT veto externally.
  pTfirstSave = pTfirst;
  // Done if only inputs for an external vetoing procedure should be stored.
  //if (!performVeto) return false;

  // Check veto.
  if (doShowerKtVeto(pTfirst))
    return true;

  // No veto if come this far.
  return false;
}

void JetMatchingEWKFxFx::setDJR(const Event& event) {
  // Clear members.
  clearDJR();
  vector<double> result;

  // Initialize SlowJetDJR jet algorithm with event
  if (!slowJetDJR->setup(event)) {
    errorMsg(
        "Warning in JetMatchingMadgraph:setDJR"
        ": the SlowJet algorithm failed on setup");
    return;
  }

  // Cluster in steps to find all hadronic jets
  while (slowJetDJR->sizeAll() - slowJetDJR->sizeJet() > 0) {
    // Save the next clustering scale.
    result.push_back(sqrt(slowJetDJR->dNext()));
    // Perform step.
    slowJetDJR->doStep();
  }

  // Save clustering scales in reserve order.
  for (int i = int(result.size()) - 1; i >= 0; --i)
    DJR.push_back(result[i]);
}

bool JetMatchingEWKFxFx::matchPartonsToJets(int iType) {
  // Use different routines for light/heavy/other jets as
  // different veto conditions and for clarity
  if (iType == 0) {
    // Record the jet separations here, also if matchPartonsToJetsLight
    // returns preemptively.
    setDJR(workEventJet);
    set_nMEpartons(origTypeIdx[0].size(), typeIdx[0].size());
    // Perform jet matching.
    return (matchPartonsToJetsLight() > 0);
  } else if (iType == 1) {
    return (matchPartonsToJetsHeavy() > 0);
  } else {
    return (matchPartonsToJetsOther() > 0);
  }
}

int JetMatchingEWKFxFx::matchPartonsToJetsLight() {
  // Store things that are necessary to perform the kT-MLM veto externally.
  workEventJetSave = workEventJet;
  // Done if only inputs for an external vetoing procedure should be stored.
  //if (!performVeto) return false;

  // Count the number of hard partons
  int nParton = typeIdx[0].size();

  // Initialize SlowJet with current working event
  if (!slowJet->setup(workEventJet)) {
    errorMsg(
        "Warning in JetMatchingMadgraph:matchPartonsToJets"
        "Light: the SlowJet algorithm failed on setup");
    return NONE;
  }
  double localQcutSq = qCutSq;
  double dOld = 0.0;
  // Cluster in steps to find all hadronic jets at the scale qCut
  while (slowJet->sizeAll() - slowJet->sizeJet() > 0) {
    if (slowJet->dNext() > localQcutSq)
      break;
    dOld = slowJet->dNext();
    slowJet->doStep();
  }
  int nJets = slowJet->sizeJet();
  int nClus = slowJet->sizeAll();

  // Debug printout.
  if (MATCHINGDEBUG)
    slowJet->list(true);

  // Count of the number of hadronic jets in SlowJet accounting
  int nCLjets = nClus - nJets;
  // Get number of partons. Different for MLM and FxFx schemes.
  //int nRequested = (doFxFx) ? npNLO() : nParton;
  //Step-FxFx-3: Change nRequested subtracting the typeIdx[2] partons
  //Exclude the highest multiplicity sample in the case of real emissions and all typeIdx[2]
  //npNLO=multiplicity,nJetMax=njmax(shower_card),typeIdx[2]="Weak" jets
  int nRequested = (doFxFx && !(npNLO() == nJetMax && npNLO() == (int)(typeIdx[2].size() - 1)))
                       ? npNLO() - typeIdx[2].size()
                       : nParton;

  //Step-FxFx-4:For FxFx veto the real emissions that have only typeIdx=2 partons
  //From Step-FxFx-3 they already have negative nRequested, so this step may not be necessary
  //Exclude the highest multiplicity sample
  if (doFxFx && npNLO() < nJetMax && !typeIdx[2].empty() && npNLO() == (int)(typeIdx[2].size() - 1)) {
    return MORE_JETS;
  }
  //----------------
  //Exclude all Weak Jets for matching
  //if (doFxFx && typeIdx[2].size()>0) {
  //      return MORE_JETS;
  //} // 2A
  //keep only Weak Jets for matching
  //if (doFxFx && typeIdx[2].size()==0) {
  //      return MORE_JETS;
  //} // 2B
  //keep only lowest multiplicity sample @0
  //if (doFxFx && npNLO()==1) {
  //      return MORE_JETS;
  //} // 2C
  //keep only highest multiplicity sample @1
  //if (doFxFx && npNLO()==0) {
  //      return MORE_JETS;
  //} // 2D
  //--------------
  // Veto event if too few hadronic jets
  if (nCLjets < nRequested)
    return LESS_JETS;

  // In exclusive mode, do not allow more hadronic jets than partons
  if (exclusive && !doFxFx) {
    if (nCLjets > nRequested)
      return MORE_JETS;
  } else {
    // For FxFx, in the non-highest multipicity, all jets need to matched to
    // partons. For nCLjets > nRequested, this is not possible. Hence, we can
    // veto here already.
    // if ( doFxFx && nRequested < nJetMax && nCLjets > nRequested )
    //Step-FxFx-5:Change the nRequested to npNLO() in the first condition
    //Before in Step-FxFx-3 it was nRequested=npNLO() for FxFx
    //This way we correctly select the non-highest multipicity regardless the nRequested
    if (doFxFx && npNLO() < nJetMax && nCLjets > nRequested)
      return MORE_JETS;

    // Now continue in inclusive mode.
    // In inclusive mode, there can be more hadronic jets than partons,
    // provided that all partons are properly matched to hadronic jets.
    // Start by setting up the jet algorithm.
    if (!slowJet->setup(workEventJet)) {
      errorMsg(
          "Warning in JetMatchingMadgraph:matchPartonsToJets"
          "Light: the SlowJet algorithm failed on setup");
      return NONE;
    }

    // For FxFx, continue clustering as long as the jet separation is above
    // qCut.
    if (doFxFx) {
      while (slowJet->sizeAll() - slowJet->sizeJet() > 0) {
        if (slowJet->dNext() > localQcutSq)
          break;
        slowJet->doStep();
      }
      // For MLM, cluster into hadronic jets until there are the same number as
      // partons.
    } else {
      while (slowJet->sizeAll() - slowJet->sizeJet() > nParton)
        slowJet->doStep();
    }

    // Sort partons in pT.  Update local qCut value.
    //  Hadronic jets are already sorted in pT.
    localQcutSq = dOld;
    if (clFact >= 0. && nParton > 0) {
      vector<double> partonPt;
      partonPt.reserve(nParton);
      for (int i = 0; i < nParton; ++i)
        partonPt.push_back(eventProcess[typeIdx[0][i]].pT2());
      sort(partonPt.begin(), partonPt.end());
      localQcutSq = max(qCutSq, partonPt[0]);
    }
    nJets = slowJet->sizeJet();
    nClus = slowJet->sizeAll();
  }
  // Update scale if clustering factor is non-zero
  if (clFact != 0.)
    localQcutSq *= pow2(clFact);

  Event tempEvent;
  tempEvent.init("(tempEvent)", particleDataPtr);
  int nPass = 0;
  double pTminEstimate = -1.;
  // Construct a master copy of the event containing only the
  // hardest nParton hadronic clusters. While constructing the event,
  // the parton type (ID_GLUON) and status (98,99) are arbitrary.
  for (int i = nJets; i < nClus; ++i) {
    tempEvent.append(
        ID_GLUON, 98, 0, 0, 0, 0, 0, 0, slowJet->p(i).px(), slowJet->p(i).py(), slowJet->p(i).pz(), slowJet->p(i).e());
    ++nPass;
    pTminEstimate = max(pTminEstimate, slowJet->pT(i));
    if (nPass == nRequested)
      break;
  }

  int tempSize = tempEvent.size();
  // This keeps track of which hadronic jets are matched to parton
  vector<bool> jetAssigned;
  jetAssigned.assign(tempSize, false);

  // This keeps track of which partons are matched to which hadronic
  // jets.
  vector<vector<bool> > partonMatchesJet;
  partonMatchesJet.reserve(nParton);
  for (int i = 0; i < nParton; ++i)
    partonMatchesJet.push_back(vector<bool>(tempEvent.size(), false));

  // Begin matching.
  // Do jet matching for FxFx.
  // Make sure that the nPartonsNow hardest hadronic jets are matched to any
  // of the nPartonsNow (+1) partons. This matching is done by attaching a jet
  // from the list of unmatched hadronic jets, and appending a jet from the
  // list of partonic jets, one at a time. The partonic jet will be clustered
  // with the hadronic jet or the beam if the distance measure is below the
  // cut. The hadronic jet is matched once this happens. Otherwise, another
  // partonic jet is tried. When a hadronic jet is matched to a partonic jet,
  // it is removed from the list of unmatched hadronic jets. This process
  // continues until the nPartonsNow hardest hadronic jets are matched to
  // partonic jets, or it is not possible to make a match for a hadronic jet.
  int iNow = 0;
  int nMatched = 0;

  while (doFxFx && iNow < tempSize) {
    // Check if this shower jet matches any partonic jet.
    Event tempEventJet;
    tempEventJet.init("(tempEventJet)", particleDataPtr);
    for (int i = 0; i < nParton; ++i) {
      //// Only assign a parton once.
      //for (int j=0; j < tempSize; ++j )
      //  if ( partonMatchesJet[i][j]) continue;

      // Attach a single hadronic jet.
      tempEventJet.clear();
      tempEventJet.append(ID_GLUON,
                          98,
                          0,
                          0,
                          0,
                          0,
                          0,
                          0,
                          tempEvent[iNow].px(),
                          tempEvent[iNow].py(),
                          tempEvent[iNow].pz(),
                          tempEvent[iNow].e());
      // Attach the current parton.
      Vec4 pIn = eventProcess[typeIdx[0][i]].p();
      tempEventJet.append(ID_GLUON, 99, 0, 0, 0, 0, 0, 0, pIn.px(), pIn.py(), pIn.pz(), pIn.e());

      // Setup jet algorithm.
      if (!slowJet->setup(tempEventJet)) {
        errorMsg(
            "Warning in JetMatchingMadgraph:matchPartonsToJets"
            "Light: the SlowJet algorithm failed on setup");
        return NONE;
      }
      // These are the conditions for the hadronic jet to match the parton
      //  at the local qCut scale
      if (slowJet->iNext() == tempEventJet.size() - 1 && slowJet->jNext() > -1 && slowJet->dNext() < localQcutSq) {
        jetAssigned[iNow] = true;
        partonMatchesJet[i][iNow] = true;
      }
    }  // End loop over hard partons.

    // Veto if the jet could not be assigned to any parton.
    if (jetAssigned[iNow])
      nMatched++;

    // Continue;
    ++iNow;
  }
  // Jet matching veto for FxFx
  if (doFxFx) {
    // if ( nRequested <  nJetMax && nMatched != nRequested )
    //   return UNMATCHED_PARTON;
    // if ( nRequested == nJetMax && nMatched <  nRequested )
    //   return UNMATCHED_PARTON;
    //Step-FxFx-6:Change the nRequested to npNLO() in the first condition (like in Step-FxFx-5)
    //Before in Step-FxFx-3 it was nRequested=npNLO() for FxFx
    //This way we correctly select the non-highest multipicity regardless the nRequested
    if (npNLO() < nJetMax && nMatched != nRequested)
      return UNMATCHED_PARTON;
    if (npNLO() == nJetMax && nMatched < nRequested)
      return UNMATCHED_PARTON;
  }
  // Do jet matching for MLM.
  // Take the list of unmatched hadronic jets and append a parton, one at
  // a time. The parton will be clustered with the "closest" hadronic jet
  // or the beam if the distance measure is below the cut. When a hadronic
  // jet is matched to a parton, it is removed from the list of unmatched
  // hadronic jets. This process continues until all hadronic jets are
  // matched to partons or it is not possible to make a match.
  iNow = 0;
  while (!doFxFx && iNow < nParton) {
    Event tempEventJet;
    tempEventJet.init("(tempEventJet)", particleDataPtr);
    for (int i = 0; i < tempSize; ++i) {
      if (jetAssigned[i])
        continue;
      Vec4 pIn = tempEvent[i].p();
      // Append unmatched hadronic jets
      tempEventJet.append(ID_GLUON, 98, 0, 0, 0, 0, 0, 0, pIn.px(), pIn.py(), pIn.pz(), pIn.e());
    }

    Vec4 pIn = eventProcess[typeIdx[0][iNow]].p();
    // Append the current parton
    tempEventJet.append(ID_GLUON, 99, 0, 0, 0, 0, 0, 0, pIn.px(), pIn.py(), pIn.pz(), pIn.e());
    if (!slowJet->setup(tempEventJet)) {
      errorMsg(
          "Warning in JetMatchingMadgraph:matchPartonsToJets"
          "Light: the SlowJet algorithm failed on setup");
      return NONE;
    }
    // These are the conditions for the hadronic jet to match the parton
    //  at the local qCut scale
    if (slowJet->iNext() == tempEventJet.size() - 1 && slowJet->jNext() > -1 && slowJet->dNext() < localQcutSq) {
      int iKnt = -1;
      for (int i = 0; i != tempSize; ++i) {
        if (jetAssigned[i])
          continue;
        ++iKnt;
        // Identify the hadronic jet that matches the parton
        if (iKnt == slowJet->jNext())
          jetAssigned[i] = true;
      }
    } else {
      return UNMATCHED_PARTON;
    }
    ++iNow;
  }
  // Minimal eT/pT (CellJet/SlowJet) of matched light jets.
  // Needed later for heavy jet vetos in inclusive mode.
  // This information is not used currently.
  if (nParton > 0 && pTminEstimate > 0)
    eTpTlightMin = pTminEstimate;
  else
    eTpTlightMin = -1.;

  // Record the jet separations.
  setDJR(workEventJet);

  // No veto
  return NONE;
}

int JetMatchingEWKFxFx::matchPartonsToJetsHeavy() {
  // Currently, heavy jets are unmatched
  // If there are no extra jets, then accept
  // jetMomenta is NEVER used by MadGraph and is always empty.
  //  This check does nothing.
  //  Rather, if there is any heavy flavor that is harder than
  //  what is present at the LHE level, then the event should
  //  be vetoed.

  // if (jetMomenta.empty()) return NONE;
  // Count the number of hard partons
  int nParton = typeIdx[1].size();

  Event tempEventJet(workEventJet);

  double scaleF(1.0);
  // Rescale the heavy partons that are from the hard process to
  //  have pT=collider energy.   Soft/collinear gluons will cluster
  //  onto them, leaving a remnant of hard emissions.
  for (int i = 0; i < nParton; ++i) {
    scaleF = eventProcessOrig[0].e() / workEventJet[typeIdx[1][i]].pT();
    tempEventJet[typeIdx[1][i]].rescale5(scaleF);
  }

  if (!hjSlowJet->setup(tempEventJet)) {
    errorMsg(
        "Warning in JetMatchingMadgraph:matchPartonsToJets"
        "Heavy: the SlowJet algorithm failed on setup");
    return NONE;
  }

  while (hjSlowJet->sizeAll() - hjSlowJet->sizeJet() > 0) {
    if (hjSlowJet->dNext() > qCutSq)
      break;
    hjSlowJet->doStep();
  }

  int nCLjets(0);
  // Count the number of clusters with pT>qCut.  This includes the
  //  original hard partons plus any hard emissions.
  for (int idx = 0; idx < hjSlowJet->sizeAll(); ++idx) {
    if (hjSlowJet->pT(idx) > qCut)
      nCLjets++;
  }

  // Debug printout.
  if (MATCHINGDEBUG)
    hjSlowJet->list(true);

  // Count of the number of hadronic jets in SlowJet accounting
  //  int nCLjets = nClus - nJets;
  // Get number of partons. Different for MLM and FxFx schemes.
  int nRequested = nParton;

  // Veto event if too few hadronic jets
  if (nCLjets < nRequested) {
    if (MATCHINGDEBUG)
      cout << "veto : hvy  LESS_JETS " << endl;
    if (MATCHINGDEBUG)
      cout << "nCLjets = " << nCLjets << "; nRequest = " << nRequested << endl;
    return LESS_JETS;
  }

  // In exclusive mode, do not allow more hadronic jets than partons
  if (exclusive) {
    if (nCLjets > nRequested) {
      if (MATCHINGDEBUG)
        cout << "veto : excl hvy  MORE_JETS " << endl;
      return MORE_JETS;
    }
  }

  // No extra jets were present so no veto
  return NONE;
}

int JetMatchingEWKFxFx::matchPartonsToJetsOther() {
  // Currently, heavy jets are unmatched
  // If there are no extra jets, then accept
  // jetMomenta is NEVER used by MadGraph and is always empty.
  //  This check does nothing.
  //  Rather, if there is any heavy flavor that is harder than
  //  what is present at the LHE level, then the event should
  //  be vetoed.

  // if (jetMomenta.empty()) return NONE;
  // Count the number of hard partons
  int nParton = typeIdx[2].size();

  Event tempEventJet(workEventJet);

  double scaleF(1.0);
  // Rescale the heavy partons that are from the hard process to
  //  have pT=collider energy.   Soft/collinear gluons will cluster
  //  onto them, leaving a remnant of hard emissions.
  for (int i = 0; i < nParton; ++i) {
    scaleF = eventProcessOrig[0].e() / workEventJet[typeIdx[2][i]].pT();
    tempEventJet[typeIdx[2][i]].rescale5(scaleF);
  }

  if (!hjSlowJet->setup(tempEventJet)) {
    errorMsg(
        "Warning in JetMatchingMadgraph:matchPartonsToJets"
        "Heavy: the SlowJet algorithm failed on setup");
    return NONE;
  }

  while (hjSlowJet->sizeAll() - hjSlowJet->sizeJet() > 0) {
    if (hjSlowJet->dNext() > qCutSq)
      break;
    hjSlowJet->doStep();
  }

  int nCLjets(0);
  // Count the number of clusters with pT>qCut.  This includes the
  //  original hard partons plus any hard emissions.
  for (int idx = 0; idx < hjSlowJet->sizeAll(); ++idx) {
    if (hjSlowJet->pT(idx) > qCut)
      nCLjets++;
  }

  // Debug printout.
  if (MATCHINGDEBUG)
    hjSlowJet->list(true);

  // Count of the number of hadronic jets in SlowJet accounting
  //  int nCLjets = nClus - nJets;
  // Get number of partons. Different for MLM and FxFx schemes.
  int nRequested = nParton;

  // Veto event if too few hadronic jets
  if (nCLjets < nRequested) {
    if (MATCHINGDEBUG)
      cout << "veto : other LESS_JETS " << endl;
    if (MATCHINGDEBUG)
      cout << "nCLjets = " << nCLjets << "; nRequest = " << nRequested << endl;
    return LESS_JETS;
  }

  // In exclusive mode, do not allow more hadronic jets than partons
  if (exclusive) {
    if (nCLjets > nRequested) {
      if (MATCHINGDEBUG)
        cout << "veto : excl other MORE_JETS" << endl;
      return MORE_JETS;
    }
  }

  // No extra jets were present so no veto
  return NONE;
}

bool JetMatchingEWKFxFx::doShowerKtVeto(double pTfirst) {
  // Only check veto in the shower-kT scheme.
  if (!doShowerKt)
    return false;

  // Reset veto code
  bool doVeto = false;

  // Find the (kinematical) pT of the softest (light) parton in the hard
  // process.
  int nParton = typeIdx[0].size();
  double pTminME = 1e10;
  for (int i = 0; i < nParton; ++i)
    pTminME = min(pTminME, eventProcess[typeIdx[0][i]].pT());

  // Veto if the softest hard process parton is below Qcut.
  if (nParton > 0 && pow(pTminME, 2) < qCutSq)
    doVeto = true;

  // For non-highest multiplicity, veto if the hardest emission is harder
  // than Qcut.
  if (exclusive && pow(pTfirst, 2) > qCutSq) {
    doVeto = true;
    // For highest multiplicity sample, veto if the hardest emission is harder
    // than the hard process parton.
  } else if (!exclusive && nParton > 0 && pTfirst > pTminME) {
    doVeto = true;
  }

  // Return veto
  return doVeto;
}

// Function to get the current number of partons in the Born state, as
// read from LHE.

int JetMatchingEWKFxFx::npNLO() {
  string npIn = infoPtr->getEventAttribute("npNLO", true);
  int np = !npIn.empty() ? std::atoi(npIn.c_str()) : -1;
  if (np < 0) {
    ;
  } else
    return np;
  return nPartonsNow;
}
