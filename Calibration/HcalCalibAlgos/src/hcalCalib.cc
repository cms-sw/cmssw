//  TSelector-based code for getting the HCAL resp. correction
//  from physics events. Works for DiJet and IsoTrack calibration.
//
//  Anton Anastassov (Northwestern)
//  Email: aa@fnal.gov
//
//

#include "Calibration/HcalCalibAlgos/interface/hcalCalib.h"
#include "Calibration/HcalCalibAlgos/interface/hcalCalibUtils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <map>
#include <numeric>
#include <algorithm>
#include <set>

#include "Calibration/Tools/interface/MinL3AlgoUniv.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

void hcalCalib::Begin(TTree* /*tree*/) {
  TString option = GetOption();

  nEvents = 0;

  if (APPLY_PHI_SYM_COR_FLAG && !ReadPhiSymCor()) {
    edm::LogError("HcalCalib") << "\nERROR: Failed to read the phi symmetry corrections.\n"
                               << "Check if the filename is correct. If the corrections are not needed, set the "
                                  "corresponding flag to \"false\"\n"
                               << "\nThe program will be terminated\n";

    exit(1);
  }

  //  cellEnergies.reserve(1000000);
  //  cellIds.reserve(1000000);
  //  targetEnergies.reserve(1000000);

  histoFile = new TFile(HISTO_FILENAME.Data(), "RECREATE");

  h1_trkP = new TH1F("h1_trkP", "Track momenta; p_{trk} (GeV); Number of tracks", 100, 0, 200);
  h1_allTrkP = new TH1F("h1_allTrkP", "Track momenta - all tracks; p_{trk} (GeV); Number of tracks", 100, 0, 200);

  h1_selTrkP_iEta10 = new TH1F(
      "h1_selTrkP_iEta10", "Track momenta - tracks with |iEta|<10; p_{trk} (GeV); Number of tracks", 100, 0, 200);

  if (CALIB_TYPE == "ISO_TRACK")
    h1_rawSumE = new TH1F("h1_rawSumE", "Cluster Energy; E_{cl} (GeV); Number of tracks", 100, 0, 200);
  else
    h1_rawSumE = new TH1F("h1_rawSumE", "Cluster Energy; E_{cl} (GeV); Number of tracks", 1000, 0, 2000);

  h1_rawResp = new TH1F("h1_rawResp", "Uncorrected response: |iEta|<24;  E_{had}/p; Number of tracks", 300, 0, 3);
  h1_corResp = new TH1F("h1_corResp", "Corrected response: |iEta|<24; E_{had}/p; Number of tracks", 300, 0, 3);

  h1_rawRespBarrel =
      new TH1F("h1_rawRespBarrel", "Uncorrected response: |iEta|<15;  E_{had}/p; Number of tracks", 300, 0, 3);
  h1_corRespBarrel =
      new TH1F("h1_corRespBarrel", "Corrected response: |iEta|<15; E_{had}/p; Number of tracks", 300, 0, 3);

  h1_rawRespEndcap =
      new TH1F("h1_rawRespEndcap", "Uncorrected response:  17<|iEta|<24;  E_{had}/p; Number of tracks", 300, 0, 3);
  h1_corRespEndcap =
      new TH1F("h1_corRespEndcap", "Corrected response: 17<|iEta|<24; E_{had}/p; Number of tracks", 300, 0, 3);

  h1_numEventsTwrIEta = new TH1F("h1_numEventsTwrIEta", "h1_numEventsTwrIEta", 80, -40, 40);

  h2_dHitRefBarrel = new TH2F("h2_dHitRefBarrel",
                              "{#Delta}i{#phi} vs {#Delta}i{#eta} of hit and most energetic "
                              "tower(|i{#eta}|<16);{#Delta}i{#eta}; {#Delta}i{#phi}",
                              10,
                              -5,
                              5,
                              10,
                              -5,
                              5);
  h2_dHitRefEndcap = new TH2F("h2_dHitRefEndcap",
                              "{#Delta}i{#phi} vs {#Delta}i{#eta} of hit and most energetic tower (16<|i{#eta}|<25) "
                              ";{#Delta}i{#eta}; {#Delta}i{#phi}",
                              10,
                              -5,
                              5,
                              10,
                              -5,
                              5);

  TString histoName = "isoTrack_";

  for (Int_t i = 0; i < 48; ++i) {
    Long_t iEta;
    if (i < 24)
      iEta = i - 24;
    else
      iEta = i - 23;
    TString hn = histoName + iEta;
    h1_corRespIEta[i] = new TH1F(hn, hn, 300, 0, 3.0);
  }

}  // end of Begin()

//void hcalCalib::SlaveBegin(TTree * /*tree*/) {
//  TString option = GetOption();
//}

Bool_t hcalCalib::Process(Long64_t entry) {
  //  fChain->GetTree()->GetEntry(entry);
  GetEntry(entry);

  std::set<UInt_t> uniqueIds;  // for testing: check if there are duplicate cells   (AA)

  Bool_t acceptEvent = kTRUE;

  ++nEvents;

  if (!(nEvents % 100000))
    edm::LogVerbatim("HcalCalib") << "event: " << nEvents;

  h1_allTrkP->Fill(targetE);

  if (targetE < MIN_TARGET_E || targetE > MAX_TARGET_E)
    return kFALSE;
  ;

  // make local copy as the cells may be modified  due to phi/depth sum, phi corrections etc
  std::vector<TCell> selectCells;

  if (cells->GetSize() == 0)
    return kFALSE;

  if (CALIB_TYPE == "DI_JET" && probeJetEmFrac > 0.999)
    return kTRUE;

  for (Int_t i = 0; i < cells->GetSize(); ++i) {
    TCell* thisCell = (TCell*)cells->At(i);

    if (HcalDetId(thisCell->id()).subdet() == HcalOuter)
      continue;  // reject HO, make a switch!

    if (HcalDetId(thisCell->id()).subdet() != HcalBarrel && HcalDetId(thisCell->id()).subdet() != HcalEndcap &&
        HcalDetId(thisCell->id()).subdet() != HcalForward) {
      edm::LogWarning("HcalCalib") << "Unknown or wrong hcal subdetector: " << HcalDetId(thisCell->id()).subdet();
    }

    // Apply phi symmetry corrections if the flag is set
    if (APPLY_PHI_SYM_COR_FLAG)
      thisCell->SetE(phiSymCor[thisCell->id()] * thisCell->e());

    if (thisCell->e() > MIN_CELL_E)
      selectCells.push_back(*thisCell);
  }

  if (selectCells.empty()) {
    edm::LogWarning("HcalCalib") << "NO CELLS ABOVE THRESHOLD FOUND FOR TARGET!!!";
  }

  if (SUM_DEPTHS)
    sumDepths(selectCells);
  else if (SUM_SMALL_DEPTHS)
    sumSmallDepths(selectCells);  // depth 1,2 in twrs 15,16

  // most energetic tower (IsoTracks) or centroid of probe jet (DiJets)
  std::pair<Int_t, UInt_t> refPos;

  Int_t dEtaHitRef = 999;
  Int_t dPhiHitRef = 999;

  if (CALIB_TYPE == "ISO_TRACK") {
    Int_t iEtaMaxE;   // filled by reference in getIEtaIPhiForHighestE
    UInt_t iPhiMaxE;  //

    getIEtaIPhiForHighestE(selectCells, iEtaMaxE, iPhiMaxE);

    dEtaHitRef = iEtaMaxE - iEtaHit;
    dPhiHitRef = iPhiMaxE - iPhiHit;

    if (dPhiHitRef < -36)
      dPhiHitRef += 72;
    if (dPhiHitRef > 36)
      dPhiHitRef -= 72;

    if (iEtaHit * iEtaMaxE < 0) {
      if (dEtaHitRef < 0)
        dEtaHitRef += 1;
      if (dEtaHitRef > 0)
        dEtaHitRef -= 1;
    }

    if (abs(iEtaHit) < 16)
      h2_dHitRefBarrel->Fill(dEtaHitRef, dPhiHitRef);
    if (abs(iEtaHit) > 16 && abs(iEtaHit) < 25)
      h2_dHitRefEndcap->Fill(dEtaHitRef, dPhiHitRef);

    // --------------------------------------------------
    // Choice of cluster definition
    //
    // fixed size NxN clusters as specified in to config file
    if (!USE_CONE_CLUSTERING) {
      if (abs(iEtaMaxE) < 16 && HB_CLUSTER_SIZE == 3)
        filterCells3x3(selectCells, iEtaMaxE, iPhiMaxE);
      if (abs(iEtaMaxE) > 15 && HE_CLUSTER_SIZE == 3)
        filterCells3x3(selectCells, iEtaMaxE, iPhiMaxE);

      if (abs(iEtaMaxE) < 16 && HB_CLUSTER_SIZE == 5)
        filterCells5x5(selectCells, iEtaMaxE, iPhiMaxE);
      if (abs(iEtaMaxE) > 15 && HE_CLUSTER_SIZE == 5)
        filterCells5x5(selectCells, iEtaMaxE, iPhiMaxE);
    } else {
      //  calculate distance at hcal surface
      const GlobalPoint hitPositionHcal(xTrkHcal, yTrkHcal, zTrkHcal);
      filterCellsInCone(selectCells, hitPositionHcal, MAX_CONE_DIST, theCaloGeometry);
    }

    refPos.first = iEtaMaxE;
    refPos.second = iPhiMaxE;

  } else if (CALIB_TYPE == "DI_JET") {  // Apply selection cuts on DiJet events here

    if (etVetoJet > MAX_ET_THIRD_JET)
      acceptEvent = kFALSE;

    Float_t jetsDPhi = probeJetP4->DeltaPhi(*tagJetP4);
    if (fabs(jetsDPhi * 180.0 / M_PI) < MIN_DPHI_DIJETS)
      acceptEvent = kFALSE;

    if (probeJetEmFrac > MAX_PROBEJET_EMFRAC)
      acceptEvent = kFALSE;
    if (fabs(probeJetP4->Eta()) < MIN_PROBEJET_ABSETA)
      acceptEvent = kFALSE;
    if (fabs(tagJetP4->Eta()) > MAX_TAGJET_ABSETA)
      acceptEvent = kFALSE;
    if (fabs(tagJetP4->Et()) < MIN_TAGJET_ET)
      acceptEvent = kFALSE;

    if (acceptEvent) {
      Int_t iEtaMaxE;   // filled by reference in getIEtaIPhiForHighestE
      UInt_t iPhiMaxE;  //

      getIEtaIPhiForHighestE(selectCells, iEtaMaxE, iPhiMaxE);

      // The ref position for the jet is not used in the minimization at this time.
      // It will be needed if we attempt to do matrix inversion: then the question is
      // which value is better suited: the centroid of the jet or the hottest tower...

      //    refPos.first  = iEtaHit;
      //    refPos.second = iPhiHit;

      refPos.first = iEtaMaxE;
      refPos.second = iPhiMaxE;

      if (abs(iEtaMaxE) > 40)
        acceptEvent = kFALSE;  // for testing :  set as parameter (AA)
    }
  }

  if (COMBINE_PHI)
    combinePhi(selectCells);

  // fill the containers for the minimization prcedures
  std::vector<Float_t> energies;
  std::vector<UInt_t> ids;

  for (std::vector<TCell>::iterator i_it = selectCells.begin(); i_it != selectCells.end(); ++i_it) {
    // for testing : fill only unique id's

    if (uniqueIds.insert(i_it->id()).second) {
      energies.push_back(i_it->e());
      ids.push_back(i_it->id());
    }
  }

  if (CALIB_TYPE == "ISO_TRACK") {
    if (accumulate(energies.begin(), energies.end(), 0.0) / targetE < MIN_EOVERP)
      acceptEvent = kFALSE;
    if (accumulate(energies.begin(), energies.end(), 0.0) / targetE > MAX_EOVERP)
      acceptEvent = kFALSE;

    if (emEnergy > MAX_TRK_EME)
      acceptEvent = kFALSE;

    if (abs(dEtaHitRef) > 1 || abs(dPhiHitRef) > 1)
      acceptEvent = kFALSE;

    // Have to check if for |iEta|>20 (and neighboring region) the dPhiHitRef
    // should be relaxed to 2. The neighboring towers have dPhi=2...
  }

  h1_rawSumE->Fill(accumulate(energies.begin(), energies.end(), 0.0));

  // here we fill the information for the minimization procedure
  if (acceptEvent) {
    cellEnergies.push_back(energies);
    cellIds.push_back(ids);
    targetEnergies.push_back(targetE);
    refIEtaIPhi.push_back(refPos);

    if (abs(refPos.first) <= 10)
      h1_selTrkP_iEta10->Fill(targetE);
  }

  // Clean up
  energies.clear();
  ids.clear();
  selectCells.clear();

  return kTRUE;
}

//void hcalCalib::SlaveTerminate() {}

void hcalCalib::Terminate() {
  edm::LogVerbatim("HcalCalib") << "\n\nFinished reading the events.\n"
                                << "Number of input objects: " << cellIds.size()
                                << "\nPerforming minimization: depending on selected method can take some time...\n\n";

  for (std::vector<std::pair<Int_t, UInt_t> >::iterator it_rp = refIEtaIPhi.begin(); it_rp != refIEtaIPhi.end();
       ++it_rp) {
    Float_t weight = (abs(it_rp->first) < 21) ? 1.0 / 72.0 : 1.0 / 36.0;
    h1_numEventsTwrIEta->Fill(it_rp->first, weight);
  }

  if (CALIB_METHOD == "MATRIX_INV_OF_ETA_AVE") {
    GetCoefFromMtrxInvOfAve();
  } else if (CALIB_METHOD == "L3" || CALIB_METHOD == "L3_AND_MTRX_INV") {
    int eventWeight = 2;  // 2 is the default (try at some point 0,1,2,3)
    MinL3AlgoUniv<UInt_t>* thisL3Algo = new MinL3AlgoUniv<UInt_t>(eventWeight);
    int numIterations = 10;  // default 10

    solution = thisL3Algo->iterate(cellEnergies, cellIds, targetEnergies, numIterations);

    // in order to handle the case where sumDepths="false", but the flag to sum depths 1,2 in HB towers 15, 16
    // is set (sumSmallDepths) we create entries in "solution" to create equal correction here
    // for each encountered coef in depth one.

    if (!SUM_DEPTHS && SUM_SMALL_DEPTHS) {
      std::vector<UInt_t> idForSummedCells;

      for (std::map<UInt_t, Float_t>::iterator m_it = solution.begin(); m_it != solution.end(); ++m_it) {
        if (HcalDetId(m_it->first).ietaAbs() != 15 && HcalDetId(m_it->first).ietaAbs() != 16)
          continue;
        if (HcalDetId(m_it->first).subdet() != HcalBarrel)
          continue;
        if (HcalDetId(m_it->first).depth() == 1)
          idForSummedCells.push_back(HcalDetId(m_it->first));
      }

      for (std::vector<UInt_t>::iterator v_it = idForSummedCells.begin(); v_it != idForSummedCells.end(); ++v_it) {
        UInt_t addCoefId = HcalDetId(HcalBarrel, HcalDetId(*v_it).ieta(), HcalDetId(*v_it).iphi(), 2);
        solution[addCoefId] = solution[*v_it];
      }

    }  // end of special treatment for "sumSmallDepths" mode

    if (CALIB_METHOD == "L3_AND_MTRX_INV") {
      GetCoefFromMtrxInvOfAve();

      // loop over the solution from L3 and multiply by the additional factor from
      // the matrix inversion. Set coef outside of the valid calibration region =1.
      for (std::map<UInt_t, Float_t>::iterator it_s = solution.begin(); it_s != solution.end(); ++it_s) {
        Int_t iEtaSol = HcalDetId(it_s->first).ieta();
        if (abs(iEtaSol) < CALIB_ABS_IETA_MIN || abs(iEtaSol) > CALIB_ABS_IETA_MAX)
          it_s->second = 1.0;
        else
          it_s->second *= iEtaCoefMap[iEtaSol];
      }

    }  // if CALIB_METHOD=="L3_AND_MTRX_INV"

  }  // end of L3 or L3 + mtrx inversion minimization

  // done getting the constants -> write the formatted file
  makeTextFile();

  // fill some histograms
  Float_t rawResp = 0;
  Float_t corResp = 0;
  Int_t maxIEta = 0;
  Int_t minIEta = 999;

  for (unsigned int i = 0; i < cellEnergies.size(); ++i) {
    Int_t iEta;
    for (unsigned int j = 0; j < (cellEnergies[i]).size(); ++j) {
      iEta = HcalDetId(cellIds[i][j]).ieta();
      rawResp += (cellEnergies[i])[j];

      if (CALIB_METHOD == "L3_AND_MTRX_INV") {
        corResp += (solution[cellIds[i][j]] * cellEnergies[i][j]);
      } else if (CALIB_METHOD == "L3") {
        corResp += (solution[cellIds[i][j]] * (cellEnergies[i][j]));
      } else if (CALIB_METHOD == "MATRIX_INV_OF_ETA_AVE") {
        corResp += (iEtaCoefMap[iEta] * cellEnergies[i][j]);
      }

      if (maxIEta < abs(iEta))
        maxIEta = abs(iEta);
      if (minIEta > abs(iEta))
        minIEta = abs(iEta);
    }

    rawResp /= targetEnergies[i];
    corResp /= targetEnergies[i];

    // fill histograms based on iEta on ref point of the cluster (for now: hottest tower)
    // expect range |iEta|<=24 (to do: add flexibility for arbitrary range)

    if (CALIB_TYPE == "ISO_TRACK") {
      Int_t ind = refIEtaIPhi[i].first;
      (ind < 0) ? (ind += 24) : (ind += 23);
      if (ind >= 0 && ind < 48) {
        h1_corRespIEta[ind]->Fill(corResp);
      }

      // fill histograms for cases where all towers are in the barrel or endcap
      if (maxIEta < 25) {
        h1_rawResp->Fill(rawResp);
        h1_corResp->Fill(corResp);
      }
      if (maxIEta < 15) {
        h1_rawRespBarrel->Fill(rawResp);
        h1_corRespBarrel->Fill(corResp);
      } else if (maxIEta < 25 && minIEta > 16) {
        h1_rawRespEndcap->Fill(rawResp);
        h1_corRespEndcap->Fill(corResp);
      }
    }  // histograms for isotrack calibration

    else {
      // put jet plots here
    }

    rawResp = 0;
    corResp = 0;
    maxIEta = 0;
    minIEta = 999;
  }

  // save the histograms
  h1_trkP->Write();
  h1_allTrkP->Write();

  h1_selTrkP_iEta10->Write();

  h1_rawSumE->Write();
  h1_rawResp->Write();
  h1_corResp->Write();
  h1_rawRespBarrel->Write();
  h1_corRespBarrel->Write();
  h1_rawRespEndcap->Write();
  h1_corRespEndcap->Write();
  h1_numEventsTwrIEta->Write();
  h2_dHitRefBarrel->Write();
  h2_dHitRefEndcap->Write();
  for (Int_t i = 0; i < 48; ++i) {
    h1_corRespIEta[i]->Write();
  }

  histoFile->Write();
  histoFile->Close();

  edm::LogVerbatim("HcalCalib") << "\n Finished calibration.\n ";

}  // end of Terminate()

//**************************************************************************************************************

void hcalCalib::GetCoefFromMtrxInvOfAve() {
  // these maps are keyed by iEta
  std::map<Int_t, Float_t> aveTargetE;
  std::map<Int_t, Int_t> nEntries;  // count hits

  //  iEtaRef  iEtaCell, energy
  std::map<Int_t, std::map<Int_t, Float_t> > aveHitE;  // add energies in the loop, normalize after that

  for (unsigned int i = 0; i < cellEnergies.size(); ++i) {
    Int_t iEtaRef = refIEtaIPhi[i].first;
    aveTargetE[iEtaRef] += targetEnergies[i];
    nEntries[iEtaRef]++;

    // for hybrid method: matrix inv of averages preceeded by L3
    if (CALIB_METHOD == "L3_AND_MTRX_INV") {
      for (unsigned int j = 0; j < (cellEnergies[i]).size(); ++j) {
        aveHitE[iEtaRef][HcalDetId(cellIds[i][j]).ieta()] += (solution[cellIds[i][j]] * cellEnergies[i][j]);
      }
    } else if (CALIB_METHOD == "MATRIX_INV_OF_ETA_AVE") {
      for (unsigned int j = 0; j < (cellEnergies[i]).size(); ++j) {
        aveHitE[iEtaRef][HcalDetId(cellIds[i][j]).ieta()] += cellEnergies[i][j];
      }
    }

  }  // end of loop of entries

  // scale by number of entries to get the averages
  Float_t norm = 1.0;
  for (std::map<Int_t, Float_t>::iterator m_it = aveTargetE.begin(); m_it != aveTargetE.end(); ++m_it) {
    Int_t iEta = m_it->first;
    norm = (nEntries[iEta] > 0) ? 1.0 / (nEntries[iEta]) : 1.0;
    aveTargetE[iEta] *= norm;

    std::map<Int_t, Float_t>::iterator n_it = (aveHitE[iEta]).begin();

    for (; n_it != (aveHitE[iEta]).end(); ++n_it) {
      (n_it->second) *= norm;
    }

  }  // end of scaling by number of entries

  Int_t ONE_SIDE_IETA_RANGE = CALIB_ABS_IETA_MAX - CALIB_ABS_IETA_MIN + 1;

  // conversion from iEta to index for the linear system
  // contains elements only in the valid range for *matrix inversion*
  std::vector<Int_t> iEtaList;

  for (Int_t i = -CALIB_ABS_IETA_MAX; i <= CALIB_ABS_IETA_MAX; ++i) {
    if (abs(i) < CALIB_ABS_IETA_MIN)
      continue;
    iEtaList.push_back(i);
  }

  TMatrixD A(2 * ONE_SIDE_IETA_RANGE, 2 * ONE_SIDE_IETA_RANGE);
  TMatrixD b(2 * ONE_SIDE_IETA_RANGE, 1);
  TMatrixD x(2 * ONE_SIDE_IETA_RANGE, 1);

  for (Int_t i = 0; i < 2 * ONE_SIDE_IETA_RANGE; ++i) {
    for (Int_t j = 0; j < 2 * ONE_SIDE_IETA_RANGE; ++j) {
      A(i, j) = 0.0;
    }
  }

  for (UInt_t i = 0; i < iEtaList.size(); ++i) {
    b(i, 0) = aveTargetE[iEtaList[i]];

    std::map<Int_t, Float_t>::iterator n_it = aveHitE[iEtaList[i]].begin();
    for (; n_it != aveHitE[iEtaList[i]].end(); ++n_it) {
      if (fabs(n_it->first) > CALIB_ABS_IETA_MAX || fabs(n_it->first) < CALIB_ABS_IETA_MIN)
        continue;
      Int_t j = Int_t(find(iEtaList.begin(), iEtaList.end(), n_it->first) - iEtaList.begin());
      A(i, j) = n_it->second;
    }
  }

  TMatrixD coef = b;
  TDecompQRH qrh(A);
  Bool_t hasSolution = qrh.MultiSolve(coef);

  if (hasSolution && (CALIB_METHOD == "L3_AND_MTRX_INV" || CALIB_METHOD == "MATRIX_INV_OF_ETA_AVE")) {
    for (UInt_t i = 0; i < iEtaList.size(); ++i) {
      iEtaCoefMap[iEtaList[i]] = coef(i, 0);
    }
  }
}

Bool_t hcalCalib::ReadPhiSymCor() {
  std::ifstream phiSymFile(PHI_SYM_COR_FILENAME.Data());

  if (!phiSymFile) {
    edm::LogWarning("HcalCalib") << "\nERROR: Can not find file with phi symmetry constants \""
                                 << PHI_SYM_COR_FILENAME.Data() << "\"";
    return kFALSE;
  }

  // assumes the format used in CSA08, first line is a comment

  Int_t iEta;
  UInt_t iPhi;
  UInt_t depth;
  TString sdName;
  UInt_t detId;

  Float_t value;
  HcalSubdetector sd;

  std::string line;

  while (getline(phiSymFile, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream linestream(line);
    linestream >> iEta >> iPhi >> depth >> sdName >> value >> std::hex >> detId;

    if (sdName == "HB")
      sd = HcalBarrel;
    else if (sdName == "HE")
      sd = HcalEndcap;
    else if (sdName == "HO")
      sd = HcalOuter;
    else if (sdName == "HF")
      sd = HcalForward;
    else {
      edm::LogWarning("HcalCalib") << "\nInvalid detector name in phi symmetry constants file: " << sdName.Data()
                                   << "\nCheck file and rerun!\n";
      return kFALSE;
    }

    // check if the data is consistent

    if (HcalDetId(sd, iEta, iPhi, depth) != HcalDetId(detId)) {
      edm::LogWarning("HcalCalib")
          << "\nInconsistent info in phi symmetry file: subdet, iEta, iPhi, depth do not match rawId!\n";
      return kFALSE;
    }
    HcalDetId hId(detId);
    if (!topo_->valid(hId)) {
      edm::LogWarning("HcalCalib") << "\nInvalid DetId from: iEta=" << iEta << " iPhi=" << iPhi << " depth=" << depth
                                   << " subdet=" << sdName.Data() << " detId=" << detId << "\n";
      return kFALSE;
    }

    phiSymCor[HcalDetId(sd, iEta, iPhi, depth)] = value;
  }

  return kTRUE;
}

void hcalCalib::makeTextFile() {
  //{ HcalEmpty=0, HcalBarrel=1, HcalEndcap=2, HcalOuter=3, HcalForward=4, HcalTriggerTower=5, HcalOther=7 };

  TString sdName[8] = {"EMPTY", "HB", "HE", "HO", "HF", "TRITWR", "UNDEF", "OTHER"};

  FILE* constFile = fopen(OUTPUT_COR_COEF_FILENAME.Data(), "w+");

  // header of the constants file
  fprintf(constFile, "%1s%16s%16s%16s%16s%9s%11s\n", "#", "eta", "phi", "depth", "det", "value", "DetId");

  // Order loops to produce sequence of constants as for phi symmetry

  for (Int_t sd = 1; sd <= 4; sd++) {
    for (Int_t e = 1; e <= 50; e++) {
      Int_t eta;

      for (Int_t side = 0; side < 2; side++) {
        eta = (side == 0) ? -e : e;  //ta *= (-1);

        for (Int_t phi = 1; phi <= 72; phi++) {
          for (Int_t d = 1; d < 5; d++) {
            if (!topo_->valid(HcalDetId(HcalSubdetector(sd), eta, phi, d)))
              continue;
            HcalDetId id(HcalSubdetector(sd), eta, phi, d);
            Float_t corrFactor = 1.0;

            if (abs(eta) >= CALIB_ABS_IETA_MIN && abs(eta) <= CALIB_ABS_IETA_MAX && HcalSubdetector(sd) != HcalOuter) {
              //	    if (abs(eta)>=CALIB_ABS_IETA_MIN && abs(eta)<=22 && HcalSubdetector(sd)!=HcalOuter) {

              // need some care when depths were summed for iEta=16 =>
              // the coeficients are saved in depth 1 of HB: affects
              Int_t subdetInd = sd;

              if (abs(eta) == 16 && HcalSubdetector(sd) == HcalEndcap && SUM_DEPTHS) {
                subdetInd = 1;
              }

              if (CALIB_METHOD == "L3" || CALIB_METHOD == "L3_AND_MTRX_INV") {
                if (SUM_DEPTHS && COMBINE_PHI)
                  corrFactor = solution[HcalDetId(HcalSubdetector(subdetInd), eta, 1, 1)];
                else if (SUM_DEPTHS)
                  corrFactor = solution[HcalDetId(HcalSubdetector(subdetInd), eta, phi, 1)];
                else if (COMBINE_PHI)
                  corrFactor = solution[HcalDetId(HcalSubdetector(sd), eta, 1, d)];
                else
                  corrFactor = solution[HcalDetId(HcalSubdetector(sd), eta, phi, d)];

                // Remark: a new case was added (sumSmallDepths) where the first two depths in towers 15,16
                // are summed and stored in  depth 1.
                // For now we create the correction coef for depth 2 (set equal to depth 1)
                // after the call to the L3 minimizer so that this case is also handled without modifying the
                // logic above. Probably it is better to move it here?

              }  // L3

              else if (CALIB_METHOD == "MATRIX_INV_OF_ETA_AVE") {
                corrFactor = iEtaCoefMap[eta];
              }

              else if (CALIB_METHOD == "ISO_TRK_PHI_SYM") {
                corrFactor = solution[HcalDetId(HcalSubdetector(sd), eta, phi, d)];
              }

            }  // if within the calibration range

            fprintf(constFile, "%17i%16i%16i%16s%9.5f%11X\n", eta, phi, d, sdName[sd].Data(), corrFactor, id.rawId());
          }
        }
      }
    }
  }

  return;
}

inline void hcalCalib::Init(TTree* tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normaly not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  cells = nullptr;
  tagJetP4 = nullptr;
  probeJetP4 = nullptr;

  // Set branch addresses and branch pointers
  if (!tree)
    return;
  fChain = tree;

  //      fChain->SetMakeClass(1);

  fChain->SetBranchAddress("eventNumber", &eventNumber, &b_eventNumber);
  fChain->SetBranchAddress("runNumber", &runNumber, &b_runNumber);
  fChain->SetBranchAddress("iEtaHit", &iEtaHit, &b_iEtaHit);
  fChain->SetBranchAddress("iPhiHit", &iPhiHit, &b_iPhiHit);
  fChain->SetBranchAddress("cells", &cells, &b_cells);
  fChain->SetBranchAddress("emEnergy", &emEnergy, &b_emEnergy);
  fChain->SetBranchAddress("targetE", &targetE, &b_targetE);
  fChain->SetBranchAddress("etVetoJet", &etVetoJet, &b_etVetoJet);

  fChain->SetBranchAddress("xTrkHcal", &xTrkHcal, &b_xTrkHcal);
  fChain->SetBranchAddress("yTrkHcal", &yTrkHcal, &b_yTrkHcal);
  fChain->SetBranchAddress("zTrkHcal", &zTrkHcal, &b_zTrkHcal);
  fChain->SetBranchAddress("xTrkEcal", &xTrkEcal, &b_xTrkEcal);
  fChain->SetBranchAddress("yTrkEcal", &yTrkEcal, &b_yTrkEcal);
  fChain->SetBranchAddress("zTrkEcal", &zTrkEcal, &b_zTrkEcal);

  fChain->SetBranchAddress("tagJetEmFrac", &tagJetEmFrac, &b_tagJetEmFrac);
  fChain->SetBranchAddress("probeJetEmFrac", &probeJetEmFrac, &b_probeJetEmFrac);

  fChain->SetBranchAddress("tagJetP4", &tagJetP4, &b_tagJetP4);
  fChain->SetBranchAddress("probeJetP4", &probeJetP4, &b_probeJetP4);
}

inline Bool_t hcalCalib::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normaly not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}
