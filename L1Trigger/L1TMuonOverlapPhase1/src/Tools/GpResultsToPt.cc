/*
 * GpResultsToPt.cpp
 *
 *  Created on: Mar 6, 2020
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/GpResultsToPt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TH2F.h"
#include "TFile.h"

GpResultsToPt::GpResultsToPt(const std::vector<std::shared_ptr<GoldenPattern> >& gps,
                             const OMTFConfiguration* omtfConfig,
                             unsigned int lutSize)
    : lutSize(lutSize),
      gps(gps),
      omtfConfig(omtfConfig),
      gpResultsToPtLuts(gps.size(), std::vector<int>(lutSize)),
      gpResultsStatLuts(gps.size(), std::vector<double>(lutSize)),
      entries(gps.size(), std::vector<int>(lutSize)),
      lowerGps(gps.size(), nullptr),
      higerGps(gps.size(), nullptr) {
  for (auto gp : gps) {
    unsigned int iGP = gp->key().number();

    auto lowerGP = gp;
    for (int iLowerGp = iGP - 1; iLowerGp >= 0; iLowerGp--) {
      if (omtfConfig->getPatternPtRange(iLowerGp).ptTo == omtfConfig->getPatternPtRange(iGP).ptFrom &&
          gp->key().theCharge == gps.at(iLowerGp)->key().theCharge) {
        lowerGP = gps.at(iLowerGp);
        break;
      }
    }
    lowerGps[iGP] = lowerGP.get();

    auto higherGP = gp;
    for (int iHigherGp = iGP + 1; iHigherGp < (int)gps.size(); iHigherGp++) {
      if (omtfConfig->getPatternPtRange(iHigherGp).ptFrom == omtfConfig->getPatternPtRange(iGP).ptTo &&
          gp->key().theCharge == gps.at(iHigherGp)->key().theCharge) {
        higherGP = gps.at(iHigherGp);
        break;
      }
    }
    higerGps[iGP] = higherGP.get();

    OMTFConfiguration::PatternPt patternPt = omtfConfig->getPatternPtRange(gp->key().theNumber);
    if (gp->key().thePt == 0)
      continue;
    ostringstream ostrName;
    ostringstream ostrTtle;
    //cout<<__FUNCTION__<<": "<<__LINE__<<" "<<gp->key()<<std::endl;
    ostrName.str("");
    ostrName << "ptGenInPatNum_" << gp->key().theNumber;
    ostrTtle.str("");
    ostrTtle << "PatNum " << gp->key().theNumber << " ptCode " << gp->key().thePt << " Pt " << patternPt.ptFrom << "-"
             << patternPt.ptTo << " GeV"
             << " charge " << gp->key().theCharge;
    ptGenInPats.push_back(new TH1I(ostrName.str().c_str(), ostrTtle.str().c_str(), 200, -200, 200));
  }

  for (unsigned int iGP = 0; iGP < gps.size(); iGP++) {
    //LogTrace("l1tOmtfEventPrint") << " "<<gps.at(iGP)->key()<<" lower "<<lowerGps[iGP]->key() <<" higher "<<higerGps[iGP]->key()<< std::endl;
  }
}

GpResultsToPt::GpResultsToPt(const std::vector<std::shared_ptr<GoldenPattern> >& gps,
                             const OMTFConfiguration* omtfConfig)
    : gps(gps), omtfConfig(omtfConfig), lowerGps(gps.size(), nullptr), higerGps(gps.size(), nullptr) {
  for (auto gp : gps) {
    unsigned int iGP = gp->key().number();

    auto lowerGP = gp;
    for (int iLowerGp = iGP - 1; iLowerGp >= 0; iLowerGp--) {
      if (omtfConfig->getPatternPtRange(iLowerGp).ptTo == omtfConfig->getPatternPtRange(iGP).ptFrom &&
          gp->key().theCharge == gps.at(iLowerGp)->key().theCharge) {
        lowerGP = gps.at(iLowerGp);
        break;
      }
    }
    lowerGps[iGP] = lowerGP.get();

    auto higherGP = gp;
    for (int iHigherGp = iGP + 1; iHigherGp < (int)gps.size(); iHigherGp++) {
      if (omtfConfig->getPatternPtRange(iHigherGp).ptFrom == omtfConfig->getPatternPtRange(iGP).ptTo &&
          gp->key().theCharge == gps.at(iHigherGp)->key().theCharge) {
        higherGP = gps.at(iHigherGp);
        break;
      }
    }
    higerGps[iGP] = higherGP.get();
  }

  for (unsigned int iGP = 0; iGP < gps.size(); iGP++) {
    LogTrace("l1tOmtfEventPrint") << " " << gps.at(iGP)->key() << " lower " << lowerGps[iGP]->key() << " higher "
                                  << higerGps[iGP]->key() << std::endl;
  }
}

GpResultsToPt::~GpResultsToPt() {}

//maxPBin is the class index, not the ptBin index
unsigned int GpResultsToPt::lutAddres(AlgoMuons::value_type& algoMuon, unsigned int& candProcIndx) {
  unsigned int iGP = algoMuon->getGoldenPatern()->key().number();
  const unsigned int& iRefHit = algoMuon->getRefHitNumber();
  //unsigned int lutSizeBits = log2(lutSize);
  unsigned int bitShift = 1;

  unsigned int bins = sqrt(lutSize);

  unsigned int addressL = 0;
  unsigned int addressR = 0;
  LogTrace("l1tOmtfEventPrint") << " iGP " << iGP << " candProcIndx " << candProcIndx << " iRefHit "
                                << " " << iRefHit << " - lowerGps pdfSum "
                                << lowerGps[iGP]->getResults()[candProcIndx][iRefHit].getPdfSum() << " "
                                << lowerGps[iGP]->getResults()[candProcIndx][iRefHit].getFiredLayerCnt()
                                << " - algoMuon pdfSum "
                                << algoMuon->getGoldenPatern()->getResults()[candProcIndx][iRefHit].getPdfSum() << " "
                                << algoMuon->getGoldenPatern()->getResults()[candProcIndx][iRefHit].getFiredLayerCnt()
                                << " - higerGps pdfSum "
                                << higerGps[iGP]->getResults()[candProcIndx][iRefHit].getPdfSum() << " "
                                << higerGps[iGP]->getResults()[candProcIndx][iRefHit].getFiredLayerCnt();

  if (lowerGps[iGP]) {
    addressL = unsigned(algoMuon->getGoldenPatern()->getResults()[candProcIndx][iRefHit].getPdfSum() -
                        lowerGps[iGP]->getResults()[candProcIndx][iRefHit].getPdfSum()) >>
               bitShift;  //
  }
  if (higerGps[iGP])
    addressR = unsigned(algoMuon->getGoldenPatern()->getResults()[candProcIndx][iRefHit].getPdfSum() -
                        higerGps[iGP]->getResults()[candProcIndx][iRefHit].getPdfSum()) >>
               bitShift;  //

  if (addressL >= bins)
    addressL = bins - 1;

  if (addressR >= bins)
    addressR = bins - 1;

  unsigned int lutSizeBits = log2(lutSize);
  unsigned int address = ((addressL << lutSizeBits / 2) | addressR);

  LogTrace("l1tOmtfEventPrint") << " addressL " << addressL << " addressR " << addressR << " address " << address
                                << std::endl;

  return address;
}

int GpResultsToPt::getValue(AlgoMuons::value_type& algoMuon, unsigned int& candProcIndx) {
  unsigned int lutAdd = lutAddres(algoMuon, candProcIndx);
  return gpResultsToPtLuts.at(algoMuon->getGoldenPatern()->key().number()).at(lutAdd);
}

void GpResultsToPt::updateStat(AlgoMuons::value_type& algoMuon,
                               unsigned int& candProcIndx,
                               double ptSim,
                               int chargeSim) {
  unsigned int lutAdd = lutAddres(algoMuon, candProcIndx);

  auto gp = algoMuon->getGoldenPatern();
  unsigned int iGP = gp->key().number();

  LogTrace("l1tOmtfEventPrint") << " ptSim " << ptSim << " chargeSim " << chargeSim << " "
                                << algoMuon->getGoldenPatern()->key() << " ptFrom "
                                << omtfConfig->getPatternPtRange(lowerGps[iGP]->key().number()).ptFrom << " ptTo "
                                << omtfConfig->getPatternPtRange(higerGps[iGP]->key().number()).ptTo;

  if ((gp->key().theCharge == chargeSim) &&
      (omtfConfig->getPatternPtRange(lowerGps[iGP]->key().number()).ptFrom < ptSim) &&
      (ptSim < omtfConfig->getPatternPtRange(higerGps[iGP]->key().number()).ptTo)) {
    gpResultsStatLuts.at(algoMuon->getGoldenPatern()->key().number()).at(lutAdd) += ptSim;
    entries.at(algoMuon->getGoldenPatern()->key().number()).at(lutAdd) += 1;  //TODO add weight if needed

    LogTrace("l1tOmtfEventPrint") << " +++++ ";
  }

  LogTrace("l1tOmtfEventPrint") << "\n" << std::endl;

  ptGenInPats.at(iGP)->Fill(ptSim * chargeSim);
}

void GpResultsToPt::caluateLutValues() {
  TFile outfile("GpResultsToPt.root", "RECREATE");

  for (unsigned int iPtBin = 0; iPtBin < gpResultsStatLuts.size(); iPtBin++) {
    std::ostringstream name;
    name << "GpResultsToPt_iGP_" << iPtBin;
    unsigned int bins = sqrt(lutSize);
    //unsigned int lutSizeBits = log2(lutSize);
    unsigned int lutSizeSqrt = sqrt(lutSize);

    TH1F hist1D = TH1F(name.str().c_str(), name.str().c_str(), lutSize, 0, lutSize);
    name << "_2D";

    TH2F hist2D = TH2F(name.str().c_str(), name.str().c_str(), bins, 0, bins, bins, 0, bins);

    name << "_entries";
    TH2F hist2DEntries = TH2F(name.str().c_str(), name.str().c_str(), bins, 0, bins, bins, 0, bins);

    for (unsigned int i = 0; i < gpResultsStatLuts[iPtBin].size(); i++) {
      if (entries[iPtBin][i]) {
        gpResultsStatLuts[iPtBin][i] /= entries[iPtBin][i];

        gpResultsToPtLuts[iPtBin][i] = omtfConfig->ptGevToHw(gpResultsStatLuts[iPtBin][i]);

        hist2D.Fill(i / lutSizeSqrt, i % lutSizeSqrt, gpResultsStatLuts[iPtBin][i]);
        hist2DEntries.Fill(i / lutSizeSqrt, i % lutSizeSqrt, entries[iPtBin][i]);
        hist1D.Fill(i, gpResultsStatLuts[iPtBin][i]);
      }
    }

    ptGenInPats.at(iPtBin)->Write();
    hist1D.Write();
    hist2D.Write();
    hist2DEntries.Write();
  }
}
