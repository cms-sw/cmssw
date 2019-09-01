/**
 * \class L1GetHistLimits
 *
 *
 * Description: use L1 scales to define histogram limits for L1 trigger objects.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GetHistLimits.h"

// system include files
#include <iostream>
#include <iomanip>
#include <string>

// user include files
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1PhiConversion.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtMiss.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GetHistLimits::L1GetHistLimits(const edm::EventSetup& evSetup) : m_evSetup(evSetup) {
  //
}

// destructor
L1GetHistLimits::~L1GetHistLimits() {
  // empty
}

void L1GetHistLimits::getHistLimits(const L1GtObject& l1GtObject, const std::string& quantity) {
  m_l1HistLimits.nrBins = 0;
  m_l1HistLimits.lowerBinValue = 0;
  m_l1HistLimits.upperBinValue = 0;
  m_l1HistLimits.binThresholds.clear();

  // number of objects is independent of the object type
  // and hardcoded in the actual version
  if (quantity == "NrObjects") {
    m_l1HistLimits.nrBins = 15;
    m_l1HistLimits.lowerBinValue = -0.5;
    m_l1HistLimits.upperBinValue = 14.5;

    m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

    for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
      m_l1HistLimits.binThresholds[iBin] =
          m_l1HistLimits.lowerBinValue +
          iBin * (m_l1HistLimits.upperBinValue - m_l1HistLimits.lowerBinValue) / m_l1HistLimits.nrBins;
    }

    // set last bin upper edge
    m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

    return;
  }

  switch (l1GtObject) {
    case Mu: {
      if (quantity == "PT") {
        edm::ESHandle<L1MuTriggerPtScale> muPtScale;
        m_evSetup.get<L1MuTriggerPtScaleRcd>().get(muPtScale);

        m_l1HistLimits.nrBins = muPtScale->getPtScale()->getNBins();
        m_l1HistLimits.lowerBinValue = muPtScale->getPtScale()->getScaleMin();
        m_l1HistLimits.upperBinValue = muPtScale->getPtScale()->getScaleMax();

        m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

        for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
          m_l1HistLimits.binThresholds[iBin] = muPtScale->getPtScale()->getValue(iBin);
        }

        // last limit for muon is set too high (10^6) - resize the last bin

        float lastBinSize = m_l1HistLimits.upperBinValue - m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins - 1];

        // limit the size of the last bin to maximum 200 GeV
        float maxLaxtBinsize = 200;
        if (lastBinSize >= maxLaxtBinsize) {
          m_l1HistLimits.upperBinValue = m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins - 1] +
                                         2. * (m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins - 1] -
                                               m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins - 2]);
          LogDebug("L1GetHistLimits") << "\n L1ExtraMuon: PT histogram"
                                      << "\nm_l1HistLimits.upperBinValue truncated to = "
                                      << m_l1HistLimits.upperBinValue << std::endl;
        }

        // set last bin upper edge
        m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

      } else if (quantity == "eta" || quantity == "phi") {
        edm::ESHandle<L1MuTriggerScales> muScales;
        m_evSetup.get<L1MuTriggerScalesRcd>().get(muScales);

        if (quantity == "eta") {
          // eta scale defined for positive values - need to be symmetrized
          int histNrBinsHalf = muScales->getGMTEtaScale()->getNBins();
          m_l1HistLimits.lowerBinValue = muScales->getGMTEtaScale()->getScaleMin();
          m_l1HistLimits.upperBinValue = muScales->getGMTEtaScale()->getScaleMax();

          m_l1HistLimits.nrBins = 2 * histNrBinsHalf;
          m_l1HistLimits.lowerBinValue = -m_l1HistLimits.upperBinValue;

          m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

          int iBin = 0;
          for (int j = histNrBinsHalf; j > 0; j--, iBin++) {
            m_l1HistLimits.binThresholds[iBin] = (-1) * muScales->getGMTEtaScale()->getValue(j);
          }
          for (int j = 0; j <= histNrBinsHalf; j++, iBin++) {
            m_l1HistLimits.binThresholds[iBin] = muScales->getGMTEtaScale()->getValue(j);
          }

          // set last bin upper edge
          m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

        } else {
          m_l1HistLimits.nrBins = muScales->getPhiScale()->getNBins();
          m_l1HistLimits.lowerBinValue = rad2deg(muScales->getPhiScale()->getScaleMin());
          m_l1HistLimits.upperBinValue = rad2deg(muScales->getPhiScale()->getScaleMax());

          m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

          for (int iBin = 0; iBin <= m_l1HistLimits.nrBins; iBin++) {
            m_l1HistLimits.binThresholds[iBin] = rad2deg(muScales->getPhiScale()->getValue(iBin));
          }

          // set last bin upper edge
          m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;
        }
      }

    } break;
    case NoIsoEG:
    case IsoEG: {
      // common scales for NoIsoEG and IsoEG
      if (quantity == "ET") {
        edm::ESHandle<L1CaloEtScale> emScale;
        m_evSetup.get<L1EmEtScaleRcd>().get(emScale);

        std::vector<double> emThresholds = emScale->getThresholds();
        m_l1HistLimits.nrBins = emThresholds.size();
        m_l1HistLimits.lowerBinValue = emThresholds.at(0);

        // FIXME high edge retrieval in the scale definition
        // now, last bin has the same width like the last but one
        m_l1HistLimits.upperBinValue =
            emThresholds[m_l1HistLimits.nrBins - 1] +
            (emThresholds[m_l1HistLimits.nrBins - 1] - emThresholds[m_l1HistLimits.nrBins - 2]);

        m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

        for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
          m_l1HistLimits.binThresholds[iBin] = static_cast<float>(emThresholds[iBin]);
        }

        // set last bin upper edge
        m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

      } else if (quantity == "eta" || quantity == "phi") {
        edm::ESHandle<L1CaloGeometry> caloGeomESH;
        m_evSetup.get<L1CaloGeometryRecord>().get(caloGeomESH);
        const L1CaloGeometry* caloGeomScales = caloGeomESH.product();

        if (quantity == "eta") {
          m_l1HistLimits.nrBins =
              2 * (caloGeomScales->numberGctCentralEtaBinsPerHalf() + caloGeomScales->numberGctForwardEtaBinsPerHalf());
          m_l1HistLimits.lowerBinValue = caloGeomScales->globalEtaBinLowEdge(0);
          m_l1HistLimits.upperBinValue = caloGeomScales->globalEtaBinHighEdge(m_l1HistLimits.nrBins - 1);

          m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

          for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
            m_l1HistLimits.binThresholds[iBin] = caloGeomScales->globalEtaBinLowEdge(iBin);
          }

          // set last bin upper edge
          m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

        } else {
          m_l1HistLimits.nrBins = caloGeomScales->numberGctEmJetPhiBins();
          m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

          for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
            m_l1HistLimits.binThresholds[iBin] = rad2deg(caloGeomScales->emJetPhiBinLowEdge(iBin));

            // treat correctly the 10 deg anti-clockwise rotation
            if (rad2deg(caloGeomScales->emJetPhiBinHighEdge(iBin)) < m_l1HistLimits.binThresholds[iBin]) {
              m_l1HistLimits.binThresholds[iBin] = m_l1HistLimits.binThresholds[iBin] - 360.;
            }
          }

          m_l1HistLimits.lowerBinValue = m_l1HistLimits.binThresholds[0];

          // last bin upper limit
          m_l1HistLimits.upperBinValue = rad2deg(caloGeomScales->emJetPhiBinHighEdge(m_l1HistLimits.nrBins - 1));

          m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;
        }
      }

    } break;
    case CenJet:
    case ForJet:
    case TauJet: {
      // common scales for all jets
      if (quantity == "ET") {
        edm::ESHandle<L1CaloEtScale> jetScale;
        m_evSetup.get<L1JetEtScaleRcd>().get(jetScale);

        std::vector<double> jetThresholds = jetScale->getThresholds();
        m_l1HistLimits.nrBins = jetThresholds.size();
        m_l1HistLimits.lowerBinValue = jetThresholds.at(0);

        // FIXME high edge retrieval in the scale definition
        // now, last bin has the same width like the last but one
        m_l1HistLimits.upperBinValue =
            jetThresholds[m_l1HistLimits.nrBins - 1] +
            (jetThresholds[m_l1HistLimits.nrBins - 1] - jetThresholds[m_l1HistLimits.nrBins - 2]);

        m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

        for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
          m_l1HistLimits.binThresholds[iBin] = static_cast<float>(jetThresholds[iBin]);
        }

        // set last bin upper edge
        m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

      } else if (quantity == "eta" || quantity == "phi") {
        edm::ESHandle<L1CaloGeometry> caloGeomESH;
        m_evSetup.get<L1CaloGeometryRecord>().get(caloGeomESH);
        const L1CaloGeometry* caloGeomScales = caloGeomESH.product();

        if (quantity == "eta") {
          m_l1HistLimits.nrBins =
              2 * (caloGeomScales->numberGctCentralEtaBinsPerHalf() + caloGeomScales->numberGctForwardEtaBinsPerHalf());
          m_l1HistLimits.lowerBinValue = caloGeomScales->globalEtaBinLowEdge(0);
          m_l1HistLimits.upperBinValue = caloGeomScales->globalEtaBinHighEdge(m_l1HistLimits.nrBins - 1);

          m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

          for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
            m_l1HistLimits.binThresholds[iBin] = caloGeomScales->globalEtaBinLowEdge(iBin);
          }

          // set last bin upper edge
          m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

        } else {
          m_l1HistLimits.nrBins = caloGeomScales->numberGctEmJetPhiBins();
          m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

          for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
            m_l1HistLimits.binThresholds[iBin] = rad2deg(caloGeomScales->emJetPhiBinLowEdge(iBin));

            // treat correctly the 10 deg anti-clockwise rotation
            if (rad2deg(caloGeomScales->emJetPhiBinHighEdge(iBin)) < m_l1HistLimits.binThresholds[iBin]) {
              m_l1HistLimits.binThresholds[iBin] = m_l1HistLimits.binThresholds[iBin] - 360.;
            }
          }

          m_l1HistLimits.lowerBinValue = m_l1HistLimits.binThresholds[0];

          // last bin upper limit
          m_l1HistLimits.upperBinValue = rad2deg(caloGeomScales->emJetPhiBinHighEdge(m_l1HistLimits.nrBins - 1));

          m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;
        }
      }
    } break;
    case ETM: {
      if (quantity == "ET") {
        edm::ESHandle<L1CaloEtScale> etMissScale;
        m_evSetup.get<L1JetEtScaleRcd>().get(etMissScale);

        const double etSumLSB = etMissScale->linearLsb();

        m_l1HistLimits.nrBins = L1GctEtMiss::kEtMissMaxValue;

        m_l1HistLimits.lowerBinValue = 0;
        m_l1HistLimits.upperBinValue = (m_l1HistLimits.nrBins + 1) * etSumLSB;

        m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

        for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
          m_l1HistLimits.binThresholds[iBin] = iBin * etSumLSB;
        }

        // set last bin upper edge
        m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

      } else if (quantity == "eta" || quantity == "phi") {
        edm::ESHandle<L1CaloGeometry> caloGeomESH;
        m_evSetup.get<L1CaloGeometryRecord>().get(caloGeomESH);
        const L1CaloGeometry* caloGeomScales = caloGeomESH.product();

        if (quantity == "eta") {
          // do nothing, eta is not defined for ETM

        } else {
          m_l1HistLimits.nrBins = caloGeomScales->numberGctEtSumPhiBins();
          m_l1HistLimits.lowerBinValue = rad2deg(caloGeomScales->etSumPhiBinLowEdge(0));
          m_l1HistLimits.upperBinValue = rad2deg(caloGeomScales->etSumPhiBinHighEdge(m_l1HistLimits.nrBins - 1));

          m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

          for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
            m_l1HistLimits.binThresholds[iBin] = rad2deg(caloGeomScales->etSumPhiBinLowEdge(iBin));
          }

          // last bin upper limit
          m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;
        }
      }

    } break;
    case ETT: {
      if (quantity == "ET") {
        edm::ESHandle<L1CaloEtScale> etMissScale;
        m_evSetup.get<L1JetEtScaleRcd>().get(etMissScale);

        const double etSumLSB = etMissScale->linearLsb();

        m_l1HistLimits.nrBins = L1GctEtTotal::kEtTotalMaxValue;

        m_l1HistLimits.lowerBinValue = 0;
        m_l1HistLimits.upperBinValue = (m_l1HistLimits.nrBins + 1) * etSumLSB;

        m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

        for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
          m_l1HistLimits.binThresholds[iBin] = iBin * etSumLSB;
        }

        // set last bin upper edge
        m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

      } else if (quantity == "eta" || quantity == "phi") {
        // do nothing, eta and phi are not defined for ETT
      }

    } break;
    case HTT: {
      if (quantity == "ET") {
        edm::ESHandle<L1GctJetFinderParams> jetFinderParams;
        m_evSetup.get<L1GctJetFinderParamsRcd>().get(jetFinderParams);
        double htSumLSB = jetFinderParams->getHtLsbGeV();

        m_l1HistLimits.nrBins = L1GctEtHad::kEtHadMaxValue;

        m_l1HistLimits.lowerBinValue = 0;
        m_l1HistLimits.upperBinValue = (m_l1HistLimits.nrBins + 1) * htSumLSB;

        m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

        for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
          m_l1HistLimits.binThresholds[iBin] = iBin * htSumLSB;
        }

        // set last bin upper edge
        m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

      } else if (quantity == "eta" || quantity == "phi") {
        // do nothing, eta and phi are not defined for HTT
      }
    } break;
    case HTM: {
      if (quantity == "ET") {
        edm::ESHandle<L1CaloEtScale> htMissScale;
        m_evSetup.get<L1HtMissScaleRcd>().get(htMissScale);

        const std::vector<double>& htThresholds = htMissScale->getThresholds();
        m_l1HistLimits.nrBins = htThresholds.size();
        m_l1HistLimits.lowerBinValue = htThresholds[0];

        // FIXME high edge retrieval in the scale definition
        // now, last bin has the same width like the last but one
        m_l1HistLimits.upperBinValue =
            htThresholds[m_l1HistLimits.nrBins - 1] +
            (htThresholds[m_l1HistLimits.nrBins - 1] - htThresholds[m_l1HistLimits.nrBins - 2]);

        m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

        for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
          m_l1HistLimits.binThresholds[iBin] = static_cast<float>(htThresholds[iBin]);
        }

        // set last bin upper edge
        m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

      } else if (quantity == "eta" || quantity == "phi") {
        edm::ESHandle<L1CaloGeometry> caloGeomESH;
        m_evSetup.get<L1CaloGeometryRecord>().get(caloGeomESH);
        const L1CaloGeometry* caloGeomScales = caloGeomESH.product();

        if (quantity == "eta") {
          // do nothing, eta is not defined for HTM

        } else {
          m_l1HistLimits.nrBins = caloGeomScales->numberGctHtSumPhiBins();
          m_l1HistLimits.lowerBinValue = rad2deg(caloGeomScales->htSumPhiBinLowEdge(0));
          m_l1HistLimits.upperBinValue = rad2deg(caloGeomScales->htSumPhiBinHighEdge(m_l1HistLimits.nrBins - 1));

          m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

          for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
            m_l1HistLimits.binThresholds[iBin] = rad2deg(caloGeomScales->htSumPhiBinLowEdge(iBin));
          }

          // last bin upper limit
          m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;
        }
      }
    } break;
    case JetCounts: {
    } break;
    case HfBitCounts: {
      // there are no scales for HfBitCounts, so one implements a fixed scale
      // use same values as GCT for 3 bits
      const unsigned int R3BINS = 8;
      const float R3MIN = -0.5;
      const float R3MAX = 7.5;

      m_l1HistLimits.nrBins = R3BINS;
      m_l1HistLimits.lowerBinValue = R3MIN;
      m_l1HistLimits.upperBinValue = R3MAX;

      m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

      for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
        m_l1HistLimits.binThresholds[iBin] = R3MIN + iBin * (R3MAX - R3MIN) / R3BINS;
      }

      // last bin upper limit
      m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

    } break;
    case HfRingEtSums: {
      if (quantity == "ET") {
        edm::ESHandle<L1CaloEtScale> hfRingEtScale;
        m_evSetup.get<L1HfRingEtScaleRcd>().get(hfRingEtScale);

        const std::vector<double>& hfRingEtThresholds = hfRingEtScale->getThresholds();
        m_l1HistLimits.nrBins = hfRingEtThresholds.size();
        m_l1HistLimits.lowerBinValue = hfRingEtThresholds[0];

        // FIXME high edge retrieval in the scale definition
        // now, last bin has the same width like the last but one
        m_l1HistLimits.upperBinValue =
            hfRingEtThresholds[m_l1HistLimits.nrBins - 1] +
            (hfRingEtThresholds[m_l1HistLimits.nrBins - 1] - hfRingEtThresholds[m_l1HistLimits.nrBins - 2]);

        m_l1HistLimits.binThresholds.resize(m_l1HistLimits.nrBins + 1);

        for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
          m_l1HistLimits.binThresholds[iBin] = static_cast<float>(hfRingEtThresholds[iBin]);
        }

        // set last bin upper edge
        m_l1HistLimits.binThresholds[m_l1HistLimits.nrBins] = m_l1HistLimits.upperBinValue;

      } else if (quantity == "eta" || quantity == "phi") {
        // do nothing, eta and phi are not defined for HfRingEtSums
      }
    } break;
    case TechTrig:
    case Castor:
    case BPTX:
    default: {
      // do nothing, for these cases ET/PT, eta and phi are not defined

    } break;
  }
}

const L1GetHistLimits::L1HistLimits& L1GetHistLimits::l1HistLimits(const L1GtObject& l1GtObject,
                                                                   const std::string& quantity) {
  getHistLimits(l1GtObject, quantity);

  if (edm::isDebugEnabled()) {
    LogDebug("L1GetHistLimits") << "\n Histogram limits for L1GtObject " << l1GtObject << " and quantity " << quantity
                                << "\n  Number of bins:           " << m_l1HistLimits.nrBins
                                << "\n  Lower limit of first bin: " << m_l1HistLimits.lowerBinValue
                                << "\n  Upper limit of last bin:  " << m_l1HistLimits.upperBinValue << std::endl;

    int binThreshSize = static_cast<int>(m_l1HistLimits.binThresholds.size());

    if (binThreshSize != (m_l1HistLimits.nrBins + 1)) {
      LogTrace("L1GetHistLimits") << "\n Warning: inconsistent nrBins and binThresholds size"
                                  << "\n   Number of bins nrBins = " << m_l1HistLimits.nrBins
                                  << "\n   binThresholds size =    " << binThreshSize
                                  << "\n Please fix the L1GetLimits class.\n\n"
                                  << std::endl;
    }

    for (int iBin = 0; iBin < binThreshSize; ++iBin) {
      LogTrace("L1GetHistLimits") << " Bin " << std::right << std::setw(5) << iBin << ":  "
                                  << m_l1HistLimits.binThresholds[iBin] << std::endl;
    }
  }

  return m_l1HistLimits;
}

const L1GetHistLimits::L1HistLimits& L1GetHistLimits::l1HistLimits(const L1GtObject& l1GtObject,
                                                                   const std::string& quantity,
                                                                   const double histMinValue,
                                                                   const double histMaxValue) {
  getHistLimits(l1GtObject, quantity);

  bool foundLowerBinValue = false;
  bool foundUpperBinValue = false;

  for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
    if (m_l1HistLimits.binThresholds[iBin] <= histMinValue) {
      m_l1HistLimits.lowerBinValue = m_l1HistLimits.binThresholds[iBin];
      foundLowerBinValue = true;
      break;
    }
  }

  for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
    if (m_l1HistLimits.binThresholds[iBin] > histMaxValue) {
      m_l1HistLimits.upperBinValue = m_l1HistLimits.binThresholds[iBin];
      foundUpperBinValue = true;
      break;
    }
  }

  if (foundLowerBinValue && foundUpperBinValue) {
    int countBins = -1;
    std::vector<float> binThresh;
    binThresh.reserve(m_l1HistLimits.binThresholds.size());

    for (int iBin = 0; iBin < m_l1HistLimits.nrBins; ++iBin) {
      if ((m_l1HistLimits.binThresholds[iBin] >= histMinValue) && m_l1HistLimits.binThresholds[iBin] < histMaxValue) {
        m_l1HistLimits.upperBinValue = m_l1HistLimits.binThresholds[iBin];

        countBins++;
        binThresh.push_back(m_l1HistLimits.binThresholds[iBin]);
      }
    }

    m_l1HistLimits.nrBins = countBins;
    m_l1HistLimits.binThresholds.clear();
    m_l1HistLimits.binThresholds = binThresh;

    // FIXME last bin untested.

  } else {
    m_l1HistLimits.nrBins = 0;
    m_l1HistLimits.lowerBinValue = 0;
    m_l1HistLimits.upperBinValue = 0;
    m_l1HistLimits.binThresholds.clear();

    LogDebug("L1GetHistLimits") << "\n Histogram limits for L1GtObject" << l1GtObject << " and quantity " << quantity
                                << " within the required range [" << histMinValue << ", " << histMaxValue
                                << "] not found."
                                << "\n The range is not included in the original histogram range." << std::endl;

    return m_l1HistLimits;
  }

  if (edm::isDebugEnabled()) {
    LogDebug("L1GetHistLimits") << "\n Histogram limits for L1GtObject" << l1GtObject << " and quantity " << quantity
                                << "\n  Number of bins:           " << m_l1HistLimits.nrBins
                                << "\n  Lower limit of first bin: " << m_l1HistLimits.lowerBinValue
                                << "\n  Upper limit of last bin:  " << m_l1HistLimits.upperBinValue << std::endl;

    int binThreshSize = static_cast<int>(m_l1HistLimits.binThresholds.size());

    if (binThreshSize != (m_l1HistLimits.nrBins + 1)) {
      LogTrace("L1GetHistLimits") << "\n Warning: inconsistent nrBins and binThresholds size"
                                  << "\n   Number of bins nrBins = " << m_l1HistLimits.nrBins
                                  << "\n   binThresholds size =    " << binThreshSize
                                  << "\n Please fix the L1GetLimits class.\n\n"
                                  << std::endl;
    }

    for (int iBin = 0; iBin < binThreshSize; ++iBin) {
      LogTrace("L1GetHistLimits") << " Bin " << std::right << std::setw(5) << iBin << ":  "
                                  << m_l1HistLimits.binThresholds[iBin] << std::endl;
    }
  }

  return m_l1HistLimits;
}

const int L1GetHistLimits::l1HistNrBins(const L1GtObject& l1GtObject, const std::string& quantity) {
  getHistLimits(l1GtObject, quantity);
  return m_l1HistLimits.nrBins;
}

const double L1GetHistLimits::l1HistLowerBinValue(const L1GtObject& l1GtObject, const std::string& quantity) {
  getHistLimits(l1GtObject, quantity);
  return m_l1HistLimits.lowerBinValue;
}

const double L1GetHistLimits::l1HistUpperBinValue(const L1GtObject& l1GtObject, const std::string& quantity) {
  getHistLimits(l1GtObject, quantity);
  return m_l1HistLimits.upperBinValue;
}

const std::vector<float>& L1GetHistLimits::l1HistBinThresholds(const L1GtObject& l1GtObject,
                                                               const std::string& quantity) {
  getHistLimits(l1GtObject, quantity);
  return m_l1HistLimits.binThresholds;
}
