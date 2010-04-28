#ifndef DQM_L1TMonitor_L1ExtraDQM_h
#define DQM_L1TMonitor_L1ExtraDQM_h

/**
 * \class L1ExtraDQM
 *
 *
 * Description: online DQM module for L1Extra trigger objects.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <iosfwd>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

// user include files
//   base classes
#include "FWCore/Framework/interface/EDAnalyzer.h"

//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

// L1Extra objects
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1PhiConversion.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GetHistLimits.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1RetrieveL1Extra.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "boost/lexical_cast.hpp"

// forward declarations


// class declaration
class L1ExtraDQM: public edm::EDAnalyzer {

public:

    // constructor(s)
    explicit L1ExtraDQM(const edm::ParameterSet&);

    // destructor
    virtual ~L1ExtraDQM();

public:

    template<class CollectionType>
    class L1ExtraMonElement {

    public:
        // constructor
        L1ExtraMonElement(const edm::EventSetup&, const int);

        // destructor
        virtual ~L1ExtraMonElement();

    public:
        typedef typename CollectionType::const_iterator CIterColl;

        void bookHistograms(const edm::EventSetup& evSetup, DQMStore* dbe,
                const std::string& l1ExtraObject,
                const std::vector<L1GtObject>& l1GtObj, const bool bookPhi =
                        true, const bool bookEta = true);

        /// number of objects
        void fillNrObjects(const CollectionType* collType,
                const bool validColl, const bool isL1Coll, const int bxInEvent);

        /// PT, eta, phi
        void fillPtPhiEta(const CollectionType* collType, const bool validColl,
                const bool bookPhi, const bool bookEta, const bool isL1Coll,
                const int bxInEvent);

        /// ET, eta, phi
        void fillEtPhiEta(const CollectionType* collType, const bool validColl,
                const bool bookPhi, const bool bookEta, const bool isL1Coll,
                const int bxInEvent);

        /// fill ET total in energy sums
        void fillEtTotal(const CollectionType* collType, const bool validColl,
                const bool isL1Coll, const int bxInEvent);

        /// fill charge
        void fillCharge(const CollectionType* collType, const bool validColl,
                const bool isL1Coll, const int bxInEvent);

        /// fill bit counts in HFRings collections
        void fillHfBitCounts(const CollectionType* collType,
                const bool validColl, const int countIndex,
                const bool isL1Coll, const int bxInEvent);

        /// fill energy sums in HFRings collections
        void fillHfRingEtSums(const CollectionType* collType,
                const bool validColl, const int countIndex,
                const bool isL1Coll, const int bxInEvent);

    private:

        std::vector<MonitorElement*> m_monElement;

        /// histogram index for each quantity, set during histogram booking
        int m_indexNrObjects;
        int m_indexPt;
        int m_indexEt;
        int m_indexPhi;
        int m_indexEta;
        int m_indexEtTotal;
        int m_indexCharge;
        int m_indexHfBitCounts;
        int m_indexHfRingEtSums;

    };

private:

    void analyzeL1ExtraMuon(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraIsoEG(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraNoIsoEG(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraCenJet(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraForJet(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraTauJet(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraETT(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraETM(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraHTT(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraHTM(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraHfBitCounts(const edm::Event&, const edm::EventSetup&);
    void analyzeL1ExtraHfRingEtSums(const edm::Event&, const edm::EventSetup&);

    virtual void beginJob();
    void beginRun(const edm::Run&, const edm::EventSetup&);

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    void endRun(const edm::Run&, const edm::EventSetup&);
    virtual void endJob();

private:

    /// input parameters

    L1RetrieveL1Extra m_retrieveL1Extra;

    /// directory name for L1Extra plots
    std::string m_dirName;

    /// number of bunch crosses in event to be monitored
    int m_nrBxInEventGmt;
    int m_nrBxInEventGct;

    /// internal members

    DQMStore* m_dbe;

    bool m_resetModule;
    int m_currentRun;

    ///
    int m_nrEvJob;
    int m_nrEvRun;


private:

    /// pointers to L1ExtraMonElement for each sub-analysis

    std::vector<L1ExtraMonElement<l1extra::L1MuonParticleCollection>*>
            m_meAnalysisL1ExtraMuon;

    std::vector<L1ExtraMonElement<l1extra::L1EmParticleCollection>*>
            m_meAnalysisL1ExtraIsoEG;
    std::vector<L1ExtraMonElement<l1extra::L1EmParticleCollection>*>
            m_meAnalysisL1ExtraNoIsoEG;

    std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>
            m_meAnalysisL1ExtraCenJet;
    std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>
            m_meAnalysisL1ExtraForJet;
    std::vector<L1ExtraMonElement<l1extra::L1JetParticleCollection>*>
            m_meAnalysisL1ExtraTauJet;

    std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>
            m_meAnalysisL1ExtraETT;

    std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>
            m_meAnalysisL1ExtraETM;

    std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>
            m_meAnalysisL1ExtraHTT;

    std::vector<L1ExtraMonElement<l1extra::L1EtMissParticleCollection>*>
            m_meAnalysisL1ExtraHTM;

    std::vector<L1ExtraMonElement<l1extra::L1HFRingsCollection>*>
            m_meAnalysisL1ExtraHfBitCounts;

    std::vector<L1ExtraMonElement<l1extra::L1HFRingsCollection>*>
            m_meAnalysisL1ExtraHfRingEtSums;

};

// constructor L1ExtraMonElement
template<class CollectionType>
L1ExtraDQM::L1ExtraMonElement<CollectionType>::L1ExtraMonElement(
        const edm::EventSetup& evSetup, const int nrElements) :
    m_indexNrObjects(-1),
    m_indexPt(-1),
    m_indexEt(-1),
    m_indexPhi(-1),
    m_indexEta(-1),
    m_indexEtTotal(-1),
    m_indexCharge(-1),
    m_indexHfBitCounts(-1),
    m_indexHfRingEtSums(-1) {

    m_monElement.reserve(nrElements);

}

// destructor L1ExtraMonElement
template<class CollectionType>
L1ExtraDQM::L1ExtraMonElement<CollectionType>::~L1ExtraMonElement() {

    //empty

}


template<class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::bookHistograms(
        const edm::EventSetup& evSetup, DQMStore* dbe,
        const std::string& l1ExtraObject,
        const std::vector<L1GtObject>& l1GtObj, const bool bookPhi,
        const bool bookEta) {

    // FIXME
    L1GtObject gtObj = l1GtObj.at(0);

    //
    std::string histName;
    std::string histTitle;
    std::string xAxisTitle;
    std::string yAxisTitle;

    std::string quantity = "";

    int indexHistogram = -1;

    if (gtObj == HfBitCounts) {

        L1GetHistLimits l1GetHistLimits(evSetup);
        const L1GetHistLimits::L1HistLimits& histLimits =
                l1GetHistLimits.l1HistLimits(gtObj, quantity);

        const int histNrBins = histLimits.nrBins;
        const double histMinValue = histLimits.lowerBinValue;
        const double histMaxValue = histLimits.upperBinValue;

        indexHistogram++;
        m_indexHfBitCounts = indexHistogram;

        for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {

            histName = l1ExtraObject + "_Count_" + boost::lexical_cast<
                    std::string>(iCount);
            histTitle = l1ExtraObject + ": count " + boost::lexical_cast<
                    std::string>(iCount);
            xAxisTitle = l1ExtraObject;
            yAxisTitle = "Entries";

            m_monElement.push_back(dbe->book1D(histName, histTitle, histNrBins,
                    histMinValue, histMaxValue));
            m_monElement[m_indexHfBitCounts + iCount]->setAxisTitle(xAxisTitle,
                    1);
            m_monElement[m_indexHfBitCounts + iCount]->setAxisTitle(yAxisTitle,
                    2);

        }

        return;

    }

    // number of objects per event
    if ((gtObj == Mu) || (gtObj == IsoEG) || (gtObj == NoIsoEG) || (gtObj
            == CenJet) || (gtObj == ForJet) || (gtObj == TauJet)) {

        quantity = "NrObjects";

        L1GetHistLimits l1GetHistLimits(evSetup);
        const L1GetHistLimits::L1HistLimits& histLimits =
                l1GetHistLimits.l1HistLimits(gtObj, quantity);

        const int histNrBins = histLimits.nrBins;
        const double histMinValue = histLimits.lowerBinValue;
        const double histMaxValue = histLimits.upperBinValue;

        histName = l1ExtraObject + "_NrObjectsPerEvent";
        histTitle = l1ExtraObject + ": number of objects per event";
        xAxisTitle = "Nr_" + l1ExtraObject;
        yAxisTitle = "Entries";

        m_monElement.push_back(dbe->book1D(histName, histTitle, histNrBins,
                histMinValue, histMaxValue));
        indexHistogram++;

        m_monElement[indexHistogram]->setAxisTitle(xAxisTitle, 1);
        m_monElement[indexHistogram]->setAxisTitle(yAxisTitle, 2);
        m_indexNrObjects = indexHistogram;

    }

    // transverse momentum (energy)  PT (ET) [GeV]


    quantity = "ET";
    std::string quantityLongName = " transverse energy ";

    if (gtObj == Mu) {
        quantity = "PT";
        quantityLongName = " transverse momentum ";
    }

    L1GetHistLimits l1GetHistLimits(evSetup);
    const L1GetHistLimits::L1HistLimits& histLimits =
            l1GetHistLimits.l1HistLimits(gtObj, quantity);

    const int histNrBinsET = histLimits.nrBins;
    const double histMinValueET = histLimits.lowerBinValue;
    const double histMaxValueET = histLimits.upperBinValue;
    const std::vector<float>& binThresholdsET = histLimits.binThresholds;

    float* binThresholdsETf;
    size_t sizeBinThresholdsET = binThresholdsET.size();
    binThresholdsETf = new float[sizeBinThresholdsET];
    copy(binThresholdsET.begin(), binThresholdsET.end(), binThresholdsETf);

    LogDebug("L1ExtraDQM") << "\n PT/ET histogram for " << l1ExtraObject
            << "\n histNrBinsET = " << histNrBinsET << "\n histMinValueET = "
            << histMinValueET << "\n histMaxValueET = " << histMaxValueET
            << "\n Last bin value represents the upper limit of the histogram"
            << std::endl;
    for (size_t iBin = 0; iBin < sizeBinThresholdsET; ++iBin) {
        LogTrace("L1ExtraDQM") << "Bin " << iBin << ": " << quantity << " = "
                << binThresholdsETf[iBin] << " GeV" << std::endl;

    }

    histName = l1ExtraObject + "_" + quantity;
    histTitle = l1ExtraObject + ":" + quantityLongName + quantity + " [GeV]";
    xAxisTitle = l1ExtraObject + "_" + quantity + " [GeV]";
    yAxisTitle = "Entries";

    if (gtObj == HfRingEtSums) {

        indexHistogram++;
        m_indexHfRingEtSums = indexHistogram;

        for (int iCount = 0; iCount < l1extra::L1HFRings::kNumRings; ++iCount) {

            histName = l1ExtraObject + "_Count_" + boost::lexical_cast<
                    std::string>(iCount);
            histTitle = l1ExtraObject + ": count " + boost::lexical_cast<
                    std::string>(iCount);
            xAxisTitle = l1ExtraObject;
            yAxisTitle = "Entries";

            m_monElement.push_back(dbe->book1D(histName, histTitle,
                    histNrBinsET, binThresholdsETf));

            m_monElement[m_indexHfRingEtSums + iCount]->setAxisTitle(xAxisTitle,
                    1);
            m_monElement[m_indexHfRingEtSums + iCount]->setAxisTitle(yAxisTitle,
                    2);

        }

    } else {

        m_monElement.push_back(dbe->book1D(histName, histTitle, histNrBinsET,
                binThresholdsETf));
        indexHistogram++;

        m_monElement[indexHistogram]->setAxisTitle(xAxisTitle, 1);
        m_monElement[indexHistogram]->setAxisTitle(yAxisTitle, 2);
        m_indexPt = indexHistogram;
        m_indexEt = indexHistogram;
        m_indexEtTotal = indexHistogram;
    }


    delete[] binThresholdsETf;

    //

    if (bookPhi) {

        quantity = "phi";

        // get limits and binning from L1Extra
        L1GetHistLimits l1GetHistLimits(evSetup);
        const L1GetHistLimits::L1HistLimits& histLimits =
                l1GetHistLimits.l1HistLimits(gtObj, quantity);

        const int histNrBinsPhi = histLimits.nrBins;
        const double histMinValuePhi = histLimits.lowerBinValue;
        const double histMaxValuePhi = histLimits.upperBinValue;
        const std::vector<float>& binThresholdsPhi = histLimits.binThresholds;

        float* binThresholdsPhif;
        size_t sizeBinThresholdsPhi = binThresholdsPhi.size();
        binThresholdsPhif = new float[sizeBinThresholdsPhi];
        copy(binThresholdsPhi.begin(), binThresholdsPhi.end(),
                binThresholdsPhif);

        LogDebug("L1ExtraDQM") << "\n phi histogram for " << l1ExtraObject
                << "\n histNrBinsPhi = " << histNrBinsPhi
                << "\n histMinValuePhi = " << histMinValuePhi
                << "\n histMaxValuePhi = " << histMaxValuePhi
                << "\n Last bin value represents the upper limit of the histogram"
                << std::endl;
        for (size_t iBin = 0; iBin < sizeBinThresholdsPhi; ++iBin) {
            LogTrace("L1ExtraDQM") << "Bin " << iBin << ": phi = "
                    << binThresholdsPhif[iBin] << " deg" << std::endl;

        }

        histName = l1ExtraObject + "_phi";
        histTitle = l1ExtraObject + ": phi distribution ";
        xAxisTitle = l1ExtraObject + "_phi [deg]";
        yAxisTitle = "Entries";

        m_monElement.push_back(dbe->book1D(histName, histTitle, histNrBinsPhi,
                histMinValuePhi, histMaxValuePhi));
        indexHistogram++;

        m_monElement[indexHistogram]->setAxisTitle(xAxisTitle, 1);
        m_monElement[indexHistogram]->setAxisTitle(yAxisTitle, 2);
        m_indexPhi = indexHistogram;

        delete[] binThresholdsPhif;
    }

    //


    if (bookEta) {

        quantity = "eta";

        // get limits and binning from L1Extra
        L1GetHistLimits l1GetHistLimits(evSetup);
        const L1GetHistLimits::L1HistLimits& histLimits =
                l1GetHistLimits.l1HistLimits(gtObj, quantity);

        const int histNrBinsEta = histLimits.nrBins;
        const double histMinValueEta = histLimits.lowerBinValue;
        const double histMaxValueEta = histLimits.upperBinValue;
        const std::vector<float>& binThresholdsEta = histLimits.binThresholds;

        //
        float* binThresholdsEtaf;
        size_t sizeBinThresholdsEta = binThresholdsEta.size();
        binThresholdsEtaf = new float[sizeBinThresholdsEta];
        copy(binThresholdsEta.begin(), binThresholdsEta.end(),
                binThresholdsEtaf);

        LogDebug("L1ExtraDQM") << "\n eta histogram for " << l1ExtraObject
                << "\n histNrBinsEta = " << histNrBinsEta
                << "\n histMinValueEta = " << histMinValueEta
                << "\n histMaxValueEta = " << histMaxValueEta
                << "\n Last bin value represents the upper limit of the histogram"
                << std::endl;
        for (size_t iBin = 0; iBin < sizeBinThresholdsEta; ++iBin) {
            LogTrace("L1ExtraDQM") << "Bin " << iBin << ": eta = "
                    << binThresholdsEtaf[iBin] << std::endl;

        }

        histName = l1ExtraObject + "_eta";
        histTitle = l1ExtraObject + ": eta distribution ";
        xAxisTitle = l1ExtraObject + "_eta";
        yAxisTitle = "Entries";

        m_monElement.push_back(dbe->book1D(histName, histTitle, histNrBinsEta,
                binThresholdsEtaf));
        indexHistogram++;

        m_monElement[indexHistogram]->setAxisTitle(xAxisTitle, 1);
        m_monElement[indexHistogram]->setAxisTitle(yAxisTitle, 2);
        m_indexEta = indexHistogram;

        delete[] binThresholdsEtaf;

    }

}

template<class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillNrObjects(
        const CollectionType* collType, const bool validColl,
        const bool isL1Coll, const int bxInEvent) {

    if (validColl && isL1Coll) {
        size_t collSize = 0;
        for (CIterColl iterColl = collType->begin(); iterColl
                != collType->end(); ++iterColl) {

            if (iterColl->bx() == bxInEvent) {
                collSize++;
            }
        }
        m_monElement[m_indexNrObjects]->Fill(collSize);
    } else {
        size_t collSize = collType->size();
        m_monElement[m_indexNrObjects]->Fill(collSize);
    }
}

template<class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillPtPhiEta(
        const CollectionType* collType, const bool validColl,
        const bool bookPhi, const bool bookEta, const bool isL1Coll, const int bxInEvent) {

    if (validColl) {
        for (CIterColl iterColl = collType->begin(); iterColl
                != collType->end(); ++iterColl) {

            if (isL1Coll && (iterColl->bx() != bxInEvent)) {
                continue;
            }

            m_monElement[m_indexPt]->Fill(iterColl->pt());

            if (bookPhi) {
                // add a very small quantity to get off the bin edge
                m_monElement[m_indexPhi]->Fill(rad2deg(iterColl->phi()) + 1.e-6);
            }

            if (bookEta) {
                m_monElement[m_indexEta]->Fill(iterColl->eta());
            }

        }
    }
}

template<class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillEtPhiEta(
        const CollectionType* collType, const bool validColl,
        const bool bookPhi, const bool bookEta, const bool isL1Coll, const int bxInEvent) {

    if (validColl) {
        for (CIterColl iterColl = collType->begin(); iterColl
                != collType->end(); ++iterColl) {

            if (isL1Coll && (iterColl->bx() != bxInEvent)) {
                continue;
            }

            m_monElement[m_indexEt]->Fill(iterColl->et());

            if (bookPhi) {
                // add a very small quantity to get off the bin edge
                m_monElement[m_indexPhi]->Fill(rad2deg(iterColl->phi()) + 1.e-6);
            }

            if (bookEta) {
                m_monElement[m_indexEta]->Fill(iterColl->eta());
            }

        }
    }
}

template<class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillEtTotal(
        const CollectionType* collType, const bool validColl, const bool isL1Coll, const int bxInEvent) {

    if (validColl) {
        for (CIterColl iterColl = collType->begin(); iterColl
                != collType->end(); ++iterColl) {

            if (isL1Coll && (iterColl->bx() != bxInEvent)) {
                continue;
            }

            m_monElement[m_indexEtTotal]->Fill(iterColl->etTotal());
        }
    }

}

template<class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillCharge(
        const CollectionType* collType, const bool validColl, const bool isL1Coll, const int bxInEvent) {

    if (validColl) {
        for (CIterColl iterColl = collType->begin(); iterColl
                != collType->end(); ++iterColl) {

            if (isL1Coll && (iterColl->bx() != bxInEvent)) {
                continue;
            }

            m_monElement[m_indexCharge]->Fill(iterColl->charge());
        }
    }

}

template<class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillHfBitCounts(
        const CollectionType* collType, const bool validColl,
        const int countIndex, const bool isL1Coll, const int bxInEvent) {

    if (validColl) {
        for (CIterColl iterColl = collType->begin(); iterColl
                != collType->end(); ++iterColl) {

            if (isL1Coll && (iterColl->bx() != bxInEvent)) {
                continue;
            }

            m_monElement[m_indexHfBitCounts + countIndex]->Fill(
                    iterColl->hfBitCount(
                            (l1extra::L1HFRings::HFRingLabels) countIndex));
        }
    }

}

template<class CollectionType>
void L1ExtraDQM::L1ExtraMonElement<CollectionType>::fillHfRingEtSums(
        const CollectionType* collType, const bool validColl,
        const int countIndex, const bool isL1Coll, const int bxInEvent) {

    if (validColl) {
        for (CIterColl iterColl = collType->begin(); iterColl
                != collType->end(); ++iterColl) {

            if (isL1Coll && (iterColl->bx() != bxInEvent)) {
                continue;
            }

            m_monElement[m_indexHfRingEtSums + countIndex]->Fill(
                    iterColl->hfEtSum(
                            (l1extra::L1HFRings::HFRingLabels) countIndex));
        }
    }

}

#endif
