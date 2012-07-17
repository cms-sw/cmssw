// -*- C++ -*-
//
// Package:    SimpleFFTJetAnalyzer
// Class:      SimpleFFTJetAnalyzer
// 
/**\class SimpleFFTJetAnalyzer SimpleFFTJetAnalyzer.cc RecoJets/JetAnalyzers/src/SimpleFFTJetAnalyzer.cc

 Description: dumps the info created by FFTJetProducer into a root ntuple

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Mon Jul 16 22:09:23 CDT 2012
// $Id: SimpleFFTJetAnalyzer.cc,v 1.1 2012/07/16 17:40:54 igv Exp $
//
//

#include <cassert>
#include <algorithm>

#include "TNtuple.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include "DataFormats/JetReco/interface/FFTJetProducerSummary.h"
#include "DataFormats/JetReco/interface/FFTPFJetCollection.h"
#include "DataFormats/JetReco/interface/FFTCaloJetCollection.h"
#include "DataFormats/JetReco/interface/FFTGenJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoJets/FFTJetAlgorithms/interface/jetConverters.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"


#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

//
// class declaration
//
class SimpleFFTJetAnalyzer : public edm::EDAnalyzer
{
public:
    explicit SimpleFFTJetAnalyzer(const edm::ParameterSet&);
    ~SimpleFFTJetAnalyzer();

private:
    SimpleFFTJetAnalyzer();
    SimpleFFTJetAnalyzer(const SimpleFFTJetAnalyzer&);
    SimpleFFTJetAnalyzer& operator=(const SimpleFFTJetAnalyzer&);

    virtual void beginJob();
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob();

    template <class JetType>
    void fillJetInfo(const edm::Event& iEvent,
                     const edm::InputTag& label,
                     TNtuple* nt);

    bool collectGenJets;
    edm::InputTag genjetCollectionLabel;
    std::string genjetTitle;

    bool collectCaloJets;
    edm::InputTag caloCollectionLabel;
    std::string caloTitle;

    bool collectPFJets;
    edm::InputTag pfCollectionLabel;
    std::string pfTitle;

    double ptConversionFactor;

    std::vector<float> ntupleData;

    TNtuple* ntGen;
    TNtuple* ntCalo;
    TNtuple* ntPF;

    unsigned long counter;
    unsigned numEventVariables;
    unsigned numSummaryVariables;
};

//
// constructors and destructor
//
SimpleFFTJetAnalyzer::SimpleFFTJetAnalyzer(const edm::ParameterSet& ps)
    : init_param(bool, collectGenJets),
      init_param(edm::InputTag, genjetCollectionLabel),
      init_param(std::string, genjetTitle),
      init_param(bool, collectCaloJets),
      init_param(edm::InputTag, caloCollectionLabel),
      init_param(std::string, caloTitle),
      init_param(bool, collectPFJets),
      init_param(edm::InputTag, pfCollectionLabel),
      init_param(std::string, pfTitle),
      init_param(double, ptConversionFactor),
      ntGen(0),
      ntCalo(0),
      ntPF(0),
      counter(0),
      numEventVariables(0),
      numSummaryVariables(0)
{
}


SimpleFFTJetAnalyzer::~SimpleFFTJetAnalyzer()
{
}


// ------------ method called once each job just before starting event loop
void SimpleFFTJetAnalyzer::beginJob()
{
    // Generic variables which identify the event
    std::string vars = "cnt:run:event";

    numEventVariables = std::count(vars.begin(), vars.end(), ':') + 1U;

    // Variables from FFTJetProducerSummary
    vars += ":unclusPt:unclusEta:unclusPhi:unclusMass:unusedEt:iterationsPerformed:converged";

    numSummaryVariables = std::count(vars.begin(), vars.end(), ':') + 1U;

    // Variables which identify the jet
    vars += ":jetNumber:nJets";

    // Peak-related variables
    vars += ":peakEta:peakPhi:peakMagnitude:peakPt:peakDriftSpeed:peakMagSpeed:peakLifetime:peakScale:peakNearestNeighborDistance:peakClusterRadius:peakClusterSeparation:hessDet:hessLaplacian";

    // Pile-up variables
    vars += ":pileupPt:pileupEta:pileupPhi:pileupMass";

    // FFTJet specific jet quantities
    vars += ":pt:eta:phi:mass:etSum:centroidEta:centroidPhi:etaWidth:phiWidth:etaPhiCorr:fuzziness:convergenceDistance:recoScale:recoScaleRatio:membershipFactor:code:status";

    // Closest jet in delta R
    vars += ":closestJetIndex:closestJetDR:closestJetDPhi:closestJetPt";

    // Generic jet info
    vars += ":area:nConstituents";

    ntupleData.reserve(std::count(vars.begin(), vars.end(), ':') + 1U);

    // Book the ntuples
    edm::Service<TFileService> fs;

    if (collectGenJets)
        ntGen = fs->make<TNtuple>(
            genjetTitle.c_str(), genjetTitle.c_str(), vars.c_str());

    if (collectCaloJets)
        ntCalo = fs->make<TNtuple>(
            caloTitle.c_str(), caloTitle.c_str(), vars.c_str());

    if (collectPFJets)
        ntPF = fs->make<TNtuple>(
            pfTitle.c_str(), pfTitle.c_str(), vars.c_str());
}


template <class JetType>
void SimpleFFTJetAnalyzer::fillJetInfo(const edm::Event& iEvent,
                                       const edm::InputTag& label,
                                       TNtuple* nt)
{
    typedef reco::FFTAnyJet<JetType> MyJet;
    typedef std::vector<MyJet> MyCollection;

    if (nt)
    {
        edm::Handle<MyCollection> jets;
        iEvent.getByLabel(label, jets);

        edm::Handle<reco::FFTJetProducerSummary> summary;
        iEvent.getByLabel(label, summary);

        ntupleData.erase(ntupleData.begin()+numEventVariables, ntupleData.end());

        ntupleData.push_back(summary->unclustered().Pt());
        ntupleData.push_back(summary->unclustered().Eta());
        ntupleData.push_back(summary->unclustered().Phi());
        ntupleData.push_back(summary->unclustered().mass());
        ntupleData.push_back(summary->unusedEt());
        ntupleData.push_back(summary->iterationsPerformed());
        ntupleData.push_back(summary->iterationsConverged());

        const unsigned nJets = jets->size();
        for (unsigned i=0; i<nJets; ++i)
        {
            const MyJet& storedJet((*jets)[i]);
            const fftjet::RecombinedJet<fftjetcms::VectorLike>& jet(
                fftjetcms::jetFromStorable(storedJet.getFFTSpecific()));
            const fftjet::Peak& peak = jet.precluster();

            ntupleData.erase(ntupleData.begin()+numSummaryVariables, ntupleData.end());

            ntupleData.push_back(i);
            ntupleData.push_back(nJets);

            const double peakScale = peak.scale();
            ntupleData.push_back(peak.eta());
            ntupleData.push_back(peak.phi());
            ntupleData.push_back(peak.magnitude());
            ntupleData.push_back(ptConversionFactor*peakScale*peakScale*peak.magnitude());
            ntupleData.push_back(peak.driftSpeed());
            ntupleData.push_back(peak.magSpeed());
            ntupleData.push_back(peak.lifetime());
            ntupleData.push_back(peakScale);
            ntupleData.push_back(peak.nearestNeighborDistance());
            ntupleData.push_back(peak.clusterRadius());
            ntupleData.push_back(peak.clusterSeparation());
            ntupleData.push_back(peak.hessianDeterminant());
            ntupleData.push_back(-peak.laplacian());

            const math::XYZTLorentzVector& pileupVec(
                storedJet.getFFTSpecific().f_pileup());
            ntupleData.push_back(pileupVec.Pt()); // identical to jet.pileup()
            ntupleData.push_back(pileupVec.Eta());
            ntupleData.push_back(pileupVec.Phi());
            ntupleData.push_back(pileupVec.mass());

            ntupleData.push_back(jet.vec().Pt());
            ntupleData.push_back(jet.vec().Eta());
            ntupleData.push_back(jet.vec().Phi());
            ntupleData.push_back(jet.vec().mass());
            ntupleData.push_back(jet.etSum());
            ntupleData.push_back(jet.centroidEta());
            ntupleData.push_back(jet.centroidPhi());
            ntupleData.push_back(jet.etaWidth());
            ntupleData.push_back(jet.phiWidth());
            ntupleData.push_back(jet.etaPhiCorr());
            ntupleData.push_back(jet.fuzziness());
            ntupleData.push_back(jet.convergenceDistance());
            ntupleData.push_back(jet.recoScale());
            ntupleData.push_back(jet.recoScaleRatio());
            ntupleData.push_back(jet.membershipFactor());
            ntupleData.push_back(jet.code());
            ntupleData.push_back(jet.status());

            int closestJetIndex   = -1;
            double closestJetDR   = -1.0;
            double closestJetPt   = -10.0;
            double closestJetDphi = 4.0;
            if (nJets > 1U)
            {
                closestJetDR = 1.0e30;
                for (unsigned k=0; k<nJets; ++k)
                    if (k != i)
                    {
                        const reco::FFTJet<float>& otherJet((*jets)[k].getFFTSpecific());
                        const double dr = reco::deltaR(jet.vec(), otherJet.f_vec());
                        if (dr < closestJetDR)
                        {
                            closestJetDR = dr;
                            closestJetIndex = k;
                            closestJetPt = otherJet.f_vec().Pt();
                            closestJetDphi = reco::deltaPhi(jet.vec().Phi(),
                                                            otherJet.f_vec().Phi());
                        }
                    }
            }
            ntupleData.push_back(closestJetIndex);
            ntupleData.push_back(closestJetDR);
            ntupleData.push_back(closestJetDphi);
            ntupleData.push_back(closestJetPt);

            ntupleData.push_back(storedJet.jetArea());
            ntupleData.push_back(storedJet.nConstituents());

            assert(ntupleData.size() == static_cast<unsigned>(nt->GetNvar()));
            nt->Fill(&ntupleData[0]);
        }
    }
}


// ------------ method called to for each event  ------------
void SimpleFFTJetAnalyzer::analyze(const edm::Event& iEvent,
                                   const edm::EventSetup& iSetup)
{
    ntupleData.clear();
    ntupleData.push_back(counter);

    const long runnumber = iEvent.id().run();
    const long eventnumber = iEvent.id().event();
    ntupleData.push_back(runnumber);
    ntupleData.push_back(eventnumber);

    fillJetInfo<reco::PFJet>(iEvent, pfCollectionLabel, ntPF);
    fillJetInfo<reco::CaloJet>(iEvent, caloCollectionLabel, ntCalo);
    fillJetInfo<reco::GenJet>(iEvent, genjetCollectionLabel, ntGen);

    ++counter;
}


// ------------ method called once each job just after ending the event loop
void SimpleFFTJetAnalyzer::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(SimpleFFTJetAnalyzer);
