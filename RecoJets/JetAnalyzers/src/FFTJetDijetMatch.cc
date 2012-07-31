// -*- C++ -*-
//
// Package:    FFTJetDijetMatch
// Class:      FFTJetDijetMatch
// 
/**\class FFTJetDijetMatch FFTJetDijetMatch.cc RecoJets/JetAnalyzers/src/FFTJetDijetMatch.cc

 Description: dumps the info created by FFTJetProducer into a root ntuple

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Mon Jul 16 22:09:23 CDT 2012
// $Id: FFTJetDijetMatch.cc,v 1.7 2012/07/17 19:35:58 igv Exp $
//
//
#include <cfloat>
#include <string>
#include <sstream>
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
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "RecoJets/FFTJetAlgorithms/interface/jetConverters.h"
#include "RecoJets/FFTJetAlgorithms/interface/adjustForPileup.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#define PILEUP_SUBTRACTION_MASK (0x400 | 0x800)

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

//
// class declaration
//
class FFTJetDijetMatch : public edm::EDAnalyzer
{
public:
    explicit FFTJetDijetMatch(const edm::ParameterSet&);
    ~FFTJetDijetMatch();

private:
    typedef fftjet::RecombinedJet<fftjetcms::VectorLike> MyFFTJet;
    typedef std::vector<MyFFTJet> FFTCollection;

    FFTJetDijetMatch();
    FFTJetDijetMatch(const FFTJetDijetMatch&);
    FFTJetDijetMatch& operator=(const FFTJetDijetMatch&);

    virtual void beginJob();
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob();

    template <class JetType>
    void loadCollection(const edm::Event& iEvent,
                        const edm::InputTag& label,
                        FFTCollection* coll);

    template <class JetType>
    void fillJetInfo(const edm::Event& iEvent,
                     const edm::InputTag& label,
                     TNtuple* nt);

    inline double calcPeakPt(const fftjet::Peak& peak) const
    {
        const double s = peak.scale();
        return ptConversionFactor*s*s*peak.magnitude();
    }

    void ntuplizePeak(const fftjet::Peak* peak, std::string *vars,
                      unsigned sequenceNumber);
    void ntuplizeJet(const MyFFTJet* jet, const math::XYZTLorentzVector& pileupVec,
                     std::string *vars, unsigned sequenceNumber);
    void ntuplizeSummary(const reco::FFTJetProducerSummary* psum,
                         std::string *vars);
    void ntuplizeParton(const reco::GenParticle* parton, double dr, std::string *vars);

    edm::InputTag genjetCollectionLabel;

    bool collectCaloJets;
    edm::InputTag caloCollectionLabel;
    std::string caloTitle;

    bool collectPFJets;
    edm::InputTag pfCollectionLabel;
    std::string pfTitle;

    double ptConversionFactor;
    bool subtractPileupAs4Vec;
    bool matchParton;
    bool usingDijetGun;

    std::vector<float> ntupleData;

    TNtuple* ntCalo;
    TNtuple* ntPF;

    unsigned long counter;
    unsigned numEventVariables;
    unsigned numSummaryVariables;

    FFTCollection genJets;
    FFTCollection myJets;
    std::vector<math::XYZTLorentzVector> myPileup;
};

//
// constructors and destructor
//
FFTJetDijetMatch::FFTJetDijetMatch(const edm::ParameterSet& ps)
    : init_param(edm::InputTag, genjetCollectionLabel),
      init_param(bool, collectCaloJets),
      init_param(edm::InputTag, caloCollectionLabel),
      init_param(std::string, caloTitle),
      init_param(bool, collectPFJets),
      init_param(edm::InputTag, pfCollectionLabel),
      init_param(std::string, pfTitle),
      init_param(double, ptConversionFactor),
      init_param(bool, subtractPileupAs4Vec),
      init_param(bool, matchParton),
      init_param(bool, usingDijetGun),
      ntCalo(0),
      ntPF(0),
      counter(0),
      numEventVariables(0),
      numSummaryVariables(0)
{
}


FFTJetDijetMatch::~FFTJetDijetMatch()
{
}


#define add_variable(varname)      \
    if (vars)                      \
    {                              \
        std::ostringstream os;     \
        os << ':';                 \
        os << #varname ;           \
        os << sequenceNumber;      \
        *vars += os.str();         \
    }                              \
    ntupleData.push_back(varname); \


void FFTJetDijetMatch::ntuplizePeak(const fftjet::Peak* inpeak,
                                    std::string *vars,
                                    const unsigned sequenceNumber)
{
    float peakEta = 0.f;
    float peakPhi = 0.f;
    float peakMagnitude = 0.f;
    float peakPt = 0.f;
    float peakDriftSpeed = 0.f;
    float peakMagSpeed = 0.f;
    float peakLifetime = 0.f;
    float peakScale = 0.f;
    float peakNearestNeighborDistance = 0.f;
    float peakClusterRadius = 0.f;
    float peakClusterSeparation = 0.f;
    float hessDet = 0.f;
    float hessLaplacian = 0.f;
    float hessEtaPhiRatio = -1.f;
    float hessEigenRatio = 0.f;

    if (inpeak)
    {
        const fftjet::Peak& peak(*inpeak);

        peakEta = peak.eta();
        peakPhi = peak.phi();
        peakMagnitude = peak.magnitude();
        peakPt = calcPeakPt(peak);
        peakDriftSpeed = peak.driftSpeed();
        peakMagSpeed = peak.magSpeed();
        peakLifetime = peak.lifetime();
        peakScale = peak.scale();
        peakNearestNeighborDistance = peak.nearestNeighborDistance();
        peakClusterRadius = peak.clusterRadius();
        peakClusterSeparation = peak.clusterSeparation();
        hessDet = peak.hessianDeterminant();
        hessLaplacian = -peak.laplacian();

        double hess[3];
        peak.hessian(hess);
        if (hess[0])
            // The following quantity approximates sigma_eta/sigma_phi squared
            hessEtaPhiRatio = hess[2]/hess[0];
        else if (hess[2])
            hessEtaPhiRatio = FLT_MAX;

        const double hDelta = hess[2] - hess[0];
        const double eigen1 = (hess[0] + hess[2] - 
                               sqrt(4.0*hess[1]*hess[1] + hDelta*hDelta))/2.0;
        if (eigen1)
        {
            const double eigen2 = hessDet/eigen1;
            const double emin = std::min(fabs(eigen1), fabs(eigen2));
            const double emax = std::max(fabs(eigen1), fabs(eigen2));
            hessEigenRatio = emin/emax;
        }
        else if (!hessDet)
            hessEigenRatio = -1.f;
    }

    add_variable(peakEta);
    add_variable(peakPhi);
    add_variable(peakMagnitude);
    add_variable(peakPt);
    add_variable(peakDriftSpeed);
    add_variable(peakMagSpeed);
    add_variable(peakLifetime);
    add_variable(peakScale);
    add_variable(peakNearestNeighborDistance);
    add_variable(peakClusterRadius);
    add_variable(peakClusterSeparation);
    add_variable(hessDet);
    add_variable(hessLaplacian);
    add_variable(hessEtaPhiRatio);
    add_variable(hessEigenRatio);    
}


void FFTJetDijetMatch::ntuplizeJet(const MyFFTJet* injet, 
                                   const math::XYZTLorentzVector& pileupVec,
                                   std::string *vars,
                                   const unsigned sequenceNumber)
{
    const fftjet::Peak* peak = injet ? &injet->precluster() : 0;
    ntuplizePeak(peak, vars, sequenceNumber);

    float pileupPt = 0.f;
    float pileupEta = 0.f;
    float pileupPhi = 0.f;
    float pileupMass = 0.f;
    float pt = 0.f;
    float eta = 0.f;
    float phi = 0.f;
    float mass = 0.f;
    float etSum = 0.f;
    float centroidEta = 0.f;
    float centroidPhi = 0.f;
    float etaWidth = 0.f;
    float phiWidth = 0.f;
    float etaPhiCorr = 0.f;
    float fuzziness = 0.f;
    float convergenceDistance = 0.f;
    float recoScale = 0.f;
    float recoScaleRatio = 0.f;
    float membershipFactor = 0.f;
    float code = 0.f;
    float status = 0.f;

    if (injet)
    {
        const MyFFTJet& jet(*injet);

        pileupPt = pileupVec.Pt();
        pileupEta = pileupVec.Eta();
        pileupPhi = pileupVec.Phi();
        pileupMass = pileupVec.mass();
        pt = jet.vec().Pt();
        eta = jet.vec().Eta();
        phi = jet.vec().Phi();
        mass = jet.vec().mass();
        etSum = jet.etSum();
        centroidEta = jet.centroidEta();
        centroidPhi = jet.centroidPhi();
        etaWidth = jet.etaWidth();
        phiWidth = jet.phiWidth();
        etaPhiCorr = jet.etaPhiCorr();
        fuzziness = jet.fuzziness();
        convergenceDistance = jet.convergenceDistance();
        recoScale = jet.recoScale();
        recoScaleRatio = jet.recoScaleRatio();
        membershipFactor = jet.membershipFactor();
        code = jet.code();
        status = jet.status();
    }

    add_variable(pileupPt);
    add_variable(pileupEta);
    add_variable(pileupPhi);
    add_variable(pileupMass);
    add_variable(pt);
    add_variable(eta);
    add_variable(phi);
    add_variable(mass);
    add_variable(etSum);
    add_variable(centroidEta);
    add_variable(centroidPhi);
    add_variable(etaWidth);
    add_variable(phiWidth);
    add_variable(etaPhiCorr);
    add_variable(fuzziness);
    add_variable(convergenceDistance);
    add_variable(recoScale);
    add_variable(recoScaleRatio);
    add_variable(membershipFactor);
    add_variable(code);
    add_variable(status);
}


void FFTJetDijetMatch::ntuplizeParton(const reco::GenParticle* parton,
                                      const double dr, std::string *vars)
{
    const unsigned sequenceNumber = 0;

    float partonPt = 0.f;
    float partonEta = 0.f;
    float partonPhi = 0.f;
    float partonMass = 0.f;
    float partonCode = 0.f;
    float partonDR = 100.f;

    if (parton)
    {
        partonPt = parton->pt();
        partonEta = parton->eta();
        partonPhi = parton->phi();
        partonMass = parton->mass();
        partonCode = parton->pdgId();
        partonDR = dr;
    }

    add_variable(partonPt);
    add_variable(partonEta);
    add_variable(partonPhi);
    add_variable(partonMass);
    add_variable(partonCode);
    add_variable(partonDR);
}


void FFTJetDijetMatch::ntuplizeSummary(const reco::FFTJetProducerSummary* summary,
                                       std::string *vars)
{
    const unsigned sequenceNumber = 0;

    float unclusPt = 0.f;
    float unclusEta = 0.f;
    float unclusPhi = 0.f;
    float unclusMass = 0.f;
    float unusedEt = 0.f;
    float iterationsPerformed = 0.f;
    float converged = 0.f;

    if (summary)
    {
        unclusPt = summary->unclustered().Pt();
        unclusEta = summary->unclustered().Eta();
        unclusPhi = summary->unclustered().Phi();
        unclusMass = summary->unclustered().mass();
        unusedEt = summary->unusedEt();
        iterationsPerformed = summary->iterationsPerformed();
        converged = summary->iterationsConverged();
    }

    add_variable(unclusPt);
    add_variable(unclusEta);
    add_variable(unclusPhi);
    add_variable(unclusMass);
    add_variable(unusedEt);
    add_variable(iterationsPerformed);
    add_variable(converged);
}


// ------------ method called once each job just before starting event loop
void FFTJetDijetMatch::beginJob()
{
    // Generic variables which identify the event
    std::string vars = "cnt:run:event";
    numEventVariables = std::count(vars.begin(), vars.end(), ':') + 1U;

    // Variables from FFTJetProducerSummary
    ntuplizeSummary(0, &vars);
    numSummaryVariables = std::count(vars.begin(), vars.end(), ':') + 1U;

    // Variables which identify the jet matching
    vars += ":genJetNumber:matchNumber:matchDR";

    // The response variable
    vars += ":ptResponse";

    // Other variables
    math::XYZTLorentzVector dummy;
    ntuplizeJet(0, dummy, &vars, 0U);
    ntuplizeJet(0, dummy, &vars, 1U);
    if (matchParton)
        ntuplizeParton(0, 0.0, &vars);

    ntupleData.reserve(std::count(vars.begin(), vars.end(), ':') + 1U);

    // Book the ntuples
    edm::Service<TFileService> fs;

    if (collectPFJets)
        ntPF = fs->make<TNtuple>(
            pfTitle.c_str(), pfTitle.c_str(), vars.c_str());

    if (collectCaloJets)
        ntCalo = fs->make<TNtuple>(
            caloTitle.c_str(), caloTitle.c_str(), vars.c_str());
}


template <class JetType>
void FFTJetDijetMatch::loadCollection(const edm::Event& iEvent,
                                      const edm::InputTag& label,
                                      FFTCollection* coll)
{
    typedef reco::FFTAnyJet<JetType> MyJet;
    typedef std::vector<MyJet> MyCollection;

    assert(coll);
    edm::Handle<MyCollection> jets;
    iEvent.getByLabel(label, jets);

    const unsigned nJets = jets->size();
    coll->clear();
    coll->reserve(nJets);
    myPileup.clear();
    myPileup.reserve(nJets);

    for (unsigned i=0; i<nJets; ++i)
    {
        const fftjet::RecombinedJet<fftjetcms::VectorLike>& jet(
            fftjetcms::jetFromStorable((*jets)[i].getFFTSpecific()));
        coll->push_back(jet);
        myPileup.push_back((*jets)[i].getFFTSpecific().f_pileup());
    }
}


template <class JetType>
void FFTJetDijetMatch::fillJetInfo(const edm::Event& iEvent,
                                   const edm::InputTag& label,
                                   TNtuple* nt)
{
    if (nt)
    {
        // genJets collection must be loaded first.
        // Collection loading reuses the same pileup vector.
        loadCollection<reco::GenJet>(iEvent, genjetCollectionLabel, &genJets);
        loadCollection<JetType>(iEvent, label, &myJets);

        unsigned previousPartonMatch = 0;
        edm::Handle<reco::GenParticleCollection> genParticles;
        if (matchParton)
        {
            iEvent.getByLabel(edm::InputTag("genParticles"), genParticles);
            previousPartonMatch = genParticles->size();
        }

        const unsigned nReco = myJets.size();
        if (genJets.size() < 2 || nReco < 2)
            // This is a very strange event. Ignore it.
            return;

        // Get the summary info for this jet collection
        edm::Handle<reco::FFTJetProducerSummary> summary;
        iEvent.getByLabel(label, summary);
        ntupleData.erase(ntupleData.begin()+numEventVariables, ntupleData.end());
        ntuplizeSummary(&*summary, 0);

        // For the first two genJets, find the best matching jet
        // from the other collection and push the jet data into
        // the ntuple. The match will be one-to-one.
        unsigned previousMatch = nReco;
        for (unsigned igen=0; igen<2U; ++igen)
        {
            const MyFFTJet& genjet(genJets[igen]);
            const double genEta = genjet.vec().Eta();
            const double genPhi = genjet.vec().Phi();

            unsigned match = nReco;
            double bestDR = DBL_MAX;
            for (unsigned ireco=0; ireco<nReco; ++ireco)
                if (ireco != previousMatch)
                {
                    const MyFFTJet& j(myJets[ireco]);
                    const double d = reco::deltaR(genEta, genPhi,
                                                  j.vec().Eta(), j.vec().Phi());
                    if (d < bestDR)
                    {
                        bestDR = d;
                        match = ireco;
                    }
                }

            assert(match < nReco);
            previousMatch = match;
            const MyFFTJet& recoJet(myJets[match]);

            ntupleData.erase(ntupleData.begin()+numSummaryVariables,
                             ntupleData.end());

            ntupleData.push_back(igen);
            ntupleData.push_back(match);
            ntupleData.push_back(bestDR);
            double response = 0.0;
            if (recoJet.status() & PILEUP_SUBTRACTION_MASK)
                // Pileup is already subtracted
                response = recoJet.vec().Pt()/genjet.vec().Pt();
            else
            {
                // Need to subtract the pileup
                const math::XYZTLorentzVector& recoP4 = 
                    fftjetcms::adjustForPileup(
                        recoJet.vec(), myPileup[match], subtractPileupAs4Vec);
                response = recoP4.Pt()/genjet.vec().Pt();
            }
            ntupleData.push_back(response);

            ntuplizeJet(&recoJet, myPileup[match], 0, 0U);
            math::XYZTLorentzVector dummy;
            ntuplizeJet(&genjet, dummy, 0, 1U);

            // Are we going to match a parton to genGet?
            if (matchParton)
            {
                unsigned partonMatch = genParticles->size();
                bestDR = DBL_MAX;

                if (usingDijetGun)
                {
                    for (unsigned igen=0; igen<2; ++igen)
                        if (igen != previousPartonMatch)
                        {
                            const reco::GenParticle& j((*genParticles)[igen]);
                            const double d = reco::deltaR(genEta, genPhi,
                                                          j.eta(), j.phi());
                            if (d < bestDR)
                            {
                                bestDR = d;
                                partonMatch = igen;
                            }
                        }
                }
                else
                    throw cms::Exception("FFTJetBadConfig")
                        << "Generic parton matching is not implemented";

                assert(partonMatch < genParticles->size());
                previousPartonMatch = partonMatch;
                ntuplizeParton(&(*genParticles)[partonMatch], bestDR, 0);
            }

            assert(ntupleData.size() == static_cast<unsigned>(nt->GetNvar()));
            nt->Fill(&ntupleData[0]);
        }
    }
}


// ------------ method called to for each event  ------------
void FFTJetDijetMatch::analyze(const edm::Event& iEvent,
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

    ++counter;
}


// ------------ method called once each job just after ending the event loop
void FFTJetDijetMatch::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetDijetMatch);
