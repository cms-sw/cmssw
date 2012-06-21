// -*- C++ -*-
//
// Package:    FFTJetProducers
// Class:      FFTJetPFPileupCleaner
// 
/**\class FFTJetPFPileupCleaner FFTJetPFPileupCleaner.cc RecoJets/FFTJetProducers/plugins/FFTJetPFPileupCleaner.cc

 Description: cleans up a collection of partice flow objects

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu Jul 14 17:50:33 CDT 2011
// $Id: FFTJetPFPileupCleaner.cc,v 1.1 2011/07/15 04:26:34 igv Exp $
//
//
#include <cmath>
#include <utility>
#include <algorithm>

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

//
// class declaration
//
class FFTJetPFPileupCleaner : public edm::EDProducer
{
public:
    explicit FFTJetPFPileupCleaner(const edm::ParameterSet&);
    ~FFTJetPFPileupCleaner();

protected:
    // methods
    void beginJob();
    void produce(edm::Event&, const edm::EventSetup&);
    void endJob();

private:
    FFTJetPFPileupCleaner();
    FFTJetPFPileupCleaner(const FFTJetPFPileupCleaner&);
    FFTJetPFPileupCleaner& operator=(const FFTJetPFPileupCleaner&);

    bool isRemovable(reco::PFCandidate::ParticleType ptype) const;
    void setRemovalBit(reco::PFCandidate::ParticleType ptype, bool onOff);
    void buildRemovalMask();
    bool isAcceptableVtx(reco::VertexCollection::const_iterator iv) const;

    reco::VertexRef findSomeVertex(
        const edm::Handle<reco::VertexCollection>& vertices,
        const reco::PFCandidate& pfcand) const;

    edm::InputTag PFCandidates;
    edm::InputTag Vertices;

    // The following, if true, will cause association of a candidate
    // with some vertex no matter what
    bool checkClosestZVertex;

    // The following, if true, will cause removal of candidates
    // associated with the main vertex
    bool removeMainVertex;

    // The following will tell us to remove candidates not associated
    // with any vertex
    bool removeUnassociated;

    // Do we want to reverse the decision?
    bool reverseRemovalDecision;

    // Flags for removing things. See PFCandidate header
    // for particle types.
    bool remove_X;         // undefined
    bool remove_h;         // charged hadron
    bool remove_e;         // electron
    bool remove_mu;        // muon
    bool remove_gamma;     // photon
    bool remove_h0;        // neutral hadron
    bool remove_h_HF;      // HF tower identified as a hadron
    bool remove_egamma_HF; // HF tower identified as an EM particle

    // Mask for removing things
    unsigned removalMask;

    // Try to make multiple associations based on the Z position
    unsigned nZAssociations;

    // Min and max eta for keeping things
    double etaMin;
    double etaMax;

    // Cut for the vertex Ndof
    double vertexNdofCut;

    // Cut for the vertex Z
    double vertexZmax;

    // Vector for associating tracks with Z positions of the vertices
    mutable std::vector<std::pair<double, unsigned> > zAssoc;
};

//
// constructors and destructor
//
FFTJetPFPileupCleaner::FFTJetPFPileupCleaner(const edm::ParameterSet& ps)
    : init_param(edm::InputTag, PFCandidates),
      init_param(edm::InputTag, Vertices),
      init_param(bool, checkClosestZVertex),
      init_param(bool, removeMainVertex),
      init_param(bool, removeUnassociated),
      init_param(bool, reverseRemovalDecision),
      init_param(bool, remove_X        ),
      init_param(bool, remove_h        ),
      init_param(bool, remove_e        ),
      init_param(bool, remove_mu       ),
      init_param(bool, remove_gamma    ),
      init_param(bool, remove_h0       ),
      init_param(bool, remove_h_HF     ),
      init_param(bool, remove_egamma_HF),
      removalMask(0),
      init_param(unsigned, nZAssociations),
      init_param(double, etaMin),
      init_param(double, etaMax),
      init_param(double, vertexNdofCut),
      init_param(double, vertexZmax)
{
    buildRemovalMask();
    produces<reco::PFCandidateCollection>();
}


FFTJetPFPileupCleaner::~FFTJetPFPileupCleaner()
{
}


bool FFTJetPFPileupCleaner::isAcceptableVtx(
    reco::VertexCollection::const_iterator iv) const
{
    return !iv->isFake() &&
            static_cast<double>(iv->ndof()) > vertexNdofCut &&
            std::abs(iv->z()) < vertexZmax;
}


// ------------ method called to produce the data  ------------
void FFTJetPFPileupCleaner::produce(
    edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // get PFCandidates
    std::auto_ptr<reco::PFCandidateCollection> 
        pOutput(new reco::PFCandidateCollection);

    edm::Handle<reco::PFCandidateCollection> pfCandidates;
    iEvent.getByLabel(PFCandidates, pfCandidates);

    // get vertices
    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByLabel(Vertices, vertices);

    const unsigned ncand = pfCandidates->size();
    for (unsigned i=0; i<ncand; ++i)
    {
        reco::PFCandidatePtr candptr(pfCandidates, i);
        bool remove = false;

        if (isRemovable(candptr->particleId()))
        {
            reco::VertexRef vertexref(findSomeVertex(vertices, *candptr));
            if (vertexref.isNull())
                remove = removeUnassociated;
            else if (vertexref.key() == 0)
                remove = removeMainVertex;
            else
                remove = true;
        }

        // Check the eta range
        if (!remove)
        {
            const double eta = candptr->p4().Eta();
            remove = eta < etaMin || eta > etaMax;
        }

        // Should we remember this candidate?
        if (reverseRemovalDecision)
            remove = !remove;
        if (!remove)
        {
            const reco::PFCandidate& cand = (*pfCandidates)[i];
            pOutput->push_back(cand);
            pOutput->back().setSourceCandidatePtr(candptr);
        }
    }

    iEvent.put(pOutput);
}


bool FFTJetPFPileupCleaner::isRemovable(
    const reco::PFCandidate::ParticleType ptype) const
{
    const unsigned shift = static_cast<unsigned>(ptype);
    assert(shift < 32U);
    return removalMask & (1U << shift);
}


void FFTJetPFPileupCleaner::setRemovalBit(
    const reco::PFCandidate::ParticleType ptype, const bool value)
{
    const unsigned shift = static_cast<unsigned>(ptype);
    assert(shift < 32U);
    const unsigned mask = (1U << shift);
    if (value)
        removalMask |= mask;
    else
        removalMask &= ~mask;
}


// The following essentially duplicates the code in PFPileUp.cc,
// with added cut on ndof and vertex Z position.
reco::VertexRef FFTJetPFPileupCleaner::findSomeVertex(
    const edm::Handle<reco::VertexCollection>& vertices,
    const reco::PFCandidate& pfcand) const
{  
    typedef reco::VertexCollection::const_iterator IV;
    typedef reco::Vertex::trackRef_iterator IT;

    reco::TrackBaseRef trackBaseRef(pfcand.trackRef());

    size_t iVertex = 0;
    unsigned index = 0;
    unsigned nFoundVertex = 0;
    double bestweight = 0;

    const IV vertend(vertices->end());
    for (IV iv=vertices->begin(); iv!=vertend; ++iv, ++index)
        if (isAcceptableVtx(iv))
        {
            const reco::Vertex& vtx = *iv;

            // loop on tracks in vertices
            IT trackend(vtx.tracks_end());
            for (IT iTrack=vtx.tracks_begin(); iTrack!=trackend; ++iTrack)
            {
                const reco::TrackBaseRef& baseRef = *iTrack;

                // one of the tracks in the vertex is the same as 
                // the track considered in the function
                if (baseRef == trackBaseRef)
                {
                    // select the vertex for which the track has the highest weight
                    const double w = vtx.trackWeight(baseRef);
                    if (w > bestweight)
                    {
                        bestweight=w;
                        iVertex=index;
                        nFoundVertex++;
                    } 	
                }
            }
        }

    if (nFoundVertex > 0)
    {
        if (nFoundVertex != 1)
            edm::LogWarning("TrackOnTwoVertex")
                << "a track is shared by at least two vertices. "
                << "Used to be an assert";
        return reco::VertexRef(vertices, iVertex);
    }

    // optional: as a secondary solution, associate the closest vertex in z
    if (checkClosestZVertex) 
    {
        const double ztrack = pfcand.vertex().z();
        bool foundVertex = false;
        index = 0;

        if (nZAssociations < 2U)
        {
            double dzmin = 10000;
            for (IV iv=vertices->begin(); iv!=vertend; ++iv, ++index)
                if (isAcceptableVtx(iv))
                {
                    const double dz = std::abs(ztrack - iv->z());
                    if (dz < dzmin)
                    {
                        dzmin = dz; 
                        iVertex = index;
                        foundVertex = true;
                    }
                }
        }
        else
        {
            zAssoc.clear();
            for (IV iv=vertices->begin(); iv!=vertend; ++iv, ++index)
                if (isAcceptableVtx(iv))
                {
                    const double dz = std::abs(ztrack - iv->z());
                    zAssoc.push_back(std::pair<double, unsigned>(dz, index));
                }

            // If the primary vertex is among first nZAssociations
            // vertices, return it. Otherwise return the best match.
            if (!zAssoc.empty())
            {
                std::sort(zAssoc.begin(), zAssoc.end());
                foundVertex = true;
                iVertex = zAssoc[0].second;

                const unsigned nVert = zAssoc.size();
                const unsigned maxInd = std::min(nZAssociations, nVert);

                for (unsigned i=0; i<maxInd; ++i)
                {
                    reco::VertexRef vertexref(vertices, zAssoc[i].second);
                    if (vertexref.key() == 0)
                    {
                        iVertex = zAssoc[i].second;
                        break;
                    }
                }
            }
        }

        if (foundVertex) 
            return reco::VertexRef(vertices, iVertex);
    }

    return reco::VertexRef();
}


void FFTJetPFPileupCleaner::buildRemovalMask()
{
    setRemovalBit(reco::PFCandidate::X,         remove_X        );
    setRemovalBit(reco::PFCandidate::h,         remove_h        );
    setRemovalBit(reco::PFCandidate::e,         remove_e        );
    setRemovalBit(reco::PFCandidate::mu,        remove_mu       );
    setRemovalBit(reco::PFCandidate::gamma,     remove_gamma    );
    setRemovalBit(reco::PFCandidate::h0,        remove_h0       );
    setRemovalBit(reco::PFCandidate::h_HF,      remove_h_HF     );
    setRemovalBit(reco::PFCandidate::egamma_HF, remove_egamma_HF);
}


// ------------ method called once each job just before starting event loop
void FFTJetPFPileupCleaner::beginJob()
{
}


// ------------ method called once each job just after ending the event loop
void FFTJetPFPileupCleaner::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetPFPileupCleaner);
