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
// $Id: FFTJetPFPileupCleaner.cc,v 1.5 2012/06/22 07:23:21 igv Exp $
//
//
#include <cmath>
#include <climits>
#include <utility>
#include <algorithm>

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
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

    reco::VertexRef findSomeVertexWFakes(
        const edm::Handle<reco::VertexCollection>& vertices,
        const edm::Handle<reco::VertexCollection>& fakeVertices,
        const reco::PFCandidate& pfcand, bool* fromFakeSet) const;

    edm::InputTag PFCandidates;
    edm::InputTag Vertices;
    edm::InputTag FakePrimaryVertices;

    // The following, if true, will switch to an algorithm
    // which takes a fake primary vertex into account
    bool useFakePrimaryVertex;

    // The following, if true, will cause association of a candidate
    // with some vertex no matter what
    bool checkClosestZVertex;

    // The following switch will check if the primary vertex
    // is a neighbor (in Z) of a track and will keep the
    // track if this is so, even if it is not directly associated
    // with the primary vertex. This switch is meaningful only if
    // "checkClosestZVertex" is true.
    bool keepIfPVneighbor;

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

    // Min and max eta for keeping things
    double etaMin;
    double etaMax;

    // Cut for the vertex Ndof
    double vertexNdofCut;

    // Cut for the vertex Z
    double vertexZmaxCut;

    // Vector for associating tracks with Z positions of the vertices
    mutable std::vector<std::pair<double, unsigned> > zAssoc;
};

//
// constructors and destructor
//
FFTJetPFPileupCleaner::FFTJetPFPileupCleaner(const edm::ParameterSet& ps)
    : init_param(edm::InputTag, PFCandidates),
      init_param(edm::InputTag, Vertices),
      init_param(edm::InputTag, FakePrimaryVertices),
      init_param(bool, useFakePrimaryVertex),
      init_param(bool, checkClosestZVertex),
      init_param(bool, keepIfPVneighbor),
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
       init_param(double, etaMin),
      init_param(double, etaMax),
      init_param(double, vertexNdofCut),
      init_param(double, vertexZmaxCut)
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
            std::abs(iv->z()) < vertexZmaxCut;
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

    edm::Handle<reco::VertexCollection> fakeVertices;
    if (useFakePrimaryVertex)
    {
        iEvent.getByLabel(FakePrimaryVertices, fakeVertices);
        if (!fakeVertices.isValid())
            throw cms::Exception("FFTJetBadConfig")
                << "ERROR in FFTJetPFPileupCleaner:"
                " could not find fake vertices"
                << std::endl;
    }

    const unsigned ncand = pfCandidates->size();
    for (unsigned i=0; i<ncand; ++i)
    {
        reco::PFCandidatePtr candptr(pfCandidates, i);
        bool remove = false;

        if (isRemovable(candptr->particleId()))
        {
            bool fromFakeSet = false;
            reco::VertexRef vertexref(findSomeVertexWFakes(vertices, fakeVertices,
                                                           *candptr, &fromFakeSet));
            if (vertexref.isNull())
            {
                // Track is not associated with any vertex 
                // in any of the vertex sets
                remove = removeUnassociated;
            }
            else if (vertexref.key() == 0 && 
                     (!useFakePrimaryVertex || fromFakeSet))
            {
                // Track is associated with the main primary vertex
                // However, if we are using fake vertices, this only
                // matters if the vertex comes from the fake set. If
                // it comes from the real set, remove the track anyway
                // because in the combined set the associated vertex
                // would not be the main primary vertex.
                remove = removeMainVertex;
            }
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


// The following essentially follows the code in PFPileUp.cc,
// with added cut on ndof, vertex Z position, and iteration
// over fakes
reco::VertexRef FFTJetPFPileupCleaner::findSomeVertexWFakes(
    const edm::Handle<reco::VertexCollection>& vertices,
    const edm::Handle<reco::VertexCollection>& fakeVertices,
    const reco::PFCandidate& pfcand, bool* fromFakeSet) const
{
    typedef reco::VertexCollection::const_iterator IV;
    typedef reco::Vertex::trackRef_iterator IT;

    *fromFakeSet = false;
    reco::TrackBaseRef trackBaseRef(pfcand.trackRef());

    size_t iVertex = 0;
    unsigned nFoundVertex = 0;
    const IV vertend(vertices->end());

    {
        unsigned index = 0;
        double bestweight = 0.0;
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
    }

    if (nFoundVertex > 0)
    {
        if (nFoundVertex != 1)
            edm::LogWarning("TrackOnTwoVertex")
                << "a track is shared by at least two vertices. "
                << "Used to be an assert";

        // Check if we can re-associate this track with one
        // of the fake vertices
        if (useFakePrimaryVertex)
        {
            const double ztrack = pfcand.vertex().z();
            double dzmin = std::abs(ztrack - ((*vertices)[iVertex]).z());

            const IV fakeEnd(fakeVertices->end());
            unsigned index = 0;
            for (IV iv=fakeVertices->begin(); iv!=fakeEnd; ++iv, ++index)
                if (isAcceptableVtx(iv))
                {
                    const double dz = std::abs(ztrack - iv->z());
                    if (dz < dzmin)
                    {
                        dzmin = dz; 
                        iVertex = index;
                        *fromFakeSet = true;
                    }
                }
        }

        return reco::VertexRef(*fromFakeSet ? fakeVertices : vertices, iVertex);
    }

    // optional: as a secondary solution, associate the closest vertex in z
    if (checkClosestZVertex) 
    {
        const double ztrack = pfcand.vertex().z();
        bool foundVertex = false;

        if (keepIfPVneighbor)
        {
            // Sort all vertices according to their Z coordinate
            zAssoc.clear();
            unsigned index = 0;
            for (IV iv=vertices->begin(); iv!=vertend; ++iv, ++index)
                if (isAcceptableVtx(iv))
                    zAssoc.push_back(std::pair<double,unsigned>(iv->z(), index));
            const unsigned numRealVertices = index;

            // Mix the fake vertex collection into zAssoc.
            // Note that we do not reset "index" before doing this.
            if (useFakePrimaryVertex)
            {
                const IV fakeEnd(fakeVertices->end());
                for (IV iv=fakeVertices->begin(); iv!=fakeEnd; ++iv, ++index)
                    if (isAcceptableVtx(iv))
                        zAssoc.push_back(std::pair<double,unsigned>(iv->z(), index));
            }

            // Check where the track z position fits into this sequence
            if (!zAssoc.empty())
            {
                std::sort(zAssoc.begin(), zAssoc.end());
                std::pair<double,unsigned> tPair(ztrack, UINT_MAX);
                const unsigned iAbove = std::upper_bound(
                    zAssoc.begin(), zAssoc.end(), tPair) - zAssoc.begin();

                // Check whether one of the vertices with indices
                // iAbove or (iAbove - 1) is a primary vertex.
                // If so, return it. Otherwise return the one
                // with closest distance to the track.
                unsigned ich[2] = {0U, 0U};
                unsigned nch = 1;
                if (iAbove)
                {
                    ich[0] = iAbove - 1U;
                    ich[1] = iAbove;
                    if (iAbove < zAssoc.size())
                        nch = 2;
                }

                double dzmin = 1.0e100;
                unsigned bestVertexNum = UINT_MAX;
                for (unsigned icheck=0; icheck<nch; ++icheck)
                {
                    const unsigned zAssocIndex = ich[icheck];
                    const unsigned vertexNum = zAssoc[zAssocIndex].second;

                    if (vertexNum == numRealVertices || 
                        (!useFakePrimaryVertex && vertexNum == 0U))
                    {
                        bestVertexNum = vertexNum;
                        break;
                    }

                    const double dz = std::abs(ztrack - zAssoc[zAssocIndex].first);
                    if (dz < dzmin)
                    {
                        dzmin = dz; 
                        bestVertexNum = vertexNum;
                    }
                }

                foundVertex = bestVertexNum < UINT_MAX;
                if (foundVertex)
                {
                    iVertex = bestVertexNum;
                    if (iVertex >= numRealVertices)
                    {
                        *fromFakeSet = true;
                        iVertex -= numRealVertices;
                    }
                }
            }
        }
        else
        {
            // This is a simple association algorithm (from PFPileUp)
            // extended to take fake vertices into account
            double dzmin = 1.0e100;
            unsigned index = 0;
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

            if (useFakePrimaryVertex)
            {
                const IV fakeEnd(fakeVertices->end());
                index = 0;
                for (IV iv=fakeVertices->begin(); iv!=fakeEnd; ++iv, ++index)
                    if (isAcceptableVtx(iv))
                    {
                        const double dz = std::abs(ztrack - iv->z());
                        if (dz < dzmin)
                        {
                            dzmin = dz; 
                            iVertex = index;
                            *fromFakeSet = true;
                            foundVertex = true;
                       }
                    }
            }
        }

        if (foundVertex) 
            return reco::VertexRef(*fromFakeSet ? fakeVertices : vertices, iVertex);
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
