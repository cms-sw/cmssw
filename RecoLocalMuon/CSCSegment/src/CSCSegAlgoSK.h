#ifndef CSCSegment_CSCSegAlgoSK_h
#define CSCSegment_CSCSegAlgoSK_h

/**
 * \class CSCSegAlgoSK
 *
 * This is the original algorithm for building endcap muon track segments
 * out of the rechit's in a CSCChamber. cf. CSCSegmentizerTC.<BR>
 * 'SK' = 'Sasha Khanov' = Speed King <BR>
 *
 * A CSCSegment is a RecSegment4D, and is built from
 * CSCRecHit2D objects, each of which is a RecHit2DLocalPos. <BR>
 *
 * This class is used by the CSCSegmentAlgorithm. <BR>
 * Alternative algorithms can be used for the segment building
 * by writing classes like this, and then selecting which one is actually
 * used via the CSCSegmentBuilder. <BR>
 *
 * Original (in FORTRAN): Alexandre.Khanov@cern.ch <BR>
 * Ported to C++ and improved: Rick.Wilkinson@cern.ch <BR>
 * Reimplemented in terms of layer index, and bug fix: Tim.Cox@cern.ch <BR>
 * Ported to CMSSW 2006-04-03: Matteo.Sani@cern.ch <BR>
 *
 *  $Date: 2013/05/28 15:41:45 $
 *  $Revision: 1.11 $
 *  \author M. Sani
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>

//#include <DataFormats/GeometryVector/interface/GlobalPoint.h>

#include <deque>
#include <vector>

class CSCSegAlgoSK : public CSCSegmentAlgorithm {

public:

    // Tim tried using map as basic container of all (space-point) RecHit's in a chamber:
    // The 'key' is a pseudo-layer number (1-6 but with 1 always closest to IP).
    // The 'value' is a vector of the RecHit's on that layer.
    // Using the layer number like this removes the need to sort in global z.
    // Instead we just have to ensure the layer index is correctly adjusted 
    // to enforce the requirement that 'layer 1' is closest in the chamber
    // to the IP.
    
    /// Typedefs
    
    typedef std::vector<int> LayerIndex;
    typedef std::vector<const CSCRecHit2D*> ChamberHitContainer;
    typedef std::vector<const CSCRecHit2D*>::const_iterator ChamberHitContainerCIt;

    // We need to be able to flag a hit as 'used' and so need a container
    // of bool's. Naively, this would be vector<bool>... but AVOID that since it's
    // non-standard i.e. packed-bit implementation which is not a standard STL container. 
    // We don't need what it offers and it could lead to unexpected trouble in the future.

    typedef std::deque<bool> BoolContainer;
    
    /// Constructor
    explicit CSCSegAlgoSK(const edm::ParameterSet& ps);
    /// Destructor
    virtual ~CSCSegAlgoSK() {};

    /**
     * Build track segments in this chamber (this is where the actual
     * segment-building algorithm hides.)
     */
    std::vector<CSCSegment> buildSegments(const ChamberHitContainer& rechits);

    /**
     * Here we must implement the algorithm
     */
    std::vector<CSCSegment> run(const CSCChamber* aChamber, const ChamberHitContainer& rechits); 

private:

    /// Utility functions 
    // Could be static at the moment, but in principle one
    // might like CSCSegmentizer-specific behaviour?
    bool areHitsCloseInLocalX(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const;
    bool areHitsCloseInGlobalPhi(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const;
    bool isHitNearSegment(const CSCRecHit2D* h) const;

    /**
     * Dump position and phi of each rechit in chamber after sort in z
     */
    void dumpHits(const ChamberHitContainer& rechits) const;

    /**
     * Try adding non-used hits to segment
     */
    void tryAddingHitsToSegment(const ChamberHitContainer& rechitsInChamber,
        const BoolContainer& used, const LayerIndex& layerIndex,
        const ChamberHitContainerCIt i1, const ChamberHitContainerCIt i2);

    /**
     * Return true if segment is 'good'.
     * In this algorithm, 'good' means has sufficient hits
     */
    bool isSegmentGood(const ChamberHitContainer& rechitsInChamber) const;

    /**
     * Flag hits on segment as used
     */
    void flagHitsAsUsed(const ChamberHitContainer& rechitsInChamber, BoolContainer& used) const;
	
    /// Utility functions 	
    bool addHit(const CSCRecHit2D* hit, int layer);
    void updateParameters(void);
    void fitSlopes(void);
    void fillChiSquared(void);
    /**
     * Always enforce direction of segment to point from IP outwards
     * (Incorrect for particles not coming from IP, of course.)
     */
    void fillLocalDirection(void);
    float phiAtZ(float z) const;
    bool hasHitOnLayer(int layer) const;
    bool replaceHit(const CSCRecHit2D* h, int layer);
    void compareProtoSegment(const CSCRecHit2D* h, int layer);
    void increaseProtoSegment(const CSCRecHit2D* h, int layer);
    CLHEP::HepMatrix derivativeMatrix(void) const;
    AlgebraicSymMatrix weightMatrix(void) const;
    AlgebraicSymMatrix calculateError(void) const;
    void flipErrors(AlgebraicSymMatrix&) const;
		
    // Member variables
    // ================

    const CSCChamber* theChamber;
    ChamberHitContainer proto_segment;
    const std::string myName; 
		
    double theChi2;
    LocalPoint theOrigin;
    LocalVector theDirection;
    float uz, vz;
    float windowScale;
    float dRPhiMax ;
    float dPhiMax;
    float dRPhiFineMax;
    float dPhiFineMax;
    float chi2Max;
    float wideSeg;
    int minLayersApart;
    bool debugInfo;
};

#endif
