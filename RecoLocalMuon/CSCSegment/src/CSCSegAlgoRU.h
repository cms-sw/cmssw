#ifndef CSCSegment_CSCSegAlgoRU_h
#define CSCSegment_CSCSegAlgoRU_h

/**
 * \class CSCSegAlgoRU
 *
 * This is the original algorithm for building endcap muon track segments
 * out of the rechit's in a CSCChamber
 * 'RU' = 'RUssia' = Road Usage
 *
 * A CSCSegment is a RecSegment4D, and is built from
 * CSCRecHit2D objects, each of which is a RecHit2DLocalPos. <BR>
 *
 * This class is used by the CSCSegmentAlgorithm. <BR>
 * Alternative algorithms can be used for the segment building
 * by writing classes like this, and then selecting which one is actually
 * used via the CSCSegmentBuilder. <BR>
 *
 * developed and implemented by Vladimir Palichik <Vladimir.Paltchik@cern.ch>
 *                          and Nikolay Voytishin <nikolay.voytishin@cern.ch>
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include "CSCSegFit.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include <vector>


class CSCSegFit;

class CSCSegAlgoRU : public CSCSegmentAlgorithm {

public:

    // Tim tried using map as basic container of all (space-point) RecHit's in a chamber:
    // The 'key' is a pseudo-layer number (1-6 but with 1 always closest to IP).
    // The 'value' is a vector of the RecHit's on that layer.
    // Using the layer number like this removes the need to sort in global z.
    // Instead we just have to ensure the layer index is correctly adjusted 
    // to enforce the requirement that 'layer 1' is closest in the chamber
    // to the IP.
    
    /// Typedefs
  
  
    // 4-dim vector                                                                                                                                                                   
    typedef ROOT::Math::SVector<double,6> SVector6;

    typedef std::vector<int> LayerIndex;
    typedef std::vector<const CSCRecHit2D*> ChamberHitContainer;
    typedef std::vector<const CSCRecHit2D*>::const_iterator ChamberHitContainerCIt;

    // We need to be able to flag a hit as 'used' and so need a container
    // of bool's. Naively, this would be vector<bool>... but AVOID that since it's
    // non-standard i.e. packed-bit implementation which is not a standard STL container. 
    // We don't need what it offers and it could lead to unexpected trouble in the future.

    typedef std::vector<bool> BoolContainer;
    
    /// Constructor
    explicit CSCSegAlgoRU(const edm::ParameterSet& ps);
    /// Destructor
    virtual ~CSCSegAlgoRU() {};

    /**
     * Build track segments in this chamber (this is where the actual
     * segment-building algorithm hides.)
     */
    std::vector<CSCSegment> buildSegments(const ChamberHitContainer& rechits);

    //    std::vector<CSCSegment> assambleRechitsInSegments(const ChamberHitContainer& rechits, int iadd, BoolContainer& used, BoolContainer& used3p, int *recHits_per_layer, const LayerIndex& layerIndex, std::vector<CSCSegment> segments);

    /**
     * Here we must implement the algorithm
     */
    std::vector<CSCSegment> run(const CSCChamber* aChamber, const ChamberHitContainer& rechits); 

private:

    /// Utility functions 
    // Could be static at the moment, but in principle one
    // might like CSCSegmentizer-specific behaviour?
    bool areHitsCloseInR(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const;
    bool areHitsCloseInGlobalPhi(const CSCRecHit2D* h1, const CSCRecHit2D* h2) const;
    bool isHitNearSegment(const CSCRecHit2D* h) const;

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
    void flagHitsAsUsed(const ChamberHitContainer& rechitsInChamber,BoolContainer& used) const;
	
    /// Utility functions 	
    bool addHit(const CSCRecHit2D* hit, int layer);
    void updateParameters(void);
    float fit_r_phi(SVector6 points, int layer) const;
    float fitX(SVector6 points, SVector6 errors, int ir, int ir2, float &chi2_str);
    void baseline(int n_seg_min);//function for arasing bad hits in case of bad chi2/NDOF 
   /**
     * Always enforce direction of segment to point from IP outwards
     * (Incorrect for particles not coming from IP, of course.)
     */
    float phiAtZ(float z) const;
    bool hasHitOnLayer(int layer) const;
    bool replaceHit(const CSCRecHit2D* h, int layer);
    void compareProtoSegment(const CSCRecHit2D* h, int layer);
    void increaseProtoSegment(const CSCRecHit2D* h, int layer, int chi2_factor);

		
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
    int chi2D_iadd;
    int strip_iadd;	
    bool doCollisions;
    float dRMax ;
    float dPhiMax;
    float dRIntMax;
    float dPhiIntMax;
    float chi2Max;
    float chi2_str_;
    float chi2Norm_2D_; 
    float wideSeg;
    int minLayersApart;
    bool debugInfo;

        std::unique_ptr<CSCSegFit> sfit_;
	//CSCSegFit* sfit_;

};

#endif
