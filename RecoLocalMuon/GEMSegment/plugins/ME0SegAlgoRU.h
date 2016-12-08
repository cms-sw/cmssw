#ifndef ME0Segment_ME0SegAlgoRU_h
#define ME0Segment_ME0SegAlgoRU_h

/**
 * \class ME0SegAlgoRU
 * adapted from CSC to ME0 bt Marcello Maggi
 *
 * This is the original algorithm for building endcap muon track segments
 * out of the rechit's in a ME0Chamber
 * 'RU' = 'RUssia' = Road Usage
 *
 * A ME0Segment is a RecSegment4D, and is built from
 * ME0RecHit2D objects, each of which is a RecHit2DLocalPos. <BR>
 *
 * This class is used by the ME0SegmentAlgorithm. <BR>
 * Alternative algorithms can be used for the segment building
 * by writing classes like this, and then selecting which one is actually
 * used via the ME0SegmentBuilder. <BR>
 *
 * developed and implemented by Vladimir Palichik <Vladimir.Paltchik@cern.ch>
 *                          and Nikolay Voytishin <nikolay.voytishin@cern.ch>
 */

#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentAlgorithmBase.h>
#include <DataFormats/GEMRecHit/interface/ME0RecHit.h>
#include "MuonSegFit.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include <vector>

class MuonSegFit;

class ME0SegAlgoRU : public ME0SegmentAlgorithmBase {

public:

    // Tim tried using map as basic container of all (space-point) RecHit's in a chamber:
    // The 'key' is a pseudo-layer number (1-6 but with 1 always closest to IP).
    // The 'value' is a vector of the RecHit's on that layer.
    // Using the layer number like this removes the need to sort in global z.
    // Instead we just have to ensure the layer index is correctly adjusted 
    // to enforce the requirement that 'layer 1' is closest in the chamber
    // to the IP.
    // For ME0 those are 6 different ME0Chamber forming an ME0Ensemble

    /// Typedefs
  
  
    // 4-dim vector                                                                                                                                                                   
    typedef ROOT::Math::SVector<double,6> SVector6;
    typedef std::vector<int> LayerIndex;
    typedef EnsembleHitContainer::const_iterator EnsembleHitContainerCIt;

    // We need to be able to flag a hit as 'used' and so need a container of bool's. 
    typedef std::vector<bool> BoolContainer;
    
    /// Constructor
    explicit ME0SegAlgoRU(const edm::ParameterSet& ps);
    /// Destructor
    virtual ~ME0SegAlgoRU() {};

    /**
     * Build track segments in this chamber (this is where the actual
     * segment-building algorithm hides.)
     */
    std::vector<ME0Segment> buildSegments(const EnsembleHitContainer& rechits);

    //    std::vector<ME0Segment> assambleRechitsInSegments(const EnsembleHitContainer& rechits, int iadd, BoolContainer& used, BoolContainer& used3p, int *recHits_per_layer, const LayerIndex& layerIndex, std::vector<ME0Segment> segments);

    /**
     * Here we must implement the algorithm
     */
    std::vector<ME0Segment> run(const ME0Ensemble& ensemble, const EnsembleHitContainer& rechits); 

private:

    /// Utility functions 
    // Could be static at the moment, but in principle one
    // might like ME0Segmentizer-specific behaviour?
    bool areHitsCloseInEta(const ME0RecHit* h1, const ME0RecHit* h2) const;
    bool areHitsCloseInGlobalPhi(const ME0RecHit* h1, const ME0RecHit* h2) const;
    bool isHitNearSegment(const ME0RecHit* h) const;

    /**
     * Try adding non-used hits to segment
     */
    void tryAddingHitsToSegment(const EnsembleHitContainer& rechitsInChamber,
	const BoolContainer& used, const LayerIndex& layerIndex,
        const EnsembleHitContainerCIt i1, const EnsembleHitContainerCIt i2);

    /**
     * Return true if segment is 'good'.
     * In this algorithm, 'good' means has sufficient hits
     */
    bool isSegmentGood(const EnsembleHitContainer& rechitsInChamber) const;

    /**
     * Flag hits on segment as used
     */
    void flagHitsAsUsed(const EnsembleHitContainer& rechitsInChamber, BoolContainer& used) const;
	
    /// Utility functions 	
    bool addHit(const ME0RecHit* hit, int layer);
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
    bool replaceHit(const ME0RecHit* h, int layer);
    void compareProtoSegment(const ME0RecHit* h, int layer);
    void increaseProtoSegment(const ME0RecHit* h, int layer, int chi2_factor);

		
    // Member variables
    // ================

    ME0Ensemble theEnsemble;
    EnsembleHitContainer proto_segment;
    const std::string myName; 
		
    double theChi2;
    LocalPoint theOrigin;
    LocalVector theDirection;
    float uz, vz;
    float windowScale;
    int chi2D_iadd=1;
    int strip_iadd=1;	
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

    std::unique_ptr<MuonSegFit> sfit_;
};

// functor to sort rechits in the segment
struct  sortByLayer{
  sortByLayer(){}
  bool operator()(const ME0RecHit* a, const ME0RecHit* b) const{
    int layer1 = a->me0Id().layer();
    int layer2 = b->me0Id().layer();
    return  layer1 < layer2;
  }
};
#endif
