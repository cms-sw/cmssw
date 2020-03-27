///=== This is the base class for the linearised chi-squared track fit algorithms.

///=== Written by: Sioni Summers and Alexander D. Morton

#ifndef __L1_CHI_SQUARED__
#define __L1_CHI_SQUARED__
 
#include "L1Trigger/TrackFindingTMTT/interface/Matrix.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include <vector>
#include <map>
#include <utility>
 
 namespace TMTT {

 
class L1ChiSquared : public TrackFitGeneric{
public:
    L1ChiSquared(const Settings* settings, const uint nPar);
 
    virtual ~L1ChiSquared(){}
 
    L1fittedTrack fit(const L1track3D& l1track3D);
 
protected:
    /* Methods */
    virtual std::vector<double> seed(const L1track3D& l1track3D)=0;
    virtual std::vector<double> residuals(std::vector<double> x)=0;
    virtual Matrix<double> D(std::vector<double> x)=0; // derivatives
    virtual Matrix<double> Vinv()=0; // Covariances
    virtual std::map<std::string, double> convertParams(std::vector<double> x)=0;
 
    /* Variables */
    std::vector<const Stub*> stubs_;
    std::map<std::string, double> trackParams_;
    uint nPar_;
    float largestresid_;
    int ilargestresid_;
    double chiSq_;
 
private:

    void calculateChiSq( std::vector<double> resids );
    void calculateDeltaChiSq( std::vector<double> deltaX, std::vector<double> covX );

    int numFittingIterations_;
    int killTrackFitWorstHit_;
    double generalResidualCut_;
    double killingResidualCut_;

    unsigned int minStubLayers_;
    unsigned int minStubLayersRed_; 
};

}

#endif
 

