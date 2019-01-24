#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalImagingAlgo_h
#define RecoLocalCalo_HGCalRecAlgos_HGCalImagingAlgo_h

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

// C/C++ headers
#include <string>
#include <vector>
#include <set>
#include <numeric>

#include "KDTreeLinkerAlgoT.h"


template <typename T>
std::vector<size_t> sorted_indices(const std::vector<T> &v) {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota (std::begin(idx), std::end(idx), 0);

        // sort indices based on comparing values in v
        std::sort(idx.begin(), idx.end(),
                  [&v](size_t i1, size_t i2) {
                return v[i1] > v[i2];
        });

        return idx;
}

class HGCalImagingAlgo
{


public:

enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 };

HGCalImagingAlgo() : vecDeltas_(), kappa_(1.), ecut_(0.),
        sigma2_(1.0),
        algoId_(reco::CaloCluster::undefined),
        verbosity_(pERROR),initialized_(false){
}

HGCalImagingAlgo(const std::vector<double>& vecDeltas_in, double kappa_in, double ecut_in,
                 reco::CaloCluster::AlgoId algoId_in,
                 bool dependSensor_in,
                 const std::vector<double>& dEdXweights_in,
                 const std::vector<double>& thicknessCorrection_in,
                 const std::vector<double>& fcPerMip_in,
                 double fcPerEle_in,
                 const std::vector<double>& nonAgedNoises_in,
                 double noiseMip_in,
                 VerbosityLevel the_verbosity = pERROR) :
        vecDeltas_(vecDeltas_in), kappa_(kappa_in),
        ecut_(ecut_in),
        sigma2_(1.0),
        algoId_(algoId_in),
        dependSensor_(dependSensor_in),
        dEdXweights_(dEdXweights_in),
        thicknessCorrection_(thicknessCorrection_in),
        fcPerMip_(fcPerMip_in),
        fcPerEle_(fcPerEle_in),
        nonAgedNoises_(nonAgedNoises_in),
        noiseMip_(noiseMip_in),
        verbosity_(the_verbosity),
        initialized_(false),
        points_(2*(maxlayer+1)),
        minpos_(2*(maxlayer+1),{
                {0.0f,0.0f}
        }),
        maxpos_(2*(maxlayer+1),{ {0.0f,0.0f} })
{
}

HGCalImagingAlgo(const std::vector<double>& vecDeltas_in, double kappa_in, double ecut_in,
                 double showerSigma,
                 reco::CaloCluster::AlgoId algoId_in,
                 bool dependSensor_in,
                 const std::vector<double>& dEdXweights_in,
                 const std::vector<double>& thicknessCorrection_in,
                 const std::vector<double>& fcPerMip_in,
                 double fcPerEle_in,
                 const std::vector<double>& nonAgedNoises_in,
                 double noiseMip_in,
                 VerbosityLevel the_verbosity = pERROR) : vecDeltas_(vecDeltas_in), kappa_(kappa_in),
        ecut_(ecut_in),
        sigma2_(std::pow(showerSigma,2.0)),
        algoId_(algoId_in),
        dependSensor_(dependSensor_in),
        dEdXweights_(dEdXweights_in),
        thicknessCorrection_(thicknessCorrection_in),
        fcPerMip_(fcPerMip_in),
        fcPerEle_(fcPerEle_in),
        nonAgedNoises_(nonAgedNoises_in),
        noiseMip_(noiseMip_in),
        verbosity_(the_verbosity),
        initialized_(false),
        points_(2*(maxlayer+1)),
	minpos_(2*(maxlayer+1),{
                {0.0f,0.0f}
        }),
	maxpos_(2*(maxlayer+1),{ {0.0f,0.0f} })
{
}

virtual ~HGCalImagingAlgo()
{
}

void setVerbosity(VerbosityLevel the_verbosity)
{
        verbosity_ = the_verbosity;
}

void populate(const HGCRecHitCollection &hits);
// this is the method that will start the clusterisation (it is possible to invoke this method more than once - but make sure it is with
// different hit collections (or else use reset)
void makeClusters();
// this is the method to get the cluster collection out
std::vector<reco::BasicCluster> getClusters(bool);
// needed to switch between EE and HE with the same algorithm object (to get a single cluster collection)
void getEventSetup(const edm::EventSetup& es){
        rhtools_.getEventSetup(es);
}
// use this if you want to reuse the same cluster object but don't want to accumulate clusters (hardly useful?)
void reset(){
        clusters_v_.clear();
        layerClustersPerLayer_.clear();
        for( auto& it: points_)
        {
                it.clear();
                std::vector<KDNode>().swap(it);
        }
        for(unsigned int i = 0; i < minpos_.size(); i++)
        {
                minpos_[i][0]=0.; minpos_[i][1]=0.;
                maxpos_[i][0]=0.; maxpos_[i][1]=0.;
        }
}
void computeThreshold();

/// point in the space
typedef math::XYZPoint Point;

//max number of layers
static const unsigned int maxlayer = 52;


private:
// last layer per subdetector
static const unsigned int lastLayerEE = 28;
static const unsigned int lastLayerFH = 40;

// The two parameters used to identify clusters
std::vector<double> vecDeltas_;
double kappa_;

// The hit energy cutoff
double ecut_;

// for energy sharing
double sigma2_;   // transverse shower size

// The vector of clusters
std::vector<reco::BasicCluster> clusters_v_;

hgcal::RecHitTools rhtools_;

// The algo id
reco::CaloCluster::AlgoId algoId_;

// various parameters used for calculating the noise levels for a given sensor (and whether to use them)
bool dependSensor_;
std::vector<double> dEdXweights_;
std::vector<double> thicknessCorrection_;
std::vector<double> fcPerMip_;
double fcPerEle_;
std::vector<double> nonAgedNoises_;
double noiseMip_;
std::vector<std::vector<double> > thresholds_;
std::vector<std::vector<double> > v_sigmaNoise_;

// The verbosity level
VerbosityLevel verbosity_;

// initialization bool
bool initialized_;

struct Hexel {

        double x;
        double y;
        double z;
        bool isHalfCell;
        double weight;
        double fraction;
        DetId detid;
        double rho;
        double delta;
        int nearestHigher;
        bool isBorder;
        bool isHalo;
        int clusterIndex;
        float sigmaNoise;
        float thickness;
        const hgcal::RecHitTools *tools;

        Hexel(const HGCRecHit &hit, DetId id_in, bool isHalf, float sigmaNoise_in, float thickness_in, const hgcal::RecHitTools *tools_in) :
                isHalfCell(isHalf),
                weight(0.), fraction(1.0), detid(id_in), rho(0.), delta(0.),
                nearestHigher(-1), isBorder(false), isHalo(false),
                clusterIndex(-1), sigmaNoise(sigmaNoise_in), thickness(thickness_in),
                tools(tools_in)
        {
                const GlobalPoint position( tools->getPosition( detid ) );
                weight = hit.energy();
                x = position.x();
                y = position.y();
                z = position.z();
        }
        Hexel() :
                x(0.),y(0.),z(0.),isHalfCell(false),
                weight(0.), fraction(1.0), detid(), rho(0.), delta(0.),
                nearestHigher(-1), isBorder(false), isHalo(false),
                clusterIndex(-1),
                sigmaNoise(0.),
                thickness(0.),
                tools(nullptr)
        {
        }
        bool operator > (const Hexel& rhs) const {
                return (rho > rhs.rho);
        }

};

typedef KDTreeLinkerAlgo<Hexel,2> KDTree;
typedef KDTreeNodeInfoT<Hexel,2> KDNode;


std::vector<std::vector<std::vector< KDNode> > > layerClustersPerLayer_;

std::vector<size_t> sort_by_delta(const std::vector<KDNode> &v) const {
        std::vector<size_t> idx(v.size());
        std::iota (std::begin(idx), std::end(idx), 0);
        sort(idx.begin(), idx.end(),
             [&v](size_t i1, size_t i2) {
                        return v[i1].data.delta > v[i2].data.delta;
                });
        return idx;
}

std::vector<std::vector<KDNode> > points_;   //a vector of vectors of hexels, one for each layer
//@@EM todo: the number of layers should be obtained programmatically - the range is 1-n instead of 0-n-1...

std::vector<std::array<float,2> > minpos_;
std::vector<std::array<float,2> > maxpos_;


//these functions should be in a helper class.
inline double distance2(const Hexel &pt1, const Hexel &pt2) const{   //distance squared
        const double dx = pt1.x - pt2.x;
        const double dy = pt1.y - pt2.y;
        return (dx*dx + dy*dy);
}   //distance squaredq
inline double distance(const Hexel &pt1, const Hexel &pt2) const{   //2-d distance on the layer (x-y)
        return std::sqrt(distance2(pt1,pt2));
}
double calculateLocalDensity(std::vector<KDNode> &, KDTree &, const unsigned int) const;   //return max density
double calculateDistanceToHigher(std::vector<KDNode> &) const;
int findAndAssignClusters(std::vector<KDNode> &, KDTree &, double, KDTreeBox &, const unsigned int, std::vector<std::vector<KDNode> >&) const;
math::XYZPoint calculatePosition(std::vector<KDNode> &) const;

// attempt to find subclusters within a given set of hexels
std::vector<unsigned> findLocalMaximaInCluster(const std::vector<KDNode>&);
math::XYZPoint calculatePositionWithFraction(const std::vector<KDNode>&, const std::vector<double>&);
double calculateEnergyWithFraction(const std::vector<KDNode>&, const std::vector<double>&);
// outputs
void shareEnergy(const std::vector<KDNode>&,
                 const std::vector<unsigned>&,
                 std::vector<std::vector<double> >&);
};

#endif
