#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalImagingAlgo_h
#define RecoLocalCalo_HGCalRecAlgos_HGCalImagingAlgo_h

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
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


#include "KDTreeLinkerAlgoT.h"


template <typename T>
std::vector<size_t> sorted_indices(const std::vector<T> &v) {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

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

HGCalImagingAlgo() : vecDeltas(), kappa(1.), ecut(0.), cluster_offset(0),
        sigma2(1.0),
        algoId(reco::CaloCluster::undefined),
        verbosity(pERROR),initialized(false){
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
        vecDeltas(vecDeltas_in), kappa(kappa_in),
        ecut(ecut_in),
        cluster_offset(0),
        sigma2(1.0),
        algoId(algoId_in),
        dependSensor(dependSensor_in),
        dEdXweights(dEdXweights_in),
        thicknessCorrection(thicknessCorrection_in),
        fcPerMip(fcPerMip_in),
        fcPerEle(fcPerEle_in),
        nonAgedNoises(nonAgedNoises_in),
        noiseMip(noiseMip_in),
        verbosity(the_verbosity),
        initialized(false),
        points(2*(maxlayer+1)),
        minpos(2*(maxlayer+1),{
                {0.0f,0.0f}
        }),
        maxpos(2*(maxlayer+1),{ {0.0f,0.0f} }),
        zees(2*(maxlayer+1),0.)
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
                 VerbosityLevel the_verbosity = pERROR) : vecDeltas(vecDeltas_in), kappa(kappa_in),
        ecut(ecut_in),
        cluster_offset(0),
        sigma2(std::pow(showerSigma,2.0)),
        algoId(algoId_in),
        dependSensor(dependSensor_in),
        dEdXweights(dEdXweights_in),
        thicknessCorrection(thicknessCorrection_in),
        fcPerMip(fcPerMip_in),
        fcPerEle(fcPerEle_in),
        nonAgedNoises(nonAgedNoises_in),
        noiseMip(noiseMip_in),
        verbosity(the_verbosity),
        initialized(false),
        points(2*(maxlayer+1)),
        minpos(2*(maxlayer+1),{
                {0.0f,0.0f}
        }),
        maxpos(2*(maxlayer+1),{ {0.0f,0.0f} }),
        zees(2*(maxlayer+1),0.)
{
}

virtual ~HGCalImagingAlgo()
{
}

void setVerbosity(VerbosityLevel the_verbosity)
{
        verbosity = the_verbosity;
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
        current_v.clear();
        clusters_v.clear();
        cluster_offset = 0;
        for( auto& it: points)
        {
                it.clear();
                std::vector<KDNode>().swap(it);
        }
        for(unsigned int i = 0; i < minpos.size(); i++)
        {
                minpos[i][0]=0.; minpos[i][1]=0.;
                maxpos[i][0]=0.; maxpos[i][1]=0.;
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
// maximum number of wafers per Layer: 666 (V7), 794 (V8)
static const unsigned int maxNumberOfWafersPerLayer = 794;

// The two parameters used to identify clusters
std::vector<double> vecDeltas;
double kappa;

// The hit energy cutoff
double ecut;

// The current offset into the temporary cluster structure
unsigned int cluster_offset;

// for energy sharing
double sigma2;   // transverse shower size

// The vector of clusters
std::vector<reco::BasicCluster> clusters_v;

hgcal::RecHitTools rhtools_;

// The algo id
reco::CaloCluster::AlgoId algoId;

// various parameters used for calculating the noise levels for a given sensor (and whether to use them)
bool dependSensor;
std::vector<double> dEdXweights;
std::vector<double> thicknessCorrection;
std::vector<double> fcPerMip;
double fcPerEle;
std::vector<double> nonAgedNoises;
double noiseMip;
std::vector<std::vector<double> >thresholds;

// The verbosity level
VerbosityLevel verbosity;

// initialization bool
bool initialized;

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
                const GlobalPoint position( std::move( tools->getPosition( detid ) ) );

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
                tools(0)
        {
        }
        bool operator > (const Hexel& rhs) const {
                return (rho > rhs.rho);
        }

};

typedef KDTreeLinkerAlgo<Hexel,2> KDTree;
typedef KDTreeNodeInfoT<Hexel,2> KDNode;


// A vector of vectors of KDNodes holding an Hexel in the clusters - to be used to build CaloClusters of DetIds
std::vector< std::vector<KDNode> > current_v;

std::vector<size_t> sort_by_delta(const std::vector<KDNode> &v){
        std::vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
        sort(idx.begin(), idx.end(),
             [&v](size_t i1, size_t i2) {
                        return v[i1].data.delta > v[i2].data.delta;
                });
        return idx;
}

std::vector<std::vector<KDNode> > points;   //a vector of vectors of hexels, one for each layer
//@@EM todo: the number of layers should be obtained programmatically - the range is 1-n instead of 0-n-1...

std::vector<std::array<float,2> > minpos;
std::vector<std::array<float,2> > maxpos;
std::vector<float> zees;


//these functions should be in a helper class.
inline double distance2(const Hexel &pt1, const Hexel &pt2) {   //distance squared
        const double dx = pt1.x - pt2.x;
        const double dy = pt1.y - pt2.y;
        return (dx*dx + dy*dy);
}   //distance squaredq
inline double distance(const Hexel &pt1, const Hexel &pt2) {   //2-d distance on the layer (x-y)
        return std::sqrt(distance2(pt1,pt2));
}
double calculateLocalDensity(std::vector<KDNode> &, KDTree &, const unsigned int);   //return max density
double calculateDistanceToHigher(std::vector<KDNode> &, KDTree &);
int findAndAssignClusters(std::vector<KDNode> &, KDTree &, double, KDTreeBox &, const int);
math::XYZPoint calculatePosition(std::vector<KDNode> &);

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
