#ifndef RecoParticleFlow_PFClusterProducer_PFClusterAlgo_h
#define RecoParticleFlow_PFClusterProducer_PFClusterAlgo_h


#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

// for E/gamma position calc
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

#include <string>
#include <vector>
#include <map>
#include <set>

#include <memory>

class TFile;
class TH2F;

/// \brief Algorithm for particle flow clustering
/*!
  This class takes as an input a map of pointers to PFRecHit's, and creates 
  PFCluster's from these rechits. Clustering is implemented for ECAL, HCAL, 
  and preshower.

  \todo describe algorithm and parameters. give a use case

  \author Colin Bernet, Patrick Janot
  \date July 2006
*/

namespace edm {
  template<> 
  struct StrictWeakOrdering<reco::PFRecHit> {   
    typedef unsigned key_type;
    bool operator()(unsigned a, reco::PFRecHit const& b) const { 
      return a < b.detId(); 
    }
    bool operator()(reco::PFRecHit const& a, unsigned b) const { 
      return a.detId() < b; 
    }
    bool operator()(reco::PFRecHit const& a, reco::PFRecHit const& b) const { 
      return ( a.detId() < b.detId()  ); 
    }
  };
}

class PFClusterAlgo {

 public:

  typedef edm::StrictWeakOrdering<reco::PFRecHit> PFStrictWeakOrdering;
  typedef edm::SortedCollection<reco::PFRecHit> SortedPFRecHitCollection;
  typedef edm::Handle<edm::View<reco::PFCluster> > PFClusterHandle;

  enum PositionCalcType { EGPositionCalc,
			  EGPositionFormula,			  
			  PFPositionCalc,
			  kNotDefined };

  /// constructor
  PFClusterAlgo();

  /// destructor
  virtual ~PFClusterAlgo() {;}

  /// set hits on which clustering will be performed 
  // void init(const std::map<unsigned, reco::PFRecHit* >& rechits );

  /// enable/disable debugging
  void enableDebugging(bool debug) { debug_ = debug;}
 

  typedef edm::Handle< reco::PFRecHitCollection > PFRecHitHandle;
  
  /// perform clustering
  void doClustering( const reco::PFRecHitCollection& rechits );
  void doClustering( const reco::PFRecHitCollection& rechits, 
		     const std::vector<bool> & mask );

  /// perform clustering in full framework
  void doClustering( const PFRecHitHandle& rechitsHandle );  
  void doClustering( const PFRecHitHandle& rechitsHandle, 
		     const std::vector<bool> & mask );
    
  /// setters -------------------------------------------------------
  
  /// set barrel threshold
  void setThreshBarrel(double thresh) {threshBarrel_ = thresh;}
  void setThreshPtBarrel(double thresh) {threshPtBarrel_ = thresh;}

  /// set barrel seed threshold
  void setThreshSeedBarrel(double thresh) {threshSeedBarrel_ = thresh;}
  void setThreshPtSeedBarrel(double thresh) {threshPtSeedBarrel_ = thresh;}

  /// set barrel clean threshold
  void setThreshCleanBarrel(double thresh) {threshCleanBarrel_ = thresh;}
  void setS4S1CleanBarrel(const std::vector<double>& coeffs) {minS4S1Barrel_ = coeffs;}

  /// set endcap thresholds for double spike cleaning
  void setThreshDoubleSpikeBarrel( double thresh ) { threshDoubleSpikeBarrel_ = thresh;}
  void setS6S2DoubleSpikeBarrel( double cut ) { minS6S2DoubleSpikeBarrel_ = cut;}

  /// set  endcap threshold
  void setThreshEndcap(double thresh) {threshEndcap_ = thresh;}
  void setThreshPtEndcap(double thresh) {threshPtEndcap_ = thresh;}

  /// set  endcap seed threshold
  void setThreshSeedEndcap(double thresh) {threshSeedEndcap_ = thresh;}
  void setThreshPtSeedEndcap(double thresh) {threshPtSeedEndcap_ = thresh;}

  /// set endcap clean threshold
  void setThreshCleanEndcap(double thresh) {threshCleanEndcap_ = thresh;}
  void setS4S1CleanEndcap(const std::vector<double>& coeffs) {minS4S1Endcap_ = coeffs;}

  /// set endcap thresholds for double spike cleaning
  void setThreshDoubleSpikeEndcap( double thresh ) { threshDoubleSpikeEndcap_ = thresh;}
  void setS6S2DoubleSpikeEndcap( double cut ) { minS6S2DoubleSpikeEndcap_ = cut;}

  /// set endcap clean threshold
  void setHistos(TFile* file, TH2F* hB, TH2F* hE) {file_=file; hBNeighbour = hB; hENeighbour = hE;}

  /// set number of neighbours for  
  void setNNeighbours(int n) { nNeighbours_ = n;}

  // set position calculation variant
  void setPositionCalcType( const PositionCalcType& t ) {which_pos_calc_ = t;}

  // setup E/gamma position calc
  void setEGammaPosCalc( const edm::ParameterSet& conf ) { 
    eg_pos_calc.reset(new PositionCalc(conf));
  }
  void setEBGeom(const CaloSubdetectorGeometry* esh) {
    eb_geom = esh;
  }
  void setEEGeom(const CaloSubdetectorGeometry* esh) {
    ee_geom = esh;
  }
  void setPreshowerGeom(const CaloSubdetectorGeometry* esh) {
    preshower_geom = esh;
  }

  // set W0 parameter for EG-type position formula
  void setPosCalcW0( double w0 ) { param_W0_ = w0; }

  /// set p1 for position calculation
  void setPosCalcP1( double p1 ) { posCalcP1_ = p1; }

  /// set number of crystals for position calculation (-1 all,5, or 9)
  void setPosCalcNCrystal(int n) { posCalcNCrystal_ = n;}

  /// set shower sigma for 
  void setShowerSigma( double sigma ) { showerSigma_ = sigma;}

  /// activate use of cells with a common corner to build topo-clusters
  void setUseCornerCells( bool usecornercells ) { useCornerCells_ = usecornercells;}
  
  /// Activate cleaning of HCAL RBX's and HPD's
  void setCleanRBXandHPDs( bool cleanRBXandHPDs) { cleanRBXandHPDs_ = cleanRBXandHPDs; }

  /// getters -------------------------------------------------------
 
  /// get barrel threshold
  double threshBarrel() const {return threshBarrel_;}

  /// get barrel seed threshold
  double threshSeedBarrel() const  {return threshSeedBarrel_;}


  /// get  endcap threshold
  double threshEndcap() const {return threshEndcap_;}

  /// get  endcap seed threshold
  double threshSeedEndcap() const {return threshSeedEndcap_;}


  /// get number of neighbours for  
  int nNeighbours() const { return nNeighbours_;}

  /// get p1 for position calculation
  double posCalcP1() const { return posCalcP1_; }

  /// get number of crystals for position calculation (-1 all,5, or 9)
  int posCalcNCrystal() const {return  posCalcNCrystal_;}

  /// get shower sigma
  double showerSigma() const { return showerSigma_ ;}
  
  /// write histos
  void write();
  /// ----------------------------------------------------------------



  /// \return hit with index i. performs a test on the vector range
  const reco::PFRecHit& rechit(unsigned i,
			       const reco::PFRecHitCollection& rechits );

  /// \return mask flag for the rechit
  bool masked(unsigned rhi) const;

  enum Color { 
    NONE=0,
    SEED, 
    SPECIAL 
  };

  /// \return color of the rechit
  unsigned color(unsigned rhi) const;

  /// \return seed flag (not seed state) for rechit with index rhi
  bool isSeed(unsigned rhi) const;

  /// \return particle flow clusters
  std::auto_ptr< std::vector< reco::PFCluster > >& clusters()  
    {return pfClusters_;}

  /// \return cleaned rechits
  std::auto_ptr< std::vector< reco::PFRecHit > >& rechitsCleaned()  
    {return pfRecHitsCleaned_;}
  
  /// \return threshold, seed threshold, (gaussian width, p1 ??)
  /// for a given zone (endcap, barrel, VFCAL ??)

  enum Parameter { THRESH, 
		   SEED_THRESH,
		   PT_THRESH,
		   SEED_PT_THRESH,
                   CLEAN_THRESH,
                   CLEAN_S4S1,
                   DOUBLESPIKE_THRESH,
                   DOUBLESPIKE_S6S2
  };
  
  /// \return the value of a parameter of a given type, see enum Parameter,
  /// in a given layer. 
    double parameter( Parameter paramtype, PFLayer::Layer layer, unsigned iCoeff = 0, int iring0=0) const; 

  
  enum SeedState {
    UNKNOWN=-1,
    NO=0,
    YES=1,
    CLEAN=2
  };


  friend std::ostream& operator<<(std::ostream& out,const PFClusterAlgo& algo);

  typedef std::map<unsigned, unsigned >::const_iterator IDH;
  typedef std::multimap<double, unsigned >::iterator EH;
 

 private:
  /// perform clustering
  void doClusteringWorker( const reco::PFRecHitCollection& rechits );  

  /// Clean HCAL readout box noise and HPD discharge
  void cleanRBXAndHPD( const reco::PFRecHitCollection& rechits );

  /// look for seeds 
  void findSeeds( const reco::PFRecHitCollection& rechits );

  /// build topoclusters around seeds
  void buildTopoClusters( const reco::PFRecHitCollection& rechits ); 

  /// build a topocluster (recursive)
  void buildTopoCluster( std::vector< unsigned >& cluster, unsigned rhi, 
			 const reco::PFRecHitCollection& rechits ); 
  
  /// build PFClusters from a topocluster
  void buildPFClusters( const std::vector< unsigned >& cluster, 
			const reco::PFRecHitCollection& rechits ); 

  /// calculate position of a cluster
  void calculateClusterPosition( reco::PFCluster& cluster, 
                                 reco::PFCluster& clusterwodepthcor,
				 bool depcor = true,
				 int posCalcNCrystal=0);
  
  /// create a reference to a rechit. 
  /// in case  rechitsHandle_.isValid(), this reference is permanent.
  reco::PFRecHitRef  createRecHitRef( const reco::PFRecHitCollection& rechits, 
				      unsigned rhi );

  /// paint a rechit with a color. 
  void paint( unsigned rhi, unsigned color=1 );

  /// distance to a crack in the ECAL barrel in eta and phi direction
  std::pair<double,double> dCrack(double phi, double eta);  

  PFRecHitHandle           rechitsHandle_;   
  // for E\gamma Position calc
  std::auto_ptr<SortedPFRecHitCollection> sortedRecHits_; 

  /// ids of rechits used in seed search
  std::set<unsigned>       idUsedRecHits_;

  /// indices to rechits, sorted by decreasing E (not E_T)
  std::multimap<double, unsigned, std::greater<double> >  eRecHits_;

  /// mask, for all rechits. only masked rechits will be clustered
  /// by default, all rechits are masked. see setMask function.
  std::vector< bool >      mask_;

  /// color, for all rechits
  std::vector< unsigned >  color_;

  /// seed state, for all rechits
  std::vector< SeedState > seedStates_;

  /// used in topo cluster? for all rechits
  std::vector< bool >      usedInTopo_;

  /// vector of indices for seeds.   
  std::vector< unsigned >  seeds_; 

  /// sets of cells having one common side, and energy over threshold
  std::vector< std::vector< unsigned > > topoClusters_;

  /// all clusters
  // std::vector< reco::PFCluster >  allClusters_;

  /// particle flow clusters
  std::auto_ptr< std::vector<reco::PFCluster> > pfClusters_;
  
  /// particle flow rechits cleaned
  std::auto_ptr< std::vector<reco::PFRecHit> > pfRecHitsCleaned_;  

  ///  barrel threshold
  double threshBarrel_;
  double threshPtBarrel_;

  ///  barrel seed threshold
  double threshSeedBarrel_;
  double threshPtSeedBarrel_;

  ///  endcap threshold 
  double threshEndcap_;
  double threshPtEndcap_;

  ///  endcap seed threshold
  double threshSeedEndcap_;
  double threshPtSeedEndcap_;

  /// Barrel cleaning threshold and S4/S1 smallest fractiom
  double threshCleanBarrel_;
  std::vector<double> minS4S1Barrel_;

  /// Barrel double-spike cleaning
  double threshDoubleSpikeBarrel_;
  double minS6S2DoubleSpikeBarrel_;

  /// Endcap cleaning threshold and S4/S1 smallest fractiom
  double threshCleanEndcap_;
  std::vector<double> minS4S1Endcap_;

  /// Endcap double-spike cleaning
  double threshDoubleSpikeEndcap_;
  double minS6S2DoubleSpikeEndcap_;

  ///  number of neighbours
  int    nNeighbours_;

  ///  number of crystals for position calculation
  int    posCalcNCrystal_;

  /// parameter for position calculation  
  PositionCalcType which_pos_calc_;
  double posCalcP1_;
  double param_W0_; // for EG position calc variant
  std::auto_ptr<PositionCalc> eg_pos_calc; //directly use E/gamma pos calc
  const CaloSubdetectorGeometry *eb_geom, *ee_geom, *preshower_geom;

  /// sigma of shower (cm)
  double showerSigma_;

  /// option to use cells with a common corner to build topo-clusters
  bool useCornerCells_;

  /// option to clean HCAL RBX's and HPD's
  bool cleanRBXandHPDs_;

  /// debugging on/off
  bool   debug_;


  /// product number 
  static  unsigned  prodNum_;

  // Histograms
  TH2F* hBNeighbour;
  TH2F* hENeighbour;
  TFile*     file_; 

};

#endif
