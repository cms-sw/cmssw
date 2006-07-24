#ifndef Demo_PFRootEvent_PFRootEventManager_h
#define Demo_PFRootEvent_PFRootEventManager_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

#include <TObject.h>
#include "TEllipse.h"
#include "TBox.h"

#include <string>
#include <map>
#include <set>
#include <vector>

class TTree;
class TBranch;
class TFile;
class TCanvas;
class TH2F;
class TGraph;
class IO;

class PFRootEventManager {

 public:
  enum View_t { XY = 0, RZ = 1, NViews = 2 };

  PFRootEventManager();
  PFRootEventManager(const char* file);
  virtual ~PFRootEventManager();
  
  /// reset before next event
  void Reset();

  /// parse options
  void ReadOptions(const char* file, bool refresh=true);

  /// process one entry 
  virtual bool ProcessEntry(int entry);

  /// performs clustering 
  void Clustering();

/*   /// display rechit  */
/*   void DisplayRecHit( const reco::PFRecHit& rh ); */

  /// display one entry 
  void Display(int ientry);

  /// display eta/phi
  void DisplayEtaPhi();

  /// display x/y or r/z
  void DisplayView(unsigned viewType);

  /// display rechits
  void DisplayRecHitsEtaPhi();

  /// display rechit
  void DisplayRecHitEtaPhi(reco::PFRecHit& rh,
			   double maxe, double thresh);

  void DisplayClustersEtaPhi();

  void DisplayClusterEtaPhi(const reco::PFCluster& cluster);

  /// display x/y or r/z

  /// display reconstructed calorimeter hits in x/y or r/z view
  void DisplayRecHits(unsigned viewType, double phi0 = 0.);

  /// display a reconstructed calorimeter hit in x/y or r/z view
  void DisplayRecHit(reco::PFRecHit& rh, unsigned viewType,
		     double maxe, double thresh, double phi0 = 0.);

  /// display clusters in x/y or r/z view
  void DisplayClusters(unsigned viewType, double phi0 = 0.);

  /// display reconstructed tracks in x/y or r/z view
  void DisplayRecTracks(unsigned viewType, double phi0 = 0.);

  /// finds max rechit energy in a given layer 
  double GetMaxE(int layer) const;

  /// max rechit energy in ecal 
  double GetMaxEEcal();

  /// max rechit energy in hcal 
  double GetMaxEHcal();


  

  /// current event
  int        iEvent_;
  
  /// options file parser 
  IO*        options_;      
  
  /// input tree  
  TTree*     tree_;          
  
  /// cluster branch  
  TBranch*   hitsBranch_;          
  
  /// reconstructed tracks branch  
  TBranch*   recTracksBranch_;          
  
  // rechits
  std::vector<reco::PFRecHit> rechits_;

  // clusters
  std::vector<reco::PFCluster> clusters_;

  // reconstructed tracks
  std::vector<reco::PFRecTrack> recTracks_;

  /// input file
  TFile*     file_; 

  /// input file name
  std::string     inFileName_;   

  /// output file
  TFile*     outFile_;   

  /// output filename
  std::string     outFileName_;   


  /// canvases for eta/phi display, one per algo
  /// each is split in 2 : HCAL, ECAL
  std::map<int, TCanvas* > displayEtaPhi_;        

  /// algos to display
  std::set<int>            algosToDisplay_;  

  /// display pad xy size for eta/phi view
  std::vector<int>         viewSizeEtaPhi_;        

  /// support histogram for eta/phi display
  TH2F*                    displayHistEtaPhi_;

  /// vector of canvas for x/y or r/z display
  std::vector<TCanvas*> displayView_;

  /// display pad xy size for (x,y) or (r,z) display
  std::vector<int>      viewSize_;     

  /// support histogram for x/y or r/z display
  std::vector<TH2F*>    displayHist_;

  /// ECAL in XY view
  TEllipse frontFaceECALXY_;

  /// ECAL in RZ view
  TBox     frontFaceECALRZ_;

  /// HCAL in XY view
  TEllipse frontFaceHCALXY_;

  /// vector of TGraph used to represent the track in XY or RZ view
  std::vector< std::vector<TGraph*> > graphTrack_;

  /// max rechit energy in ecal
  double                   maxERecHitEcal_;

  /// max rechit energy in hcal
  double                   maxERecHitHcal_;


  //----------------- clustering parameters ---------------------

  /// ecal barrel threshold
  double threshEcalBarrel_;

  /// ecal barrel seed threshold
  double threshSeedEcalBarrel_;

  /// ecal endcap threshold
  double threshEcalEndcap_;

  /// ecal endcap seed threshold
  double threshSeedEcalEndcap_;

  /// ecal number of neighbours
  int    nNeighboursEcal_;


  /// hcal barrel threshold
  double threshHcalBarrel_;

  /// hcal barrel seed threshold
  double threshSeedHcalBarrel_;

  /// hcal endcap threshold
  double threshHcalEndcap_;

  /// hcal endcap seed threshold
  double threshSeedHcalEndcap_;

  /// hcal number of neighbours
  int    nNeighboursHcal_;

};
#endif
