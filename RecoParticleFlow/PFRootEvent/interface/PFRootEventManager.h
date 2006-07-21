#ifndef RecoParticleFlow_PFRootEvent_PFRootEventManager_h
#define RecoParticleFlow_PFRootEvent_PFRootEventManager_h

#include "DataFormats/PFReco/interface/PFRecHit.h"
#include "DataFormats/PFReco/interface/PFCluster.h"

#include <TObject.h>

#include <string>
#include <map>
#include <set>
#include <vector>

class TTree;
class TBranch;
class TFile;
class TCanvas;
class TH2F;
class IO;

class PFRootEventManager {

 public:
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

  /// display rechits
  void DisplayRecHitsEtaPhi();

  /// display rechit
  void DisplayRecHitEtaPhi(reco::PFRecHit& rh,
			   double maxe, double thresh);

  void DisplayClustersEtaPhi();

  void DisplayClusterEtaPhi(const reco::PFCluster& cluster);


  /// display x/y
  void DisplayXY();

  /// display rechit
  void DisplayRecHitXY(reco::PFRecHit& rh,
		       double maxe, double thresh);


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
  
  // rechits
  std::vector<reco::PFRecHit> rechits_;

  // rechits
  std::vector<reco::PFCluster> clusters_;

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

  /// canvas for xy display
  TCanvas                 *displayXY_;

  /// display pad xy size for (x,y) display
  std::vector<int>         viewSizeXY_;     

  /// support histogram for xy display
  TH2F*                    displayHistXY_;

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

  /// ecal barrel threshold
  double threshPS_;

  /// ecal barrel seed threshold
  double threshSeedPS_;

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
