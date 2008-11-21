#ifndef __RCT_CALIBRATOR_H__
#define __RCT_CALIBRATOR_H__
// -*- C++ -*-
//
// Package:    L1RCTCalibrator
// Class:      L1RCTCalibrator
//
/**\class L1RCTCalibrator L1RCTCalibrator.cc src/L1RCTCalibrator/src/L1RCTCalibrator.cc

 Description: Base Class for analyzers which perform calibrations which are somewhat like Lindsey's calibration.

 Implementation:

 The functions below are defined in the base class:
     beginJob() -- calls bookHistograms(), opens root file and checkity-checks itself for consistency
     analyze()  -- gets regions, e/hcal tpgs and any candidates you specify in "CalibrationInputs" (a vector of InputTags)
                   and provides them to saveCalibrationInfo().
     endJob()   -- this calls postProcessing() and takes care of printing out the resulting _cff.py  and .root files.

 When you inherit from this class you must define:
     bookHistograms() -- instanciate all your histograms here... also use putHist(histo = new TH1X()) to keep track of your histograms (or anything else for that matter)
     saveCalibrationInfo() -- anything you need to do to your data during analyze(), i.e. plotting raw energies, 
                              saving data for later and making sanity check plots.
     postProcessing() -- this is anything you need to do in endJob(). i.e. doing fits and determining coefficients
                         If you set "FarmoutMode" to true then this is not run, Just remember to set up a TTree for your data.
 
Debug levels: (feel free to add more in your derived classes!) requires you to use " if (debug()) { code }"
-1 = quiet mode no output from anything
 0 = LogInfo / LogWarning / LowError are turned on
 1 = debug level 0 + LogDebug and any direct cout statements
 9 = create an empty cff and root file
10 = create a debugging cff that contains the element index for all entries and a root file
11 = create pass-thru LUTs

*/
//
// Original Author:  pts/47
//         Created:  Thu Jul 13 21:38:08 CEST 2006
// $Id: L1RCTCalibrator.h,v 1.6 2008/08/19 20:22:46 lgray Exp $
//
//


// system include files
#include <memory>
#include <iostream>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// calo digis
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

// handle
#include "DataFormats/Common/interface/Handle.h"

// default scales
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

// RCT Parameters / Channel Mask
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"


// L1 calo geometry
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"


// Abstract Collection
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "TFile.h"
#include "TTree.h"

//
// class declaration
//
class L1RCTCalibrator : public edm::EDAnalyzer 
{
 public: 
  // nested classes
  class rct_location
    {       
      public:
      int crate, card, region;
      
      bool operator==(const rct_location& r) const { return (crate == r.crate && card == r.card && region == r.region); }
    };

  class region
  {
  public:
    int linear_et, ieta, iphi;
    double eta, phi;
    rct_location loc;

    bool operator==(const region& r) const { return (loc == r.loc); }    
  };

  class tpg
  {
  public:    
    int ieta, iphi;
    double ecalEt, hcalEt, ecalE, hcalE;
    double eta,phi;
    rct_location loc;

    bool operator==(const tpg& r) const { return ((ieta == r.ieta) && (iphi == r.iphi)); }
  };

  // useful typedefs
  typedef edm::View<reco::Candidate> cand_view;
  typedef edm::View<EcalTriggerPrimitiveDigi> ecal_view;
  typedef edm::View<HcalTriggerPrimitiveDigi> hcal_view;
  typedef edm::View<L1CaloRegion> reg_view;
  typedef std::vector<edm::Handle<cand_view> > view_vector;
  typedef cand_view::const_iterator cand_iter;
  typedef ecal_view::const_iterator ecal_iter;
  typedef hcal_view::const_iterator hcal_iter;
  typedef reg_view::const_iterator region_iter;

  explicit L1RCTCalibrator(edm::ParameterSet const&);
  virtual ~L1RCTCalibrator();
  
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&);
  void endJob();

  virtual void saveCalibrationInfo(const view_vector&,const edm::Handle<ecal_view>&, 
				   const edm::Handle<hcal_view>&, const edm::Handle<reg_view>&) = 0;
  virtual void postProcessing() = 0;

  virtual void bookHistograms() = 0;
    
protected:
  // ----------protected member functions---------------
  void deltaR(const double& eta1, double phi1, 
	      const double& eta2, double phi2,double& dr) const; // calculates delta R between two coordinates
  void etaBin(const double&, int&) const; // calculates Trigger Tower number
  void etaValue(const int&, double&) const; // calculates Trigger Tower eta bin center
  void phiBin(double, int&) const; // calculates TT phi bin
  void phiValue(const int&, double&) const; // calculates TT phi bin center
  double uniPhi(const double&) const; // returns phi that is in [0, 2*pi]
  bool isSelfOrNeighbor(const rct_location&, const rct_location&) const;
  rct_location makeRctLocation(const double& eta, const double& phi) const; // makes an rct_location struct from detector coordinates
  rct_location makeRctLocation(const int& ieta, const int& iphi) const; // makes an rct_location struct from TT ieta and iphi

  // returns -1 if item not present, [index] if it is, T must have an == operator
  template<typename T> 
    int find(const T&, const std::vector<T>&) const;

  // returns Et of the TPG
  double ecalEt(const EcalTriggerPrimitiveDigi&) const;
  double hcalEt(const HcalTriggerPrimitiveDigi&) const;
  // these calculate the energy from Et by using the eta bin center
  double ecalE(const EcalTriggerPrimitiveDigi&) const;
  double hcalE(const HcalTriggerPrimitiveDigi&) const;

  // returns tpgs within a specified delta r near the point (eta,phi)
  std::vector<tpg> tpgsNear(const double& eta, const double& phi, const std::vector<tpg>&, const double& dr = .5) const;
  // returns an ordered pair of (Et, deltaR)
  std::pair<double,double> showerSize(const std::vector<tpg>&, const double frac = .95, const double& max_dr = .5, 
				      const bool& ecal = true, const bool& hcal = true) const;
  // returns the sum of tpg Et near the point (eta,phi) within a specified delta R, 
  // can choose to only give ecal or hcal sum through bools
  double sumEt(const double& eta, const double& phi, const std::vector<region>&, const double& dr = .5) const;
  double sumEt(const double& eta, const double& phi, const std::vector<tpg>&, const double& dr = .5, 
	       const bool& ecal = true, const bool& hcal = true, const bool& apply_corrections = false,
	       const double& high_low_crossover = 23) const;
  // returns energy weighted average of Eta
  double avgPhi(const std::vector<tpg>&) const;
  // returns energy weighted average of Phi
  double avgEta(const std::vector<tpg>&) const;

  // saves pointer to Histogram in a vector, making writing out easier later.
  void putHist(TObject* o) { hists_.push_back(o); }

  TTree* Tree() {return theTree_;}

  // use these to access debug level, fit options, and event/run number.
  const std::string& fitOpts() const { return fitOpts_; }
  const int& debug() const { return debug_; }
  const int& eventNumber() const { return event_; }
  const int& runNumber() const { return run_; }
  const bool& farmout() const { return farmout_; }
  const int& totalEvents() const { return total_; }

  // use these functions to access the pointers to various scales
  // it is safer than using the bare pointer.
  const L1CaloEcalScale* eScale() const { return const_cast<const L1CaloEcalScale*>(eScale_); }
  const L1CaloHcalScale* hScale() const { return const_cast<const L1CaloHcalScale*>(hScale_); }
  const L1RCTParameters* rctParams() const { return const_cast<const L1RCTParameters*>(rctParams_); }
  const L1RCTChannelMask* chMask() const { return const_cast<const L1RCTChannelMask*>(chMask_); }
  const L1CaloGeometry* l1Geometry() const { return const_cast<const L1CaloGeometry*>(l1Geometry_); }

  double ecal_[28][3], hcal_[28][3], hcal_high_[28][3], cross_[28][6], he_low_smear_[28], he_high_smear_[28];

 private: 
  // returns the regions that neighbor the input
  std::vector<rct_location> neighbors(const rct_location&) const;
  // prints a CFG or python language configuration file fragment that contains the
  // resultant calibration information
  void printCfFragment(std::ostream&) const;
  // makes sure that etaBin = etaBin(etaValue(etaBin)) and phiBin = phiBin(phiValue(phiBin))
  bool sanityCheck() const;

  // loops through hists_ and calls Write for each one
  void writeHistograms();

  // ----------member data----------------------------
  L1CaloEcalScale *eScale_;
  L1CaloHcalScale *hScale_;
  L1RCTParameters *rctParams_;
  L1RCTChannelMask *chMask_;
  L1CaloGeometry *l1Geometry_;
  std::vector<edm::InputTag> cand_inputs_;
  edm::InputTag ecalTPG_, hcalTPG_, regions_;
  std::string outfile_;
  const int debug_; // -1 for quiet mode, 0 for LogInfo/Warning/Error, 1 for "0 + verbose couts and fits", 
                    // 9 for empty coefficient tables, 10 for output sanity check
  bool python_, farmout_;  
  const double deltaEtaBarrel_, maxEtaBarrel_, deltaPhi_;
  const std::vector<double> endcapEta_;
  const std::string fitOpts_;
  int event_, run_, total_;
  TFile *rootOut_;
  TTree *theTree_;
  //vector of all pointers to histograms
  std::vector<TObject*> hists_;
};
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// templated function definitions
//

template<typename T> 
int L1RCTCalibrator::find(const T& item, const std::vector<T>& v) const
{
  for(unsigned i = 0; i < v.size(); ++i)
    if(item == v[i]) return i;
  return -1;
}

#endif
