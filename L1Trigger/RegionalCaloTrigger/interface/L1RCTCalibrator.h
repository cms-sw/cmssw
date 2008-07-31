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
     beginJob() 
     analyze()
     endJob() -- this already takes care of printing out the resulting _cff.py file.

 When you inherit from this class you must define:
     saveCalibrationInfo() -- anything you need to do during analyze()
     postProcessing() -- this is anything you need to do in end job.
 

*/
//
// Original Author:  pts/47
//         Created:  Thu Jul 13 21:38:08 CEST 2006
// $Id: L1RCTCalibrator.h,v 1.1 2008/07/31 14:17:13 lgray Exp $
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

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

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

// Abstract Collection
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "TH1F.h"
#include "TH2F.h"

//
// class declaration
//
class L1RCTCalibrator : public edm::EDAnalyzer 
{
 public: 

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

  void deltaR(const double& eta1, const double& phi1, 
	      const double& eta2, const double& phi2,double& dr) const;  
  void etaBin(const double&, int&) const; // returns Trigger Tower number
  void etaValue(const int&, double&) const; // return Trigger Tower eta bin center
  void phiBin(const double&, int&) const; // returns TT phi bin
  void phiValue(const int&, double&) const; // return TT phi bin center

  const std::string& fitOpts() const { return fitOpts_; }
  const L1CaloEcalScale* eScale() const { return const_cast<const L1CaloEcalScale*>(eScale_); }
  const L1CaloHcalScale* hScale() const { return const_cast<const L1CaloHcalScale*>(hScale_); }
  const L1RCTParameters* rctParams() const { return const_cast<const L1RCTParameters*>(rctParams_); }
  const L1RCTChannelMask* chMask() const { return const_cast<const L1RCTChannelMask*>(chMask_); }

  double ecal_[28][3], hcal_[28][3], hcal_high_[28][3], cross_[28][6], he_low_smear_[28], he_high_smear_[28];

 private: 
  void printCfFragment(std::ostream&) const;
  bool sanityCheck() const;

  // ----------member data----------------------------
  L1CaloEcalScale *eScale_;
  L1CaloHcalScale *hScale_;
  L1RCTParameters *rctParams_;
  L1RCTChannelMask *chMask_;
  std::vector<edm::InputTag> cand_inputs_;
  edm::InputTag ecalTPG_, hcalTPG_, regions_;
  std::string outfile_;
  int debug_; // -1 for quiet mode, 0 for LogInfo/Warning/Error, 1 for "0 + verbose couts and fits", 
             // 9 for empty coefficient tables, 10 for output sanity check
  bool python_;  
  const double deltaEtaBarrel_, maxEtaBarrel_, deltaPhi_;
  const std::vector<double> endcapEta_;
  const std::string fitOpts_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

#endif
