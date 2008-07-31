#ifndef __RCT_CALIBRATOR_H__
#define __RCT_CALIBRATOR_H__
// -*- C++ -*-
//
// Package:    L1RCTCalibrator
// Class:      L1RCTCalibrator
//
/**\class L1RCTCalibrator L1RCTCalibrator.cc src/L1RCTCalibrator/src/L1RCTCalibrator.cc

 Description: An analyzer that determines calibration information for the RCT based on generator 
              or RECO data.

 Implementation:
     When calibrating using generator information, the raw RCT response is calibrated to generator level information.
     - Photons are needed to calibrate ECAL
     - Charged Pions are used to calibrate HCAL and cross terms.
     
     When calibrating using RECO data, the RCT response is calibrated to RECO photons and RECO single prong jets.
     - Reco photons are used for ECAL
     - Reco pf jets used for HCAL and cross terms.
     
     How it works:
     beginJob() -- nothing special here, just instanciates histograms and sanity checks
     analyze()  -- collects all need data from the event.
     endJob()   -- Writes calibration information to a .cfi file. Processes data collected in analyze()

     If you want to add a calibration type you'll probably need to edit the following functions:
     - RCTCalibrator()
     - analyze()
     - endJob()
     - saveCalibrationInfo()

     You'll also need to write functions to get the information you need out of the event, and functions to do your processing.
     examples: recoCalibration(const edm::Event&), recoCalibration() ...

     TODO: Change to strongly typed inheritance... This makes more sense... Proof of concept works fine though.

*/
//
// Original Author:  pts/47
//         Created:  Thu Jul 13 21:38:08 CEST 2006
// $Id: L1RCTCalibrator.h,v 1.0 2008/05/02 16:53:01 lgray Exp $
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


// forward declarations
namespace reco
{
  class Photon;
  class PFJet;
  class Jet;
  class GenParticle;
}


//
// class declaration
//
class L1RCTCalibrator : public edm::EDAnalyzer 
{
  // condensed data structs
  struct generator
  {
    int particle_type;
    double et, phi, eta;
  };
  
  struct photon
  {
    double et, phi, eta, hovere;
  };
  
  struct oneprong_jet
  {
    double et, phi, eta, hovere;
  };
  
  struct region
  {
    int linear_et, rank, ieta, iphi, crate, card, region;  
  };
  
  struct tpg
  {
    int ieta, iphi, crate, card, region;
    double ecal, hcal;
  };
  
  struct event_data
  {
    unsigned event;
    unsigned run;
    std::vector<generator> gen_particles;
    std::vector<photon> photons;
    std::vector<oneprong_jet> jets;
    std::vector<region> regions;
    std::vector<tpg> tpgs;
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

public:
  enum calib_types { GEN, RECO };

  explicit L1RCTCalibrator(edm::ParameterSet const&);
  ~L1RCTCalibrator();
  
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&);
  void endJob();

private:
  // ----------private member functions---------------

  // specialized member functions -- feel free to make additions
  void recoCalibration(const edm::Event&); //for analyze()
  void recoCalibration(); // for endJob()
  void genCalibration(const edm::Event&); //for analyze()
  void genCalibration(); // for endJob()

  void savePhotonInfo(const reco::Photon*, const edm::Handle<ecal_view>&, const edm::Handle<hcal_view>&, 
		      const edm::Handle<reg_view>&, std::vector<photon>*, std::vector<region>*,
		      std::vector<tpg>*); 
  void savePFJetInfo(const reco::PFJet*, const edm::Handle<ecal_view>&, const edm::Handle<hcal_view>&,
		     const edm::Handle<reg_view>&, std::vector<oneprong_jet>*, std::vector<region>*,
		     std::vector<tpg>*);
  void saveJetInfo(const reco::Jet*, const edm::Handle<ecal_view>&, const edm::Handle<hcal_view>&,
		   const edm::Handle<reg_view>&, std::vector<oneprong_jet>*, std::vector<region>*,
		   std::vector<tpg>*);
  void saveGenInfo(const reco::GenParticle*, const edm::Handle<ecal_view>&, const edm::Handle<hcal_view>&,
		   const edm::Handle<reg_view>&, std::vector<generator>*, std::vector<region>*,
		   std::vector<tpg>*);
  // end specialized member functions
  
  // saveCalibrationInfo is *nearly* generic,
  // it only needs to be modified if a new 
  // algorithm that saves data is made.
  
  // Feed this function a list of data you want to calibrate to,
  // The function gets tpg information and calibrates the tpg info to whatever we're calibrating to.
  // Works for any kind of Jet (even better if they're PFJets), reco::Photon's, reco::*Electron,
  // and gen particles. Add specializations if you want to.
  void saveCalibrationInfo(const view_vector&, const edm::Event&);

  // generic member functions you don't need to edit these.
  void printCfFragment(std::ostream&) const;
  void deltaR(const double& eta1, const double& phi1, 
	      const double& eta2, const double& phi2,double& dr) const;  
  void etaBin(const double&, int&) const; // returns Trigger Tower number
  void etaValue(const int&, double&) const; // return Trigger Tower eta bin center
  void phiBin(const double&, int&) const; // returns TT phi bin
  void phiValue(const int&, double&) const; // return TT phi bin center
  bool sanityCheck() const;

  // ----------member data----------------------------
  L1CaloEcalScale *eScale;
  L1CaloHcalScale *hScale;
  L1RCTParameters *rctParams;
  L1RCTChannelMask *chMask;  
  edm::InputTag gen, rphoton, rjet, ecalTPG, hcalTPG, regions;
  std::string outfile, calib_mode;
  int debug; // -1 for quiet mode, 0 for LogInfo/Warning/Error, 1 for "0 + verbose couts and fits", 
             // 9 for empty coefficient tables, 10 for output sanity check
  bool python;
  calib_types calibration;
  double ecal[28][3], hcal[28][3], hcal_high[28][3], cross[28][6], he_low_smear[28], he_high_smear[28];
  std::map<std::string,calib_types> allowed_calibs;
  const double deltaEtaBarrel, maxEtaBarrel, deltaPhi;
  const std::vector<double> endcapEta;
  const std::string fitOpts;

  // vector of all event data
  std::vector<event_data> data_;

  // histograms

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

#endif
