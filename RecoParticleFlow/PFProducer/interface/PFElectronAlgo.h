#ifndef PFProducer_PFElectronAlgo_H
#define PFProducer_PFElectronAlgo_H

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "TMVA/Reader.h"
#include "boost/shared_ptr.hpp"
#include <iostream>


class PFSCEnergyCalibration;
class PFEnergyCalibration;

namespace reco { 
class PFCandidate;
}

class PFElectronAlgo {
 public:
  
  //constructor
  PFElectronAlgo(const double mvaEleCut,
		 std::string  mvaWeightFileEleID,
		 const boost::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
		 const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
		 bool applyCrackCorrections,
		 bool usePFSCEleCalib,
		 bool useEGElectrons,
		 bool useEGammaSupercluster,
		 double sumEtEcalIsoForEgammaSC_barrel,
		 double sumEtEcalIsoForEgammaSC_endcap,
		 double coneEcalIsoForEgammaSC,
		 double sumPtTrackIsoForEgammaSC_barrel,
		 double sumPtTrackIsoForEgammaSC_endcap,
		 unsigned int nTrackIsoForEgammaSC,
		 double coneTrackIsoForEgammaSC);

  //destructor
  ~PFElectronAlgo(){delete tmvaReader_;};
  
  //check candidate validity
  bool isElectronValidCandidate(const reco::PFBlockRef&  blockRef,
				std::vector<bool>&  active,
				const reco::Vertex & primaryVertex)
  {
    isvalid_=false;
    RunPFElectron(blockRef,active, primaryVertex);
    return isvalid_;};
  
  //get electron PFCandidate
  const std::vector<reco::PFCandidate>& getElectronCandidates() {return elCandidate_;};

  //get all electron PFCandidate
  const std::vector<reco::PFCandidate>& getAllElectronCandidates() {return allElCandidate_;};

  //get the electron PFCandidateExtra (for all candidates)
  const std::vector< reco::PFCandidateElectronExtra>& getElectronExtra() {return electronExtra_;};

  // retrieve the list of pre-defined e/g electrons
  void setEGElectronCollection(const reco::GsfElectronCollection & egelectrons);

 private: 
  typedef  std::map< unsigned int, std::vector<unsigned int> >  AssMap;

  void RunPFElectron(const reco::PFBlockRef&  blockRef,
		     std::vector<bool>& active,
			 const reco::Vertex & primaryVertex);

  unsigned int FindClosestElement(const unsigned int iele,
			  std::multimap<double, unsigned int>& Elems, 
			  float& chi2cut,
			  std::vector<bool>& active,
			  const reco::PFBlockRef&  blockRef);
  
  bool SetLinks(const reco::PFBlockRef&  blockRef,
		AssMap& associatedToGsf_,
		AssMap& associatedToBrems_,
		AssMap& associatedToEcal_,
		std::vector<bool>& active,
		const reco::Vertex & primaryVertex);
  
  void SetIDOutputs(const reco::PFBlockRef&  blockRef,
		    AssMap& associatedToGsf_,
		    AssMap& associatedToBrems_,
		    AssMap& associatedToEcal_,
			const reco::Vertex & primaryVertex);
  
  void SetCandidates(const reco::PFBlockRef&  blockRef,
		     AssMap& associatedToGsf_,
		     AssMap& associatedToBrems_,
		     AssMap& associatedToEcal_);
  
  void SetActive(const reco::PFBlockRef&  blockRef, 
		 AssMap& associatedToGsf_, 
		 AssMap& associatedToBrems_, 
		 AssMap& associatedToEcal_,
		 std::vector<bool>& active);
  


  bool isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
		      const reco::PFBlockElementGsfTrack& GsfEl);

  std::vector<reco::PFCandidate> elCandidate_;
  std::vector<reco::PFCandidate> allElCandidate_;
  std::map<unsigned int,std::vector<reco::PFCandidate> > electronConstituents_;
  std::vector<double> BDToutput_;
  std::vector<reco::PFCandidateElectronExtra > electronExtra_;
  std::vector<bool> lockExtraKf_;
  std::vector<bool> GsfTrackSingleEcal_;
  std::vector< std::pair <unsigned int, unsigned int> > fifthStepKfTrack_;
  std::vector< std::pair <unsigned int, unsigned int> > convGsfTrack_;

  
  TMVA::Reader    *tmvaReader_;
  double mvaEleCut_;
  boost::shared_ptr<PFSCEnergyCalibration> thePFSCEnergyCalibration_; 
  boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_; 
  bool applyCrackCorrections_;
  bool usePFSCEleCalib_;
  bool useEGElectrons_;
  bool useEGammaSupercluster_;
  double sumEtEcalIsoForEgammaSC_barrel_;
  double sumEtEcalIsoForEgammaSC_endcap_;
  double coneEcalIsoForEgammaSC_;
  double sumPtTrackIsoForEgammaSC_barrel_;
  double sumPtTrackIsoForEgammaSC_endcap_;
  unsigned int nTrackIsoForEgammaSC_;
  double coneTrackIsoForEgammaSC_;

  const char  *mvaWeightFile_;

  // New BDT observables
  // Normalization 
  float lnPt_gsf,Eta_gsf;
  
  // Pure Tracking observ.
  float dPtOverPt_gsf,chi2_gsf,DPtOverPt_gsf,
    chi2_kf,DPtOverPt_kf;
  //  int nhit_gsf,nhit_kf;
  float nhit_gsf,nhit_kf;
  
  // Tracker-Ecal observ. 
  float EtotPinMode,EGsfPoutMode,EtotBremPinPoutMode;
  float DEtaGsfEcalClust;
  float SigmaEtaEta; 
  //int lateBrem,firstBrem,earlyBrem;
  float lateBrem,firstBrem,earlyBrem;
  float HOverHE,HOverPin;

  bool isvalid_;

  const std::vector<reco::GsfElectron> * theGsfElectrons_;
};


#endif
