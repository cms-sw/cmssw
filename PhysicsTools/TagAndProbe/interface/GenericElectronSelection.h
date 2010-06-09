#ifndef PhysicsTools_TagAndProbe_GenericElectronSelection_h
#define PhysicsTools_TagAndProbe_GenericElectronSelection_h

// system include files
#include <memory>
#include "TH1.h"
#include "TString.h"
#include "TNamed.h"
#include "TFile.h"
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "PhysicsTools/TagAndProbe/interface/GenericElectronSelection.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"


// forward declarations
template< class gsfelectron >
class GenericElectronSelection : public edm::EDProducer 
{
 public:
  explicit GenericElectronSelection(const edm::ParameterSet&);
  ~GenericElectronSelection();

 private:
  virtual void beginRun(edm::Run& iRun, edm::EventSetup const& iSetup);
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  bool cutDecision ( int classification, double deta, 
		     double dphi, double sietaeta, double tkiso,
		     double ecaliso, double hcaliso);

  bool CheckTriggerMatch( edm::Handle<trigger::TriggerEvent> triggerObj,
			  double eta, double phi);

  void FillHist(const TString& histName, std::map<TString, TH1*> 
		HistNames, const double& x);
  // ----------member data ---------------------------

  std::string histogramFile_;
  TFile* m_file_;
  std::map<TString, TH1*> m_HistNames1D;

  bool _requireTrigMatch;
  bool _verbose;      
  bool _requireTkIso;
  bool _requireEcalIso;
  bool _requireHcalIso;


  std::string _inputProducer;
  double _etMin;
  double _etMax;
  double deltaEtaCutBarrel_;
  double deltaEtaCutEndcaps_;
  double deltaPhiCutBarrel_;
  double deltaPhiCutEndcaps_;
  double sigmaEtaEtaCutBarrel_;
  double sigmaEtaEtaCutEndcaps_;
  double tkIsoCutBarrel_;
  double tkIsoCutEndcaps_;
  double ecalIsoCutBarrel_;
  double ecalIsoCutEndcaps_;
  double hcalIsoCutBarrel_;
  double hcalIsoCutEndcaps_;
  int _charge;

  edm::InputTag  triggerSummaryLabel_;
  edm::InputTag  hltTag_;
  bool changed_;
  HLTConfigProvider hltConfig_;
};
#include "PhysicsTools/TagAndProbe/src/GenericElectronSelection.icc"
#endif
