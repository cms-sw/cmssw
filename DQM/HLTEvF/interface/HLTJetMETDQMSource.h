 #ifndef HLTJetMETDQMSource_H
#define HLTJetMETDQMSource_H
// -*- C++ -*-
//
// Package:    HLTJetMETDQMSource
// Class:      HLTJetMETDQMSource
// Code for HLT JetMET DQ monitoring. Based on FourVectorOnline code.


 


// system include files
#include <memory>
#include <unistd.h>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <vector>

class HLTJetMETDQMSource : public edm::EDAnalyzer {
   public:
      explicit HLTJetMETDQMSource(const edm::ParameterSet&);
      ~HLTJetMETDQMSource();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // BeginRun
      void beginRun(const edm::Run& run, const edm::EventSetup& c);
     
      void histobooking( const edm::EventSetup& c);

      // EndRun
      void endRun(const edm::Run& run, const edm::EventSetup& c);
      

      void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				const edm::EventSetup& c) ;

      /// DQM Client Diagnostic
	void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				const edm::EventSetup& c);


      // ----------member data --------------------------- 
      int nev_;
      DQMStore * dbe_;

      MonitorElement* total_;

      bool plotAll_;
      bool plotEff_;
      bool plotwrtMu_;
      bool resetMe_;
      int currentRun_;
      int nhltcfg;
      bool hltconfigchanged;
      unsigned int nBins_; 
      double ptMin_ ;
      double ptMax_ ;
      
      
      double muonEtaMax_;
      double muonEtMin_;
      double muonDRMatch_;
      
      double jetEtaMax_;
      double jetEtMin_;
      double jetDRMatch_;
      
      double metMin_;
      double htMin_;
      double sumEtMin_;

      std::string  custompathnamemu_;
      std::vector<std::pair<std::string, std::string> > custompathnamepairs_;
      std::vector<int>  prescUsed_;


      std::string dirname_;
      std::string processname_;
      bool verbose_;
      bool monitorDaemon_;
      int theHLTOutputType;
      
      edm::InputTag triggerSummaryLabel_;
      edm::InputTag triggerResultsLabel_;
      HLTConfigProvider hltConfig_;
      // data across paths
      MonitorElement* scalersSelect;
      // helper class to store the data path

      class PathInfo {
	PathInfo():
	  pathIndex_(-1), prescaleUsed_(-1),denomPathName_("unset"), pathName_("unset"), l1pathName_("unset"), filterName_("unset"), DenomfilterName_("unset"), processName_("unset"), objectType_(-1)
	  {};
      public:
	void setHistos( 
		       MonitorElement* const N, 
                       MonitorElement* const Et, 
                       MonitorElement* const EtaPhi,  
                       MonitorElement* const Eta,  
                       MonitorElement* const Phi,  
                       MonitorElement* const NL1, 
                       MonitorElement* const l1Et, 
		       MonitorElement* const l1EtaPhi,
		       MonitorElement* const l1Eta,
		       MonitorElement* const l1Phi)

          {
          N_ = N;
	  Et_ = Et;
	  EtaPhi_ = EtaPhi;
          Eta_ = Eta;
          Phi_ = Phi;
          NL1_ = NL1;
	  l1Et_ = l1Et;
	  l1EtaPhi_ = l1EtaPhi;
	  l1Eta_ = l1Eta;
	  l1Phi_ = l1Phi;
          
	}
	void setHistosEff(
			  MonitorElement* const NEff, 
			  MonitorElement* const EtEff, 
			  MonitorElement* const EtaEff,
			  MonitorElement* const PhiEff,
			  MonitorElement* const NNum, 
			  MonitorElement* const EtNum, 
			  MonitorElement* const EtaNum,
			  MonitorElement* const PhiNum,
			  MonitorElement* const NDenom, 
			  MonitorElement* const EtDenom, 
			  MonitorElement* const EtaDenom,
			  MonitorElement* const PhiDenom)

          {
	    NEff_ = NEff;
	    EtEff_ = EtEff;
	    EtaEff_ = EtaEff;
	    PhiEff_ = PhiEff;
	    NNum_ = NNum;
	    EtNum_ = EtNum;
	    EtaNum_ = EtaNum;
	    PhiNum_ = PhiNum;
	    NDenom_ = NDenom;
	    EtDenom_ = EtDenom;
	    EtaDenom_ = EtaDenom;
	    PhiDenom_ = PhiDenom;
	  }
	void setHistoswrtMu(
			    MonitorElement* const NwrtMu, 
			    MonitorElement* const EtwrtMu, 
			    MonitorElement* const EtaPhiwrtMu,
			    MonitorElement* const PhiwrtMu)
                       

          {
	    NwrtMu_ = NwrtMu;
	  EtwrtMu_ = EtwrtMu;
	  EtaPhiwrtMu_ = EtaPhiwrtMu;
	  PhiwrtMu_ = PhiwrtMu;
	  }
	MonitorElement * getNHisto() {
	  return N_;
	}
	MonitorElement * getEtHisto() {
	  return Et_;
	}
	MonitorElement * getEtaPhiHisto() {
	  return EtaPhi_;
	}
	MonitorElement * getEtaHisto() {
	  return Eta_;
	}
	MonitorElement * getPhiHisto() {
	  return Phi_;
	}
	MonitorElement * getNL1Histo() {
	  return NL1_;
	}
	MonitorElement * getL1EtHisto() {
	  return l1Et_;
	}
	MonitorElement * getL1EtaHisto() {
	  return l1Eta_;
	}
	MonitorElement * getL1EtaPhiHisto() {
	  return l1EtaPhi_;
	}
	MonitorElement * getL1PhiHisto() {
	  return l1Phi_;
	}
	MonitorElement * getNwrtMuHisto() {
	  return NwrtMu_;
	}
	MonitorElement * getEtwrtMuHisto() {
	  return EtwrtMu_;
	}
	MonitorElement * getEtaPhiwrtMuHisto() {
	  return EtaPhiwrtMu_;
	}
	MonitorElement * getPhiwrtMuHisto() {
	  return PhiwrtMu_;
	}
	MonitorElement * getNEffHisto() {
	  return NEff_;
	}
	MonitorElement * getEtEffHisto() {
	  return EtEff_;
	}
	MonitorElement * getEtaEffHisto() {
	  return EtaEff_;
	}
	MonitorElement * getPhiEffHisto() {
	  return PhiEff_;
	}
	MonitorElement * getNNumHisto() {
	  return NNum_;
	}
	MonitorElement * getEtNumHisto() {
	  return EtNum_;
	}
	MonitorElement * getEtaNumHisto() {
	  return EtaNum_;
	}
	MonitorElement * getPhiNumHisto() {
	  return PhiNum_;
	}
	MonitorElement * getNDenomHisto() {
	  return NDenom_;
	}
	MonitorElement * getEtDenomHisto() {
	  return EtDenom_;
	}
	MonitorElement * getEtaDenomHisto() {
	  return EtaDenom_;
	}
	MonitorElement * getPhiDenomHisto() {
	  return PhiDenom_;
	}
	const std::string getLabel(void ) const {
	  return filterName_;
	}
	const std::string getDenomLabel(void ) const {
	  return DenomfilterName_;
	}
	
	void setLabel(std::string labelName){
	  filterName_ = labelName;
          return;
	}
	void setDenomLabel(std::string labelName){
	  DenomfilterName_ = labelName;
          return;
	}
	const std::string getPath(void ) const {
	  return pathName_;
	}
	const std::string getl1Path(void ) const {
	  return l1pathName_;
	}
	const std::string getDenomPath(void ) const {
	  return denomPathName_;
	}
	const int getprescaleUsed(void) const {
	  return prescaleUsed_;
	}
	const std::string getProcess(void ) const {
	  return processName_;
	}
	const int getObjectType(void ) const {
	  return objectType_;
	}

        const edm::InputTag getTag(void) const{
	  edm::InputTag tagName(filterName_,"",processName_);
          return tagName;
	}
	const edm::InputTag getDenomTag(void) const{
	  edm::InputTag tagName(DenomfilterName_,"",processName_);
          return tagName;
	}
	~PathInfo() {};
	PathInfo(int prescaleUsed, std::string denomPathName, std::string pathName, std::string l1pathName, std::string filterName, std::string DenomfilterName, std::string processName, size_t type, float ptmin, 
		 float ptmax):
	  prescaleUsed_(prescaleUsed),denomPathName_(denomPathName), pathName_(pathName), l1pathName_(l1pathName), filterName_(filterName), DenomfilterName_(DenomfilterName), processName_(processName), objectType_(type),
          N_(0), Et_(0), EtaPhi_(0),Eta_(0),
	  NL1_(0), l1Et_(0), l1EtaPhi_(0),l1Eta_(0),l1Phi_(0),
          NwrtMu_(0), EtwrtMu_(0), EtaPhiwrtMu_(0),PhiwrtMu_(0),
          NEff_(0), EtEff_(0), EtaEff_(0),PhiEff_(0),
	  NNum_(0), EtNum_(0), EtaNum_(0),PhiNum_(0),
	  NDenom_(0), EtDenom_(0), EtaDenom_(0),PhiDenom_(0),
	  ptmin_(ptmin), ptmax_(ptmax)
	  {
	  };
	  PathInfo(int prescaleUsed, std::string denomPathName, std::string pathName, std::string l1pathName, std::string filterName, std::string DenomfilterName, std::string processName, size_t type,
		   MonitorElement *N,
		   MonitorElement *Et,
		   MonitorElement *EtaPhi,
		   MonitorElement *Eta,
		   MonitorElement *Phi,
		   MonitorElement *NL1,
		   MonitorElement *l1Et,
		   MonitorElement *l1EtaPhi,
		   MonitorElement *l1Eta,
		   MonitorElement *l1Phi,
		   MonitorElement *NwrtMu,
		   MonitorElement *EtwrtMu,
		   MonitorElement *EtaPhiwrtMu,
		   MonitorElement *PhiwrtMu,
		   MonitorElement *NEff,
		   MonitorElement *EtEff,
		   MonitorElement *EtaEff,
		   MonitorElement *PhiEff,
		   MonitorElement *NNum,
		   MonitorElement *EtNum,
		   MonitorElement *EtaNum,
		   MonitorElement *PhiNum,
		   MonitorElement *NDenom,
		   MonitorElement *EtDenom,
		   MonitorElement *EtaDenom,
		   MonitorElement *PhiDenom,
		   float ptmin, float ptmax
		   ):
	     prescaleUsed_(prescaleUsed), denomPathName_(denomPathName), pathName_(pathName), l1pathName_(l1pathName), filterName_(filterName), DenomfilterName_(DenomfilterName), processName_(processName), objectType_(type),
            N_(N), Et_(Et), EtaPhi_(EtaPhi),Eta_(Eta), Phi_(Phi),
	    NL1_(NL1), l1Et_(l1Et), l1EtaPhi_(l1EtaPhi),l1Eta_(l1Eta),l1Phi_(l1Phi),
            NwrtMu_(NwrtMu), EtwrtMu_(EtwrtMu), EtaPhiwrtMu_(EtaPhiwrtMu),PhiwrtMu_(PhiwrtMu),
            NEff_(NEff), EtEff_(EtEff), EtaEff_(EtaEff), PhiEff_(PhiEff),
	    NNum_(NNum), EtNum_(EtNum), EtaNum_(EtaNum), PhiNum_(PhiNum),
	    NDenom_(NDenom), EtDenom_(EtDenom), EtaDenom_(EtaDenom),PhiDenom_(PhiDenom),
	    ptmin_(ptmin), ptmax_(ptmax)
	    {};
	    bool operator==(const std::string v) 
	    {
	      return v==filterName_;
	    }
      private:
	  int pathIndex_;
	  int prescaleUsed_;
	  std::string denomPathName_;
	  std::string pathName_;
	  std::string l1pathName_;
	  std::string filterName_;
	  std::string DenomfilterName_;
	  std::string processName_;
	  int objectType_;

	  // we don't own this data
          MonitorElement *N_, *Et_, *EtaPhi_, *Eta_, *Phi_;
	  MonitorElement *NL1_, *l1Et_, *l1EtaPhi_ ,*l1Eta_ , *l1Phi_;
	  MonitorElement *NwrtMu_, *EtwrtMu_, *EtaPhiwrtMu_, *PhiwrtMu_;
	  MonitorElement *NEff_, *EtEff_, *EtaEff_, *PhiEff_;
	  MonitorElement *NNum_, *EtNum_, *EtaNum_, *PhiNum_;
	  MonitorElement *NDenom_, *EtDenom_, *EtaDenom_, *PhiDenom_;

	  float ptmin_, ptmax_;

	  const int index() { 
	    return pathIndex_;
	  }
	  const int type() { 
	    return objectType_;
	  }
      public:
	  float getPtMin() const { return ptmin_; }
	  float getPtMax() const { return ptmax_; }
      };

      // simple collection - just 
      class PathInfoCollection: public std::vector<PathInfo> {
      public:
	PathInfoCollection(): std::vector<PathInfo>() 
	  {};
	  std::vector<PathInfo>::iterator find(std::string pathName) {
	    return std::find(begin(), end(), pathName);
	  }
      };
      PathInfoCollection hltPathsAll_;
      PathInfoCollection hltPathsEff_;
      PathInfoCollection hltPathswrtMu_;
      MonitorElement* rate_All;
      MonitorElement* rate_All_L1;
      MonitorElement* rate_Eff;
      MonitorElement* rate_Denom;
      MonitorElement* rate_Num;
      MonitorElement* rate_wrtMu;
	
	
};
#endif
