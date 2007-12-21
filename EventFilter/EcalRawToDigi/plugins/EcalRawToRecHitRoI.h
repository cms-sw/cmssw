#ifndef EventFilter_EcalRawToRecHitRoI_H
#define EventFilter_EcalRawToRecHitRoI_H

#include <FWCore/Framework/interface/EDProducer.h>
                                                                                             
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometry.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
                                                                                             
#include <iostream>
#include <string>
#include <vector>

#include "DataFormats/EcalRecHit/interface/LazyGetter.h"
#include "DataFormats/EcalRecHit/interface/RefGetter.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

class EcalRawToRecHitRoI : public edm::EDProducer {

  typedef edm::LazyGetter<EcalRecHit> EcalRecHitLazyGetter;
  typedef edm::RefGetter<EcalRecHit> EcalRecHitRefGetter;

public:

	EcalRawToRecHitRoI(const edm::ParameterSet& pset);
	virtual ~EcalRawToRecHitRoI();
	void produce(edm::Event & e, const edm::EventSetup& c);
	void beginJob(const edm::EventSetup& c);
	void endJob(void);

 private:

	void Egamma(edm::Event& e, const edm::EventSetup& es, std::vector<int>& FEDs);
	void Muon(edm::Event& e, const edm::EventSetup& es, std::vector<int>& FEDs);
	void Jet(edm::Event& e, const edm::EventSetup& es, std::vector<int>& FEDs);
	void Cand(edm::Event& e, const edm::EventSetup& es, std::vector<int>& FEDs);


	/// input tag for the lazy getter
        edm::InputTag sourceTag_;
   
        /// tools
	const EcalElectronicsMapping* TheMapping;
     
        /// generic class to drive the job
        class CalUnpackJobPSet {
	public:
	  CalUnpackJobPSet(){}
	  CalUnpackJobPSet(edm::ParameterSet &cfg){
	    Source = cfg.getParameter<edm::InputTag>("Source");
	    Ptmin = cfg.getParameter<double>("Ptmin");
	    regionEtaMargin = cfg.getParameter<double>("regionEtaMargin");
	    regionPhiMargin = cfg.getParameter<double>("regionPhiMargin");
	  }
	  edm::InputTag Source;
	  double Ptmin;
	  double regionEtaMargin;
	  double regionPhiMargin;
	};

	///Egamma part flag
	bool EGamma_;
	/// class to drive the job on L1Em
        class EmJobPSet : public CalUnpackJobPSet {
        public:
          EmJobPSet(edm::ParameterSet &cfg) : CalUnpackJobPSet(cfg){}
            ~EmJobPSet(){}
        };
	/// process one collection of L1Em
        void Egamma_OneL1EmCollection(const edm::Handle< l1extra::L1EmParticleCollection > emColl,
                                      const EmJobPSet & ejpset,
                                      const edm::ESHandle< L1CaloGeometry > & l1CaloGeom,
                                      std::vector<int> & FEDs);
	/// what drive the job on L1Em collection
	std::vector< EmJobPSet > EmSource_;
	
	///Muon part flag
	bool Muon_ ;
	/// class to drive the job on L1Muon
	class MuJobPSet : public CalUnpackJobPSet {
	public:
	  MuJobPSet(){}
	  MuJobPSet(edm::ParameterSet &cfg) : CalUnpackJobPSet(cfg), epsilon(0.01) {}
	    ~MuJobPSet(){}
	    double epsilon;
	};
	/// what drives the job from ONE L1Muon collection
	MuJobPSet MuonSource_;

	///jet part flag
	bool Jet_ ;
	/// class to drive the job on L1Jet
	class JetJobPSet :public CalUnpackJobPSet {
        public:
          JetJobPSet(edm::ParameterSet &cfg) : CalUnpackJobPSet(cfg), epsilon(0.01) {}
	    ~JetJobPSet(){}
	    double epsilon;
        };
	/// process on collection of L1Jets
        void Jet_OneL1JetCollection(const edm::Handle< l1extra::L1JetParticleCollection > jetColl,
                                    const JetJobPSet & jjpset,
                                    std::vector<int> & feds);
	/// what drive the job on L1Jet collection
	std::vector< JetJobPSet > JetSource_;

	///Candidate-inheriting objects part flag
	bool Candidate_;
	/// class to drive the job on Candidate-inheriting object
	class CandJobPSet : public CalUnpackJobPSet {
	public:
	  CandJobPSet(edm::ParameterSet &cfg) : CalUnpackJobPSet(cfg), epsilon(0.01) {}
	    ~CandJobPSet(){}
	    double epsilon;
	};
	/// process one collection of Candidate-inheriting objects
	void OneCandCollection(const edm::Handle< edm::View< reco::Candidate > >Coll,
			       const CandJobPSet & cjpset,
			       std::vector<int> & feds);
	/// what drives the job from candidate
	std::vector< CandJobPSet > CandSource_;

	///if all need to be done
	bool All_;

	///output
	std::string OutputLabel_;

	/// actually fill the vector with FED numbers
	void ListOfFEDS(double etaLow, double etaHigh, double phiLow,
			double phiHigh, double etamargin, double phimargin,
			std::vector<int>& FEDs);
	
	/// remove duplicates
	void unique(std::vector<int>& FEDs){
	  std::vector<int>::iterator n_end = std::unique(FEDs.begin(),FEDs.end());
	  FEDs.erase(n_end,FEDs.end());}
	std::string dumpFEDs(const std::vector<int>& FEDs);
	/*        void notDone(const std::vector<int> & DONE,std::vector<int> & FEDs){
		  std::vector<int>::iterator it= FEDs.begin();
		  while(it!=FEDs.end()){
		  if (std::find(DONE.begin(),DONE.end(), *it)==DONE.end()) it++;
		  else it=FEDs.erase(it);}
		  }
	*/
};

#endif


