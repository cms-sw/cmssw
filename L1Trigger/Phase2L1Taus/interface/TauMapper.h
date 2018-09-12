#ifndef L1Trigger_Phase2L1Tau_TauMapper_h
#define L1Trigger_Phase2L1Tau_TauMapper_h

#include "DataFormats/L1Trigger/interface/L1PFTau.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <unordered_map>

#define tracker_eta 3.5
#define tau_size_eta 6*0.08750 //6 towers
#define tau_size_phi 6*0.08726 //6 towers
#define tower_size 0.087

//namespace l1ttau_impl { 
class TauMapper;

typedef std::vector<TauMapper> tauMapperCollection;

class TauMapper{
      public:
  //TauMapper( const edm::ParameterSet& ) ;

	TauMapper();

	~TauMapper();

	l1t::L1PFTau l1PFTau;

	typedef struct
	{
	  float et = 0;
	  float eta = 0;
	  float phi = 0;
	} simple_object_t;
	
	// set parameters
	void minimumHSeedPt();
	void minimumEmSeedPt();
	void deltaR3Prong();
	void deltaRStrip();
	void deltaZ();

	void setSeedChargedHadron(l1t::PFCandidate in){
	  seedCH = in;
	  seedHadronSet = true;
	  buildStripGrid();
	};

	// add PF Charged Hadron, sets as Seed if it is the First Charged Hadron
	bool addPFChargedHadron(l1t::PFCandidate in);

	// add E/G to Strip or to Isolation Cone, sets as Seed if it is the first E/G 
	bool addEG(l1t::PFCandidate in);

	// Use for isolation cone
	void addNeutral();
	
	// determine if 1 prong, 1 prong pi0 or 3 prong
	void process();

	void process_strip();

	void merge_strip(simple_object_t cluster_1, simple_object_t cluster_2, simple_object_t &strip);

	float delta_r_cluster(simple_object_t cluster_1, simple_object_t cluster_2);

	float weighted_avg_phi(simple_object_t cluster_1, simple_object_t cluster_2);

	float weighted_avg_eta(simple_object_t cluster_1, simple_object_t cluster_2);

	//fixed cone container
	bool contains(l1t::PFCandidate in);

	float round_to_tower(float input){
	  return round(input/0.087)*0.087;
	  
	};

	void buildStripGrid();

	simple_object_t egGrid[5][5];

	bool isolationConeContains( l1t::PFCandidate in );
	bool seedCHConeContains( l1t::PFCandidate in);

	bool isSeedHadronSet(){
	  return seedHadronSet;
	}

	void ClearSeedHadron(){
	  seedHadronSet = false;;
	}
 private:

	bool seedHadronSet;
	l1t::PFCandidate seedCH;
	l1t::PFCandidate prong2;
	l1t::PFCandidate prong3;

	float sumChargedIso;
	float strip_pt;
	float strip_eta;
};

#endif
