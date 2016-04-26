#ifndef __EmulatorClasses_
#define __EmulatorClasses_

#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"
#include "L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h"

using namespace L1TMuon;

typedef std::vector<std::vector<PhiMemoryImage>> ImageCollector;
typedef std::vector<std::vector<int>> Code;
typedef std::vector<std::vector<std::vector<int>>> BXHold;

//


class ConvertedHit{

	public:

                ConvertedHit() { SetNull();}
		
		void SetValues(int phi,int theta,int ph_hit,int phzvl,int station,int sub,int id,int quality,int pattern,int wire,int strip,int BX){
		
			_ph = phi;_th = theta;_phit = ph_hit;_phzvl = phzvl;_sta = station;_sub = sub;_id = id;_qual = quality;_patt = pattern;
			_wire = wire;_strip = strip;_zhit = -999;_bx = BX;
		
		};
		
		void SetNull(){
		
			_ph = -999;_th = -999;_phit = -999;_phzvl = -999;_sta = -999;_sub = -999;_id = -999;_qual = -999;_patt = 0;
			_wire = -999;_strip = -999;_zhit = -999;
		};
		
		
		void SetId(int id){

			_id = id;
		};
		
		
		void SetStrip(int strip){

			_strip = strip;
		};
		
		void SetZhit(int zhit){_zhit = zhit;};
		
		void SetTheta(int theta){_th = theta;};
		
		void SetTP(TriggerPrimitive tp){_tp = tp;};
		
		void SetZoneContribution(std::vector<int> zonecontribution){_zonecont = zonecontribution;};
		
		
		int Phi(){return _ph;};
		int Theta(){return _th;};
		int Ph_hit(){return _phit;};
		int Phzvl(){return _phzvl;};
		int Station(){return _sta;};
		int Sub(){return _sub;};
		int Id(){return _id;};
		int Quality(){return _qual;};
		int Pattern(){return _patt;};
		int Wire(){return _wire;};
		int Strip(){return _strip;};
		int Zhit(){return _zhit;};
		int BX(){return _bx;};
		TriggerPrimitive TP(){return _tp;};
		std::vector<int> ZoneContribution(){return _zonecont;};
		
		
	private:
	
		int _ph,_th,_phit,_phzvl,_sta,_sub,_id,_qual,_patt,_wire,_strip,_zhit,_bx;
		TriggerPrimitive _tp;
		std::vector<int> _zonecont;

};

struct ZonesOutput{

	std::vector<PhiMemoryImage> zone;
	std::vector<ConvertedHit> convertedhits;
	
};

struct QualityOutput{

	Code rank, layer,straightness;

	
};


struct PatternOutput{

	QualityOutput detected;
	std::vector<ConvertedHit> hits;

};


struct Wier{

	int rank, strip;
};

class Winner{

	public:
	
	
		//Default Constructor: rank & strip = 0///
		Winner(){_rank = 0;_strip = 0;};
		
		int Rank(){return _rank;}
		int Strip(){return _strip;}
		
		void SetValues(int rank, int strip){
		
			_rank = rank;
			_strip = strip;
		}
		void SetRank(int rank){
		
			_rank = rank;
		}
	
	private:
	
	
		int _rank, _strip;
		
};

class SortingOutput{

	public:
	
		void SetHits(std::vector<ConvertedHit> hits){_hits = hits;};
		void SetWinners(std::vector<std::vector<Winner>> winners){_winners = winners;};
		void SetValues(std::vector<std::vector<Winner>> winners, std::vector<ConvertedHit> hits){_winners = winners;_hits = hits;};
		
		std::vector<ConvertedHit> Hits(){return _hits;};
		std::vector<std::vector<Winner>> Winners(){return _winners;};
		
	
	private:
	
		std::vector<ConvertedHit> _hits;
		std::vector<std::vector<Winner>> _winners;
	
};


typedef std::vector<std::vector<std::vector<std::vector<ConvertedHit>>>> ThOutput;
typedef std::vector<std::vector<std::vector<ConvertedHit>>> PhOutput;
class MatchingOutput{

	public:
	
		void SetHits(std::vector<ConvertedHit> hits){_hits = hits;};
		void SetWinners(std::vector<std::vector<Winner>> winners){_winners = winners;};
		void SetThOut(ThOutput th_output){_th_output = th_output;};
		void SetPhOut(PhOutput ph_output){_ph_output = ph_output;};
		void SetSegment(std::vector<int> segment){_segment = segment;};
		void SetValues(ThOutput th_output,PhOutput ph_output,std::vector<ConvertedHit> hits,std::vector<std::vector<Winner>> winners,std::vector<int> segment){
			
			_th_output = th_output;
			_ph_output = ph_output;
			_hits = hits;
			_winners = winners;
			_segment = segment;
		}
		
		
		
		std::vector<ConvertedHit> Hits(){return _hits;}; 
		std::vector<std::vector<Winner>> Winners(){return _winners;};
		ThOutput ThetaMatch(){return _th_output;};
		PhOutput PhiMatch(){return _ph_output;};
		std::vector<int> Segment(){return _segment;};
		
		
	private:
	
		std::vector<ConvertedHit> _hits;
		std::vector<std::vector<Winner>> _winners;
		ThOutput _th_output;
		PhOutput _ph_output;
		std::vector<int> _segment;
	
	
};

 
class DeltaOutput{

	public:
		void SetNull(){_Phi = -999;_Theta = -999;};
		void SetValues(MatchingOutput Mout, std::vector<std::vector<int>> Deltas, int Phi, int Theta, Winner winner){
		
			_Mout = Mout;_Deltas = Deltas;_Phi = Phi; _Theta = Theta;_winner = winner;
		}
		
		MatchingOutput GetMatchOut(){return _Mout;};
		std::vector<std::vector<int>> Deltas(){return _Deltas;};
		int Phi(){return _Phi;};
		int Theta(){return _Theta;};
		Winner GetWinner(){return _winner;};
	
	private:
	
		MatchingOutput _Mout;
		std::vector<std::vector<int>> _Deltas;
		int _Phi, _Theta;
		Winner _winner;
		


	
};

struct BTrack{

	BTrack(): phi(0),theta(0),clctpattern(0){}

	Winner winner;
	int phi;
	int theta;
	int clctpattern;
	std::vector<std::vector<int>>   deltas;
	std::vector<ConvertedHit> AHits;
};

#endif
