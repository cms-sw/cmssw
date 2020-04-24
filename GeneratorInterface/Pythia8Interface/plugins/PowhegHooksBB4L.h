// PowhegHooksBB4L.h 

// Author: Tomas Jezo, Markus Seidel and Ben Nachman based on 
// PowhegHooks.h by Richard Corke

#ifndef Pythia8_PowhegHooksBB4L_H
#define Pythia8_PowhegHooksBB4L_H

// Includes
#include "Pythia8/Pythia.h"
//#include "Pythia8Plugins/PowhegHooks.h"

namespace Pythia8 {

//==========================================================================

// Use userhooks to veto PYTHIA emissions above the POWHEG scale.

class PowhegHooksBB4L : public UserHooks {
//class PowhegHooksBB4L : public PowhegHooks {

public:

	// Constructor and destructor.
	PowhegHooksBB4L() {}
	~PowhegHooksBB4L() {}
//--------------------------------------------------------------------------

	// Initialize settings
	bool initAfterBeams() {			
		// settings of the parent class
//		PowhegHook::initAfterBeams();
		// settings of this class
		onlyDistance1 = settingsPtr->flag("POWHEG:bb4l:onlyDistance1");
		useScaleResonanceInstead = settingsPtr->flag("POWHEG:bb4l:useScaleResonanceInstead");
		// variables used during the veto
		topresscale = -1;	
		atopresscale = -1;	
		return true;
	}

//--------------------------------------------------------------------------

	// Calculates the scale of the hardest emission from within the resonance system
	// translated by Markus Seidel modified by Tomas Jezo
	inline double findresscale( const int iRes, const Event& event) {
		double scale = 0.;
    
		int nDau = event[iRes].daughterList().size();
    
		if (nDau == 0) {
			// No resonance found, set scale to high value
			// Pythia will shower any MC generated resonance unrestricted
			scale = 1e30;
		}
		else if (nDau < 3) {
			// No radiating resonance found
			scale = 0.8;
		}
		else if (std::abs(event[iRes].id()) == 6) {
			// Find top daughters
			int idw = -1, idb = -1, idg = -1;
			
			for (int i = 0; i < nDau; i++) {
				int iDau = event[iRes].daughterList()[i];
				if (std::abs(event[iDau].id()) == 24) idw = iDau;
				if (std::abs(event[iDau].id()) ==  5) idb = iDau;
				if (std::abs(event[iDau].id()) == 21) idg = iDau;
			}
			
			// Get daughter 4-vectors in resonance frame
			Vec4 pw(event[idw].p());
			pw.bstback(event[iRes].p());
			
			Vec4 pb(event[idb].p());
			pb.bstback(event[iRes].p());
			
			Vec4 pg(event[idg].p());
			pg.bstback(event[iRes].p());
			
			// Calculate scale
			scale = sqrt(2*pg*pb*pg.e()/pb.e());
		}
		else {
			scale = 1e30;
		}

		return scale;
	}

//--------------------------------------------------------------------------

	// The following routine will match daughters of particle `e[iparticle]` to an expected pattern specified via the list of expected particle PDG ID's `ids`, 
	// id wildcard is specified as 0 if match is obtained, the positions and the momenta of these particles are returned in vectors `positions` and `momenta` 
	// respectively
	// if exitOnExtraLegs==true, it will exit if the decay has more particles than expected, but not less
	inline bool match_decay(int iparticle, const Event &e, const vector<int> &ids, vector<int> &positions, vector<Vec4> &momenta, bool exitOnExtraLegs = true){
		// compare sizes
		if (e[iparticle].daughterList().size() != ids.size()) {
			if (exitOnExtraLegs && e[iparticle].daughterList().size() > ids.size()) exit(-1);
			return false; 
		}
		// compare content
		for (unsigned int i = 0; i < e[iparticle].daughterList().size(); i++) {
			int di = e[iparticle].daughterList()[i];
			if (ids[i] != 0 && e[di].id() != ids[i]) 
				return false;
		}
		// reset the positions and momenta vectors (because they may be reused)
		positions.clear();
		momenta.clear();
		// construct the array of momenta
		for (unsigned int i = 0; i < e[iparticle].daughterList().size(); i++) {
			int di = e[iparticle].daughterList()[i];
			positions.push_back(di);
			momenta.push_back(e[di].p());
		}
		return true;
	}

	inline double qSplittingScale(Vec4 pt, Vec4 p1, Vec4 p2){
		p1.bstback(pt);
		p2.bstback(pt);
		return sqrt( 2*p1*p2*p2.e()/p1.e() );
	}

	inline double gSplittingScale(Vec4 pt, Vec4 p1, Vec4 p2){
		p1.bstback(pt);
		p2.bstback(pt);		
		return sqrt( 2*p1*p2*p1.e()*p2.e()/(pow(p1.e(),2)+pow(p2.e(),2)) );
	}


	inline double getdechardness(int topcharge, const Event &e){
		// construct pdg ids of top and its decay products
		int tid = 6*topcharge, wid = 24*topcharge, bid = 5*topcharge, gid = 21, wildcard = 0;
		// find last top in the record
		int i_top = -1;
		Vec4 p_top, p_b, p_g, p_g1, p_g2;
		for (int i = 0; i < e.size(); i++) 
			if (e[i].id() == tid) {				
				i_top = i;
				p_top = e[i].p();
			}
		if (i_top == -1) return -1.0;
				
		// summary of cases
		// 1.) t > W b
		//   a.) b > 3     ... error
		//   b.) b > b g   ... h = sqrt(2*p_g*p_b*p_g.e()/p_b.e())
		//   c.) b > other ... h = -1
		//   return h
		// 2.) t > W b g
		//   a.)   b > 3     ... error
		//   b.)   b > b g   ... h1 = sqrt(2*p_g*p_b*p_g.e()/p_b.e())
		//   c.)   b > other ... h1 = -1
		//   i.)   g > 3     ... error
		//   ii.)  g > 2     ... h2 = sqrt(2*p_g1*p_g2*p_g1.e()*p_g2.e()/(pow(p_g1.e(),2)+pow(p_g2.e(),2))) );
		//   iii.) g > other ... h2 = -1
		//   return max(h1,h2)
		// 3.) else ... error

		vector<Vec4> momenta;
		vector<int> positions;

		// 1.) t > b W
		if ( match_decay(i_top, e, vector<int> {wid, bid}, positions, momenta, false) ) {
			double h;
			int i_b = positions[1];
			// a.+b.) b > 3 or b > b g 
			if ( match_decay(i_b, e, vector<int> {bid, gid}, positions, momenta) )
				h = qSplittingScale(e[i_top].p(), momenta[0], momenta[1]);
			// c.) b > other
			else 
				h = -1;
			return h;
		} 
		// 2.) t > b W g
		else if ( match_decay(i_top, e, vector<int> {wid, bid, gid}, positions, momenta, false) ) {
			double h1, h2;
			int i_b = positions[1], i_g = positions[2];
			// a.+b.) b > 3 or b > b g
			if ( match_decay(i_b, e, vector<int> {bid, gid}, positions, momenta) )
				h1 = qSplittingScale(e[i_top].p(), momenta[0], momenta[1]);
			// c.) b > other
			else 
				h1 = -1;
			// i.+ii.) g > 3 or g > 2
			if ( match_decay(i_g, e, vector<int> {wildcard, wildcard}, positions, momenta) )
				h2 = gSplittingScale(e[i_top].p(), momenta[0], momenta[1]);
			// c.) b > other
			else 
				h2 = -1;
			return max(h1, h2);
		}
		// 3.) else
		else { 
			exit(-1);
		}
	}	

//--------------------------------------------------------------------------

	// called after shower -- cannot be used for veto, because the event will get discarded
	inline bool canVetoPartonLevel() { return true; }
	inline bool doVetoPartonLevel(const Event &e) {
		double topdechardness = getdechardness(1, e),  atopdechardness = getdechardness(-1, e);
		if ((topdechardness > topresscale) or (atopdechardness > atopresscale)) {
//			cout << " PYTHIA Warning in PowhegHooksBB4L::doVetoPartonLevel: passed doVetoFSREmission veto, but wouldn't have passed veto based on the full event listing" << endl;
			infoPtr->errorMsg("Warning in PowhegHooksBB4L::doVetoPartonLevel: passed doVetoFSREmission veto, but wouldn't have passed veto based on the full event listing");
		}
//		cout << " veto scales: " << fixed << setprecision(17) << setw(30) << topresscale << setw(30) << atopresscale << endl;
		topresscale = -1;	
		atopresscale = -1;	

		return false;
	}

	// called before each resonance decay shower
	inline bool canSetResonanceScale() { return true; }
	inline double scaleResonance(int iRes, const Event &e) {		
		double scale = 1e30;
		if (e[iRes].id() == 6) 
			scale = topresscale = findresscale(iRes, e);
		else if (e[iRes].id() == -6) 
			scale = atopresscale = findresscale(iRes, e);
		if (useScaleResonanceInstead) 
			return scale;
		else 			
			return 1e30;
	}

//--------------------------------------------------------------------------

	// FSR veto
	inline bool canVetoFSREmission() { 
		if (useScaleResonanceInstead) 
			return false;
		else 
			return true; 
	}
	inline bool doVetoFSREmission(int sizeOld, const Event &e, int iSys, bool inResonance) {

		// call parent PowhegHook veto
//		if (PowhegHooks::doVectoFSREmission(sizeOld, Event &e, iSys, inResonance)) return true;

		if (inResonance) {

			// get the participants of the splitting: the radiator and the emitted
			int iEmt = e.size() - 2;
			int iRadAft = e.size() - 3;
			int iRadBef = e[iEmt].mother1();

			// find the top resonance the radiator originates from
			int iTop = e[iRadBef].mother1();
			int distance = 1;
			while (std::abs(e[iTop].id()) != 6 && iTop > 0) {
				iTop = e[iTop].mother1();
				distance ++;
			}
			if (iTop == 0) {
				infoPtr->errorMsg("Warning in PowhegHooksBB4L::doVetoFSREmission: emission in resonance not from top quark, not vetoing");
				return false;
			}
			int iTopCharge = (e[iTop].id()>0)?1:-1;


			// calculate the scale of the emission
			Vec4 pr(e[iRadAft].p()), pe(e[iEmt].p()), pt(e[iTop].p());
			double scale;
			// gluon splitting into two partons
			if (e[iRadBef].id() == 21)
				scale = gSplittingScale(pt, pr, pe);
			// quark emitting a gluon
			else if (std::abs(e[iRadBef].id()) <= 5)
				scale = qSplittingScale(pt, pr, pe);
			// other stuff (which we should not veto)
			else {
				scale = 0;
			}

			if (iTopCharge > 0) {
				if (onlyDistance1) {
					if ((distance == 1) && scale > topresscale) {
					  nFSRveto++;
					  return true;
					}
				}
				else
					if (scale > topresscale) {
					  nFSRveto++;
					  return true;
					}
			}
			else {
				if (onlyDistance1) {
					if ((distance == 1) && scale > atopresscale) {
					  nFSRveto++;
					  return true;
					}
				}
				else
					if (scale > atopresscale) {
					  nFSRveto++;
					  return true;
					}
			}
		}
		return false;
	}

//--------------------------------------------------------------------------

  // Functions to return information

  inline int    getNFSRveto() { return nFSRveto; }

//--------------------------------------------------------------------------

private:
	int nFSRveto = 0;
	double topresscale, atopresscale;
	bool onlyDistance1, useScaleResonanceInstead;
};

//==========================================================================

} // end namespace Pythia8

#endif // end Pythia8_PowhegHooksBB4L_H
