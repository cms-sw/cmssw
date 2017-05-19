#ifndef CALOWINDOW_H_
#define CALOWINDOW_H_

//#include <boost/shared_ptr.hpp>

//#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <iostream>
#include <map>
namespace pftools {

/**
 * As we shall see, a CaloWindow is composed of a number of CaloRings.
 * Each ring is divided into panes. Numbering of the panes starts from 0.
 */
class CaloRing {
public:

	/**
	 * Constructor for just one pane.
	 * @return
	 */
	CaloRing();

	/**
	 * Constructor for nPanes
	 * @param nPanes the number of panes
	 * @return
	 */
	CaloRing(unsigned nPanes);

	/**
	 * Zeroes all energies contained.
	 */
	void reset();

	/**
	 * Energy contained in that pane
	 * @param pane - numbering starts from 0
	 * @return energy contained in GeV
	 */
	double getEnergy(unsigned pane) const;

	/**
	 * Sets the energy in the specified pane
	 * @param pane - numbering starts from 0
	 * @param energy
	 * @return true if pane exists, false if outside of range
	 */
	bool setEnergy(unsigned pane, double energy);

	/**
	 * Adds energy to the specified pane
	 * @param pane - numbering starts from 0
	 * @param energy
	 * @return true if the pane exists, false if outside of range
	 */
	bool addEnergy(unsigned pane, double energy);

	/**
	 * Total energy contained by this CaloRing
	 * @return energy in GeV
	 */
	double totalE() const;

	/**
	 * A copy of the vector of energies for each pane
	 * @return a vector of energies
	 */
	std::vector<double> const & getEnergies() const {
		return myPanes_;
	}

	/**
	 * Get the number of panes
	 * @return
	 */
	unsigned getNPanes() const {
		return panes_;
	}

	void printEnergies(std::ostream& s, double range = 1.0);


private:
	unsigned panes_;
	std::vector<double> myPanes_;
};
 
std::ostream& operator<<(std::ostream& s, const CaloRing& caloRing);

class CaloWindow {
public:
	/* Default constructor - do not use (this has to be here for Reflex to work?)*/
	CaloWindow();

	/**
	 * Create a circular calo window centred on eta, phi, with nRings of size
	 * deltaR.
	 *
	 * */
	CaloWindow(double eta, double phi, unsigned nRings, double deltaR, unsigned nPanes, double axis = 0.0);

	void init(double eta, double phi, unsigned nRings, double deltaR, unsigned nPanes, double axis = 0.0);

	/**
	 * Adds a hit contribution. If eta, phi are outside the window, this method
	 * returns false and does not add the energy to the window.
	 *
	 * If ok, it returns true.
	 */
	bool addHit(double eta, double phi, double energy);

	/**
	 * Return a vector of vectors:
	 * Each inner vector corresponds to a ring of window panes around the barycentre
	 * and the first entry of the bary centre itself
	 *
	 */
	std::map<unsigned, CaloRing>const&  getRingDepositions() const {
		return energies_;
	}

	std::map<unsigned, double> getRingEnergySummations() const;


	virtual ~CaloWindow();

	void reset();

	void printEnergies(std::ostream& s, double range = 1.0);

	/**
	 * Collapses all energies in all CaloRings into one vector of doubles
	 * @param normalisation - divide each energy by this value
	 * @return a vector of doubles
	 */
	std::vector<double> stream(double normalisation = 1.0) const;

	double baryEta() const {return baryEta_;}
	double baryPhi() const { return baryPhi_;}

private:
	//Where is the barycentre of this window?
	double baryEta_;
	double baryPhi_;
	//How many rings is it composed of?
	unsigned nRings_;
	//What is the deltaR separation of the rings?
	double deltaR_;
	//How many panels will each calo ring be composed of? (central panel = 1)
	unsigned nPanes_;

	//Angle in radians fromm which we start counting rings (offset from zero)
	double axis_;

	/*
	 * Determines which window pane to put the hit in.
	 *
	 */
	std::pair<unsigned, unsigned> relativePosition(double eta, double phi) const;

	//std::vector<boost::shared_ptr<std::vector<double> > > energies_;
	std::map<unsigned, CaloRing> energies_;



};

std::ostream& operator<<(std::ostream& s,
                         const CaloWindow& caloWindow);

class TestCaloWindow {
public:
	TestCaloWindow() {
	}

	virtual ~TestCaloWindow() {
	}

	void doTest();
};

//typedef boost::shared_ptr<CaloWindow> CalowWindowPtr;

}
#endif /* CALOWINDOW_H_ */
