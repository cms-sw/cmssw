#ifndef HepMCFileReader_H
#define HepMCFileReader_H

/** \class HepMCFileReader
* 
*  This class is used by the implementation of DaqEventFactory present 
*  in this package to read in the full event raw data from a flat 
*  binary file. 
*  WARNING: If you want to use this class for other purposes you must 
*  always invoke the method initialize before starting using the interface
*  it exposes.
*
*  $Date: 2005/08/19 09:01:43 $
*  $Revision: 1.2 $
*  \author G. Bruno - CERN, EP Division
*/   
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include "CLHEP/HepMC/ReadHepMC.h"
#include "CLHEP/HepMC/GenEvent.h"

class HepMCFileReader {
	
	protected:
	
	/// Constructor
	HepMCFileReader();
	void setInitialized(bool value);
	
	public:	
	/// Destructor
	virtual ~HepMCFileReader();	
	static HepMCFileReader * instance();
	virtual void initialize(const std::string & filename);	
	bool isInitialized();

	
	virtual bool setEvent(int event);
	virtual bool readCurrentEvent();
	virtual bool printHepMcEvent() const;	
	HepMC::GenEvent* fillCurrentEventData();
	  //	virtual bool fillEventData(HepMC::GenEvent  * event);
	// this method prints the event information as 
	// obtained by the input file in HepEvt style
	void printEvent() const;
	// get all the 'integer' properties of a particle 
	// like mother, daughter, pid and status
	// 'j' is the number of the particle in the HepMc
	virtual void getStatsFromTuple(int &mo1, int &mo2, 
	int &da1, int &da2, 
	int &status, int &pid,
	int j) const;
	virtual void ReadStats();



	private:
	static HepMCFileReader * instance_;
	// current  HepMC evt
	HepMC::GenEvent * evt;
	bool initialized_;
	std::ifstream * input_;
      	// # of particles in evt
	int  nParticles;

	//maps to convert HepMC::GenParticle to particles # and vice versa
	// -> needed for HepEvt like output
	std::vector<HepMC::GenParticle*> index_to_particle;  
	std::map<HepMC::GenParticle *,int> particle_to_index;    
	// find index to HepMC::GenParticle* p in map m
	int find_in_map(const std::map<HepMC::GenParticle*,int>& m,
	HepMC::GenParticle* p) const;
	

	
};

#endif // HepMCFileReader_H

