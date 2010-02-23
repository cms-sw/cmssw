#ifndef Ntuple2HepMCFiller_H
#define Ntuple2HepMCFiller_H

/** \class Ntuple2HepMCFiller ***********
* Fills information from a converted cmkin ntpl into an HepMC event;
* The "h2root" converted ntpl is read by the class NtupleROOTFile
* Joanna Weng 1/2006 
***************************************/    


#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include "HepMC/GenEvent.h"
class NtupleROOTFile;
class Ntuple2HepMCFiller {
	
	protected:
	/// Constructor
	Ntuple2HepMCFiller();
	void setInitialized(bool value);	
	
	public:	
	/// Destructor
	virtual ~Ntuple2HepMCFiller();		
	static Ntuple2HepMCFiller * instance();	
	virtual void initialize(const std::string & filename, int id);	
	bool isInitialized();
	virtual bool setEvent(unsigned int event);
	virtual bool readCurrentEvent();
	virtual bool printHepMcEvent() const;	
	HepMC::GenEvent* fillCurrentEventData();
	virtual bool toGenEvent( int evtnum, HepMC::GenEvent* evt );
	HepMC::GenParticle* createParticle( int index );
	
	
	
	private:
	static Ntuple2HepMCFiller * instance_;
	// current  HepMC evt
	HepMC::GenEvent * evt;
	bool initialized_;
	NtupleROOTFile* input_;

	// # of particles in evt
	int  nParticles;
	//maps to convert HepMC::GenParticle to particles # and vice versa
	// -> needed for HepEvt like output
	std::vector<HepMC::GenParticle*> index_to_particle;  
	std::map<HepMC::GenParticle *,int> particle_to_index;	
	int evtid;  	
	int ntpl_id;   
	
	// find index to HepMC::GenParticle* p in map m
	int find_in_map(const std::map<HepMC::GenParticle*,int>& m,
	HepMC::GenParticle* p) const;
	void buildProductionVertex( int i, 
	std::vector<HepMC::GenParticle*>& hepevt_particle, 
	HepMC::GenEvent* evt, bool printInconsistencyErrors );		       
	int number_children( int index ) ;
	int number_parents( int index );
};

#endif // Ntuple2HepMCFiller_H

