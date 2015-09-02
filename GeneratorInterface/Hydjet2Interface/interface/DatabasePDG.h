/*
  Copyright   : The FASTMC and SPHMC Collaboration
  Author      : Ionut Cristian Arsene 
  Affiliation : Oslo University, Norway & Institute for Space Sciences, Bucharest, Romania
  e-mail      : i.c.arsene@fys.uio.no
  Date        : 2007/05/30

  This class is using the particle and decay lists provided by the 
  THERMINATOR (Computer Physics Communications 174 669 (2006)) and
  SHARE (Computer Physics Communications 167 229 (2005)) collaborations.
*/

#ifndef DATABASE_PDG
#define DATABASE_PDG

#include "Rtypes.h"

#ifndef PARTICLE_PDG
#include "ParticlePDG.h"
#endif

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

const int kMaxParticles = 1000;

class DatabasePDG {
 private:
  int fNParticles;                        // no. of particles in database
  ParticlePDG *fParticles[kMaxParticles];   // array of particle pointers
  bool fStatus[kMaxParticles];            // status of each particle
  char fParticleFilename[256];            // particle list filename
  char fDecayFilename[256];               // decay channels filename
  bool fUseCharmParticles;                // flag for using (or not) charm particles
  double fMinimumWidth;                   // minimum allowed width for resonances
  double fMaximumWidth;                   // maximum allowed width for resonances
  double fMinimumMass;                    // minimum allowed mass for resonances
  double fMaximumMass;                    // maximum allowed mass for resonances

  bool LoadParticles();
  bool LoadDecays();
  void SortParticles();                     // put the good status particles at the beggining of the list
 public:
  DatabasePDG();
  ~DatabasePDG();

  // Load the particle PDG information from the particle and decay files
  bool LoadData();                        
  
  // Set particle and decay filenames
  void SetParticleFilename(char *filename);
  void SetDecayFilename(char *filename);
  // Set criteria for using particles. Those particle which do not match
  // these criteria will be flagged with FALSE in the fStatus array.
  void SetUseCharmParticles(bool flag);
  void SetMinimumWidth(double value);
  void SetMaximumWidth(double value);
  void SetMinimumMass(double value);
  void SetMaximumMass(double value);
  void SetWidthRange(double min, double max);
  void SetMassRange(double min, double max);
  
  // Read a list of pdg codes from a specified file. The corresponding particles
  // will be flagged as good particles. If the exclusive flag is TRUE than
  // only this criteria will be used in selecting particles and, in consequence,
  // all the other particles will be flagged as NOT good. If the exclusive flag
  // is FALSE than we will take into account all the previous applied criterias
  // and we will flag as good only particles in this list which match also the mass, width and
  // charmness criteria.
  // Note: In order for the exclusive=FALSE to be effective, this function must be called after
  // calling all the width, mass and charmness criteria functions.
  void UseThisListOfParticles(char *filename, bool exclusive = kTRUE);
  
  char* GetParticleFilename() {return fParticleFilename;}
  char* GetDecayFilename() {return fDecayFilename;}
  int GetNParticles(bool all = kFALSE);      // true - no. of all particles; false - no. of good status particles
  ParticlePDG* GetPDGParticleByIndex(int index);
  bool GetPDGParticleStatusByIndex(int index);
  ParticlePDG* GetPDGParticle(int pdg);
  bool GetPDGParticleStatus(int pdg);
  ParticlePDG* GetPDGParticle(char *name);
  bool GetPDGParticleStatus(char *name);
  bool GetUseCharmParticles() {return fUseCharmParticles;};
  double GetMinimumWidth() {return fMinimumWidth;};
  double GetMaximumWidth() {return fMaximumWidth;};
  double GetMinimumMass() {return fMinimumMass;};
  double GetMaximumMass() {return fMaximumMass;};
  void DumpData(bool dumpAll = kFALSE); // print the PDG information in the console
  int CheckImpossibleDecays(bool dump = kFALSE);   // print all impossible decays included in the database
  bool IsChannelAllowed(DecayChannel *channel, double motherMass);
  int GetNAllowedChannels(ParticlePDG *particle, double motherMass);
};
#endif
