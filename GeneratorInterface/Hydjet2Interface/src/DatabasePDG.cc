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
#include "GeneratorInterface/Hydjet2Interface/interface/DatabasePDG.h"
#endif

#include <cstring> 
#include <iostream>
#include <string>
#include <fstream>

using namespace std;
using namespace edm;

const char* particlesDATAstr;
const char* tableDECAYstr;

DatabasePDG::DatabasePDG() {
  fNParticles = 0;

  std::string file1="GeneratorInterface/Hydjet2Interface/data/particles.data";  
  edm::FileInPath f1(file1);  
  particlesDATAstr = ( (f1.fullPath()).c_str() );

  std::string file2="GeneratorInterface/Hydjet2Interface/data/tabledecay.txt";  
  edm::FileInPath f2(file2);  
  tableDECAYstr = ( (f2.fullPath()).c_str() );

  strcpy(fParticleFilename, particlesDATAstr);
  strcpy(fDecayFilename, tableDECAYstr);
  for(int i=0; i<kMaxParticles; i++) {
    fParticles[i] = new ParticlePDG();
    fStatus[i] = kFALSE;
  }
  fUseCharmParticles = kTRUE;
  fMinimumWidth = 0.;
  fMaximumWidth = 10.;
  fMinimumMass = 0.;
  fMaximumMass = 200.;
}

DatabasePDG::~DatabasePDG() {
  for(int i=0; i<kMaxParticles; i++)
    delete fParticles[i];
}

void DatabasePDG::SetParticleFilename(char *filename) {
  strcpy(fParticleFilename, filename);
}

void DatabasePDG::SetDecayFilename(char *filename) {
  strcpy(fDecayFilename, filename);
}

bool DatabasePDG::LoadData() {
  return (LoadParticles() && LoadDecays());
}

bool DatabasePDG::LoadParticles() {
  ifstream particleFile;
  particleFile.open(fParticleFilename);
  if(!particleFile) {
    edm::LogError("DatabasePDG")<< "The ASCII file containing the PDG particle list " << fParticleFilename << " was not found";
    return kFALSE;
  }
  
  char name[9];
  double mass, width, spin, isospin, isospinZ, q, s, aq, as, c, ac;
  int pdg;
  int goodStatusParticles = 0;

  edm::LogInfo("DatabasePDG")<< "Start loading particles with the following criteria:" << endl
       << "       Use particles containing charm quarks (1:yes;0:no) : " << fUseCharmParticles << endl
       << "       Mass range                                         : (" << fMinimumMass << "; " << fMaximumMass << ")" << endl
       << "       Width range                                        : (" << fMinimumWidth << "; " << fMaximumWidth << ")";
  
  particleFile.exceptions(ios::failbit);
  while(!particleFile.eof()) {
    try {
      particleFile >> name >> mass >> width >> spin >> isospin >> isospinZ >> q >> s >> aq >> as >> c >> ac >> pdg;
    }
    catch (ios::failure const &problem) {
      LogDebug("DatabasePDG")<<" ios:failure in particle file "<< problem.what();
      break;
    }
        
    fParticles[fNParticles]->SetName(name);
    fParticles[fNParticles]->SetPDG(pdg);
    fParticles[fNParticles]->SetMass(mass);
    fParticles[fNParticles]->SetWidth(width);
    fParticles[fNParticles]->SetSpin(spin);
    fParticles[fNParticles]->SetIsospin(isospin);
    fParticles[fNParticles]->SetIsospinZ(isospinZ);
    fParticles[fNParticles]->SetLightQNumber(q);
    fParticles[fNParticles]->SetStrangeQNumber(s);
    fParticles[fNParticles]->SetLightAQNumber(aq);
    fParticles[fNParticles]->SetStrangeAQNumber(as);
    fParticles[fNParticles]->SetCharmQNumber(c);
    fParticles[fNParticles]->SetCharmAQNumber(ac);
    goodStatusParticles++;
    fStatus[fNParticles] = kTRUE;
    // check if we want charmed particles
    if(!fUseCharmParticles && (c>0 || ac>0)) {
      fStatus[fNParticles] = kFALSE;
      goodStatusParticles--;
    }
    // check that the particle mass is inside accepted limits
    if(!(fMinimumMass<=mass && mass<=fMaximumMass)) {
      fStatus[fNParticles] = kFALSE;
      goodStatusParticles--;
    }
    // check that the particle width is inside accepted limits
    if(!(fMinimumWidth<=width && width<=fMaximumWidth)) {
      fStatus[fNParticles] = kFALSE;
      goodStatusParticles--;
    }

    fNParticles++;
  }
  particleFile.close();
  if(fNParticles==0) {

    LogWarning("DatabasePDG")<<" No particles were found in the file specified!!";
    return kFALSE;
  }
  SortParticles();
  edm::LogInfo("DatabasePDG")<< " Particle definitions found: " << fNParticles << ". Good status particles: " << goodStatusParticles;
  return kTRUE;
}

bool DatabasePDG::LoadDecays() {
  ifstream decayFile;
  decayFile.open(fDecayFilename);
  if(!decayFile) {
    edm::LogError("DatabasePDG")<< "The ASCII file containing the decays list " << fDecayFilename << " was not found";
    return kFALSE;
  }
  
  int mother_pdg, daughter_pdg[3];
  double branching;
  
  decayFile.exceptions(ios::failbit);
  while(!decayFile.eof()) {
    mother_pdg = 0;
    for(int i=0; i<3; i++) daughter_pdg[i] = 0;
    branching = -1.0;
    try {
      decayFile >> mother_pdg;
      for(int i=0; i<3; i++) 
        decayFile >> daughter_pdg[i];
      decayFile >> branching;
    }
    catch (ios::failure const &problem) {
      LogDebug("DatabasePDG")<<" ios:failure in decay file "<< problem.what();
      break;
    }
    if((mother_pdg!=0) && (daughter_pdg[0]!=0) && (branching>=0)) {
      int nDaughters = 0;
      for(int i=0; i<3; i++)
        if(daughter_pdg[i]!=0)
          nDaughters++;
      ParticlePDG* particle = GetPDGParticle(mother_pdg);
      if(!particle) {
	LogWarning("DatabasePDG")<<" Mother particle PDG (" << mother_pdg 
		<< ") not found in the particle definition list:"<< mother_pdg << " >>> ";
	for(int kk=0; kk<nDaughters; kk++) 
	  LogWarning("DatabasePDG")<< daughter_pdg[kk] << "  ";
	return kFALSE;
      }
      for(int kk=0; kk<nDaughters; kk++) {
	if(!GetPDGParticle(daughter_pdg[kk])) {
	  LogWarning("DatabasePDG")<<"Daughter particle PDG (" << daughter_pdg[kk] 
		<< ") not found in the particle definition list: " << mother_pdg << ">>> ";
	  for(int kkk=0; kkk<nDaughters; kkk++) 
	    LogWarning("DatabasePDG")<< daughter_pdg[kkk] << "  ";
	}
      }
      DecayChannel decay(mother_pdg, branching, nDaughters, daughter_pdg);
      particle->AddChannel(decay);
    }
  }
  decayFile.close();
  int nDecayChannels = 0;
  for(int i=0; i<fNParticles; i++) {
    nDecayChannels += fParticles[i]->GetNDecayChannels();
  }
  edm::LogInfo("DatabasePDG")<< "Number of decays found in the database is " << nDecayChannels;
  return kTRUE;
}

ParticlePDG* DatabasePDG::GetPDGParticleByIndex(int index) {
  if(index<0 || index>fNParticles) {
    edm::LogWarning("DatabasePDG")<< "Particle index is negative or too big !!" << endl
         << " It must be inside this range: (0, " << fNParticles-1 << ")" << endl
         << " Returning null pointer!!";
    return 0x0;
  }
  return fParticles[index];
}

bool DatabasePDG::GetPDGParticleStatusByIndex(int index) {
  if(index<0 || index>fNParticles) {
    edm::LogWarning("DatabasePDG")<< "Particle index is negative or too big !!" << endl
         << " It must be inside this range: (0, " << fNParticles-1 << ")" << endl
         << " Returning null pointer!!";
    return kFALSE;
  }
  return fStatus[index];
}

ParticlePDG* DatabasePDG::GetPDGParticle(int pdg) {
  int nFindings = 0;
  int firstTimeIndex = 0;
  for(int i=0; i<fNParticles; i++) {
    if(pdg == fParticles[i]->GetPDG()) {
      if(nFindings == 0) firstTimeIndex = i;
      nFindings++;
    }
  }
  if(nFindings == 1) return fParticles[firstTimeIndex];
  if(nFindings == 0) {
    edm::LogWarning("DatabasePDG")<< "The particle required with PDG: " << pdg
         << " was not found in the database!!";
    return 0x0;
  }
  if(nFindings >= 2) {
    edm::LogWarning("DatabasePDG")<< "The particle required with PDG: " << pdg
         << " was found with " << nFindings << " entries in the database. Check it out !!" << endl
	 << "Returning the first instance found";
    return fParticles[firstTimeIndex];
  }
  return 0x0;
}

bool DatabasePDG::GetPDGParticleStatus(int pdg) {
  int nFindings = 0;
  int firstTimeIndex = 0;
  for(int i=0; i<fNParticles; i++) {
    if(pdg == fParticles[i]->GetPDG()) {
      if(nFindings == 0) firstTimeIndex = i;
      nFindings++;
    }
  }
  if(nFindings == 1) return fStatus[firstTimeIndex];
  if(nFindings == 0) {
    edm::LogWarning("DatabasePDG")<< "The particle required with PDG: " << pdg
         << " was not found in the database!!";
    return kFALSE;
  }
  if(nFindings >= 2) {
    edm::LogWarning("DatabasePDG")<< "The particle status required for PDG: " << pdg
         << " was found with " << nFindings << " entries in the database. Check it out !!" << endl
	 << "Returning the status of first instance found";
    return fStatus[firstTimeIndex];
  }
  return kFALSE;
}

ParticlePDG* DatabasePDG::GetPDGParticle(char* name) {
  int nFindings = 0;
  int firstTimeIndex = 0;
  for(int i=0; i<fNParticles; i++) {
    if(!strcmp(name, fParticles[i]->GetName())) {
      if(nFindings == 0) firstTimeIndex = i;
      nFindings++;
    }
  }
  if(nFindings == 1) return fParticles[firstTimeIndex];
  if(nFindings == 0) {
    edm::LogWarning("DatabasePDG")<< "The particle required with name (" << name
         << ") was not found in the database!!";
    return 0x0;
  }
  if(nFindings >= 2) {
    edm::LogWarning("DatabasePDG")<< "The particle required with name (" << name
         << ") was found with " << nFindings << " entries in the database. Check it out !!" << endl
	 << "Returning the first instance found";
    return fParticles[firstTimeIndex];
  }
  return 0x0;
}

bool DatabasePDG::GetPDGParticleStatus(char* name) {
  int nFindings = 0;
  int firstTimeIndex = 0;
  for(int i=0; i<fNParticles; i++) {
    if(!strcmp(name, fParticles[i]->GetName())) {
      if(nFindings == 0) firstTimeIndex = i;
      nFindings++;
    }
  }
  if(nFindings == 1) return fStatus[firstTimeIndex];
  if(nFindings == 0) {
    edm::LogWarning("DatabasePDG")<< "The particle required with name (" << name
         << ") was not found in the database!!";
    return kFALSE;
  }
  if(nFindings >= 2) {
    edm::LogWarning("DatabasePDG")<< "The particle status required for name (" << name
         << ") was found with " << nFindings << " entries in the database. Check it out !!" << endl
	 << "Returning the first instance found";
    return fStatus[firstTimeIndex];
  }
  return kFALSE;
}

void DatabasePDG::DumpData(bool dumpAll) {
  cout << "***********************************************************************************************************" << endl;
  cout << "Dumping all the information contained in the database..." << endl;
  int nDecays = 0;
  int nGoodStatusDecays = 0;
  int nGoodStatusParticles = 0;
  for(int currPart=0; currPart<fNParticles; currPart++) {
    nGoodStatusParticles += (fStatus[currPart] ? 1:0);
    nGoodStatusDecays += (fStatus[currPart] ? fParticles[currPart]->GetNDecayChannels() : 0);
    nDecays += fParticles[currPart]->GetNDecayChannels();
    if(!(dumpAll || (!dumpAll && fStatus[currPart]))) continue;
    cout << "###### Particle: " << fParticles[currPart]->GetName() << " with PDG code " << fParticles[currPart]->GetPDG() << endl;
    cout << "   status          = " << fStatus[currPart] << endl;
    cout << "   mass            = " << fParticles[currPart]->GetMass() << " GeV" << endl;
    cout << "   width           = " << fParticles[currPart]->GetWidth() << " GeV" << endl;
    cout << "   2*spin          = " << int(2.*fParticles[currPart]->GetSpin()) << endl;
    cout << "   2*isospin       = " << int(2.*fParticles[currPart]->GetIsospin()) << endl;
    cout << "   2*isospin3      = " << int(2.*fParticles[currPart]->GetIsospinZ()) << endl;
    cout << "   u,d quarks      = " << int(fParticles[currPart]->GetLightQNumber()) << endl;
    cout << "   s quarks        = " << int(fParticles[currPart]->GetStrangeQNumber()) << endl;
    cout << "   c quarks        = " << int(fParticles[currPart]->GetCharmQNumber()) << endl;
    cout << "   anti u,d quarks = " << int(fParticles[currPart]->GetLightAQNumber()) << endl;
    cout << "   anti s quarks   = " << int(fParticles[currPart]->GetStrangeAQNumber()) << endl;
    cout << "   anti c quarks   = " << int(fParticles[currPart]->GetCharmAQNumber()) << endl;
    cout << "   baryon number   = " << int(fParticles[currPart]->GetBaryonNumber()) << endl;
    cout << "   strangeness     = " << int(fParticles[currPart]->GetStrangeness()) << endl;
    cout << "   charmness       = " << int(fParticles[currPart]->GetCharmness()) << endl;
    cout << "   electric charge = " << int(fParticles[currPart]->GetElectricCharge()) << endl;
    cout << "   full branching  = " << fParticles[currPart]->GetFullBranching() << endl;
    cout << "   decay modes     = " << fParticles[currPart]->GetNDecayChannels() << endl;
    for(int currChannel=0; currChannel<fParticles[currPart]->GetNDecayChannels(); currChannel++) {
      cout << "   channel " << currChannel+1 << " with branching " << fParticles[currPart]->GetDecayChannel(currChannel)->GetBranching() << endl;
      cout << "   daughters PDG codes: ";
      double daughtersMass = 0.0;
      for(int currDaughter=0; currDaughter<fParticles[currPart]->GetDecayChannel(currChannel)->GetNDaughters(); currDaughter++) {
        cout << fParticles[currPart]->GetDecayChannel(currChannel)->GetDaughterPDG(currDaughter) << "\t";
	ParticlePDG *daughter = GetPDGParticle(fParticles[currPart]->GetDecayChannel(currChannel)->GetDaughterPDG(currDaughter));
        daughtersMass += daughter->GetMass();
      }
      cout << endl;
      cout << "   daughters sum mass = " << daughtersMass << endl;
    }
  }
  if(dumpAll) {
    cout << "Finished dumping information for " << fNParticles << " particles with " << nDecays << " decay channels in total." << endl;
    cout << "*************************************************************************************************************" << endl;
  }
  else {
    cout << "Finished dumping information for " << nGoodStatusParticles << "(" << fNParticles << ")" 
	 << " particles with " << nGoodStatusDecays << "(" << nDecays << ")" << " decay channels in total." << endl;
    cout << "*************************************************************************************************************" << endl;
  }
}

int DatabasePDG::CheckImpossibleDecays(bool dump) {
  // Check the database for impossible decays
  int nImpossibleDecays = 0;
  for(int currPart=0; currPart<fNParticles; currPart++) {
    if(!fStatus[currPart]) continue;
    int allChannels = fParticles[currPart]->GetNDecayChannels();
    int allowedChannels = GetNAllowedChannels(fParticles[currPart], fParticles[currPart]->GetMass());
    if(dump) {
      cout << "Particle " << fParticles[currPart]->GetPDG() << " has " << allChannels << " decay channels specified in the database" << endl;
      cout << " Allowed channels assuming table mass = " << allowedChannels << endl;
    }
    if(dump && allChannels>0 && allowedChannels == 0) {
      cout << "**********************************************************************" << endl;
      cout << "       All channels for this particles are not allowed" << endl;
      cout << "**********************************************************************" << endl;
    }
    if(dump && fParticles[currPart]->GetWidth() > 0. && allChannels == 0) {
      cout << "**********************************************************************" << endl;
      cout << "    Particle has finite width but no decay channels specified" << endl;
      cout << "**********************************************************************" << endl;
    }
    for(int currChannel=0; currChannel<fParticles[currPart]->GetNDecayChannels(); currChannel++) {
      double motherMass = fParticles[currPart]->GetMass();
      double daughtersSumMass = 0.;
      for(int currDaughter=0; currDaughter<fParticles[currPart]->GetDecayChannel(currChannel)->GetNDaughters(); currDaughter++) {
        ParticlePDG *daughter = GetPDGParticle(fParticles[currPart]->GetDecayChannel(currChannel)->GetDaughterPDG(currDaughter));
        daughtersSumMass += daughter->GetMass();
      }
      if(daughtersSumMass >= motherMass) {
        nImpossibleDecays++;
        if(dump) {
          cout << "Imposible decay for particle " << fParticles[currPart]->GetPDG() << endl;
          cout << "  Channel: " << fParticles[currPart]->GetPDG() << " --> ";
          for(int currDaughter=0; currDaughter<fParticles[currPart]->GetDecayChannel(currChannel)->GetNDaughters(); currDaughter++) {
            ParticlePDG *daughter = GetPDGParticle(fParticles[currPart]->GetDecayChannel(currChannel)->GetDaughterPDG(currDaughter));
            cout << daughter->GetPDG() << " ";
          }
          cout << endl;
          cout << "  Mother particle mass = " << motherMass << endl;
          cout << "  Daughters sum mass   = " << daughtersSumMass << endl;
        }
      }
    }
  }
  return nImpossibleDecays;
}

void DatabasePDG::SetUseCharmParticles(bool flag) {
  if(fNParticles>0) {
    fUseCharmParticles = flag;
    for(int i=0; i<fNParticles; i++) {
      if(fParticles[i]->GetCharmQNumber()>0 || fParticles[i]->GetCharmAQNumber())		  
	fStatus[i] = flag;
    }
    SortParticles();
    return;
  }
  else
    fUseCharmParticles = flag;
  return;
}

void DatabasePDG::SetMinimumWidth(double value) {
  if(fNParticles>0) {
    fMinimumWidth = value;
    for(int i=0; i<fNParticles; i++) {
      if(fParticles[i]->GetWidth() < fMinimumWidth)		  
	fStatus[i] = kFALSE;
    }
    SortParticles();
    return;
  }
  else
    fMinimumWidth = value;
  return;
}

void DatabasePDG::SetMaximumWidth(double value) {
  if(fNParticles>0) {
    fMaximumWidth = value;
    for(int i=0; i<fNParticles; i++) {
      if(fParticles[i]->GetWidth() > fMaximumWidth)		  
	fStatus[i] = kFALSE;
    }
    SortParticles();
    return;
  }
  else
    fMaximumWidth = value;
  return;
}

void DatabasePDG::SetWidthRange(double min, double max) {
  if(fNParticles>0) {
    fMinimumWidth = min;
    fMaximumWidth = max;
    for(int i=0; i<fNParticles; i++) {
      if((fParticles[i]->GetWidth()<fMinimumWidth) || (fParticles[i]->GetWidth()>fMaximumWidth))  
	fStatus[i] = kFALSE;
    }
    SortParticles();

    return;
  }
  else {
    fMinimumWidth = min;
    fMaximumWidth = max;
  }

  return;
}

void DatabasePDG::SetMinimumMass(double value) {
  if(fNParticles>0) {
    fMinimumMass = value;
    for(int i=0; i<fNParticles; i++) {
      if(fParticles[i]->GetMass() < fMinimumMass)		  
	fStatus[i] = kFALSE;
    }
    SortParticles();
    return;
  }
  else
    fMinimumMass = value;
  return;
}

void DatabasePDG::SetMaximumMass(double value) {
  if(fNParticles>0) {
    fMaximumMass = value;
    for(int i=0; i<fNParticles; i++) {
      if(fParticles[i]->GetMass() > fMaximumMass)		  
	fStatus[i] = kFALSE;
    }
    SortParticles();
    return;
  }
  else
    fMaximumMass = value;
  return;
}

void DatabasePDG::SetMassRange(double min, double max) {



  if(fNParticles>0) {
    fMinimumMass = min;
    fMaximumMass = max;
    for(int i=0; i<fNParticles; i++) {
      if((fParticles[i]->GetMass()<fMinimumMass) || (fParticles[i]->GetMass()>fMaximumMass))  
	fStatus[i] = kFALSE;
    }
    SortParticles();

    return;
  }
  else {
    fMinimumMass = min;
    fMaximumMass = max;
  } 

  return;
}

void DatabasePDG::SortParticles() {


  if(fNParticles<2) {
    edm::LogWarning("DatabasePDG")<< "No particles to sort. Load data first!!";
    return;
  }

  int nGoodStatus = 0;
  for(int i=0; i<fNParticles; i++)
    if(fStatus[i]) nGoodStatus++;

  if(nGoodStatus==fNParticles)    // if all particles have good status then there is nothing to do
    return;

  if(nGoodStatus==0)              // no good status particles, again nothing to do
    return;

  int shifts = 1;
  while(shifts) {
    shifts = 0;
    for(int i=0; i<fNParticles-1; i++) {
      if(!fStatus[i] && fStatus[i+1]) {   // switch if false status is imediately before a true status particle
	ParticlePDG *temporaryPointer = fParticles[i];
	fParticles[i] = fParticles[i+1];
	fParticles[i+1] = temporaryPointer;
	bool temporaryStatus = fStatus[i];
	fStatus[i] = fStatus[i+1];
	fStatus[i+1] = temporaryStatus;
	shifts++;
      }
    }
  }

  return;
}

int DatabasePDG::GetNParticles(bool all) {
  if(all)
    return fNParticles;

  int nGoodStatus = 0;
  for(int i=0; i<fNParticles; i++)
    if(fStatus[i]) nGoodStatus++;
  return nGoodStatus;
}

void DatabasePDG::UseThisListOfParticles(char *filename, bool exclusive) {
  if(fNParticles<1) {
    edm::LogError("DatabasePDG")<< "You must load the data before calling this function!!";
    return;
  }

  ifstream listFile;
  listFile.open(filename);
  if(!listFile) {
    edm::LogError("DatabasePDG")<< "The ASCII file containing the PDG codes list ("
         << filename << ") was not found !!";
    return;
  }

  bool flaggedIndexes[kMaxParticles];
  for(int i=0; i<kMaxParticles; i++)
    flaggedIndexes[i] = kFALSE;
  int pdg = 0;
  listFile.exceptions(ios::failbit);
  while(!listFile.eof()) {
    try {
      listFile >> pdg;
    }
    catch (ios::failure const &problem) {
      LogDebug("DatabasePDG")<< "ios:failure in list file"<<  problem.what();
      break;
    }
    int found = 0;
    for(int i=0; i<fNParticles; i++) {
      if(fParticles[i]->GetPDG()==pdg) {
	found++;
	flaggedIndexes[i] = kTRUE;
      }
    }
    if(!found) {
      edm::LogWarning("DatabasePDG")<< "The particle with PDG code "
	   << pdg << " was asked but not found in the database!!";
    }
    if(found>1) {
      edm::LogWarning("DatabasePDG")<< "The particle with PDG code "
	   << pdg << " was found more than once in the database!!";
    }
  }

  if(exclusive) {
    for(int i=0; i<kMaxParticles; i++)
      fStatus[i] = flaggedIndexes[i];
  }
  else {
    for(int i=0; i<kMaxParticles; i++)
      fStatus[i] = (fStatus[i] && flaggedIndexes[i]);
  }
  SortParticles();

  return;
}

bool DatabasePDG::IsChannelAllowed(DecayChannel *channel, double motherMass) {
  double daughtersSumMass = 0.0;
  for(int i=0; i<channel->GetNDaughters(); i++)
    daughtersSumMass += GetPDGParticle(channel->GetDaughterPDG(i))->GetMass();
  if(daughtersSumMass<=motherMass)
    return kTRUE;
  return kFALSE;
}

int DatabasePDG::GetNAllowedChannels(ParticlePDG *particle, double motherMass) {
  int nAllowedChannels = 0;
  for(int i=0; i<particle->GetNDecayChannels(); i++)
    nAllowedChannels += (IsChannelAllowed(particle->GetDecayChannel(i), motherMass) ? 1:0);
  return nAllowedChannels;
}
