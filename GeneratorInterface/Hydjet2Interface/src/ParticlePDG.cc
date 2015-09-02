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

#ifndef PARTICLE_PDG
#include "GeneratorInterface/Hydjet2Interface/interface/ParticlePDG.h"
#endif

#include <iostream>

using namespace std;

ParticlePDG::ParticlePDG() {
  fPDG   = kNonsensePDG;
  fMass  = -1.0;
  fWidth = 0.0;
  fNDecayChannels = 0;
  for(int i=0; i<kMaxDecayChannels; i++)
    fDecayChannels[i] = new DecayChannel();
}

ParticlePDG::ParticlePDG(char *name, int pdg, double mass, double width) {
  for(int i=0; i<9; i++)
    if(*(name+i) != '\0') fName[i] = *(name+i);
    else break;
  fPDG   = pdg;
  fMass  = mass;
  fWidth = width;
  fNDecayChannels = 0;
  for(int i=0; i<kMaxDecayChannels; i++)
    fDecayChannels[i] = new DecayChannel();
}

ParticlePDG::~ParticlePDG() {
  for(int i=0; i<kMaxDecayChannels; i++)
    delete fDecayChannels[i];
}

double ParticlePDG::GetFullBranching() {
  double fullBranching = 0.0;
  for(int i=0; i<fNDecayChannels; i++)
    fullBranching += fDecayChannels[i]->GetBranching();
  return fullBranching;
}

void ParticlePDG::AddChannel(DecayChannel &channel) {
  if(channel.GetMotherPDG() != fPDG) {
    edm::LogError("ParticlePDG") <<" AddChannel() : You try to add a channel which has a different mother PDG";
    return;
  }
  fDecayChannels[fNDecayChannels]->SetMotherPDG(channel.GetMotherPDG());
  fDecayChannels[fNDecayChannels]->SetBranching(channel.GetBranching());
  fDecayChannels[fNDecayChannels]->SetDaughters(channel.GetDaughters(), channel.GetNDaughters());
  fNDecayChannels++;
}
