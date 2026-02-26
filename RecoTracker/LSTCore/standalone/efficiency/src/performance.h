#ifndef process_h
#define process_h

#include "LSTEff.h"
#include "rooutil.h"
#include "cxxopts.h"
#include "helper.h"

void bookEfficiencySets(std::vector<SimTrackSetDefinition>& effset);
void bookEfficiencySet(SimTrackSetDefinition& effset);
void bookFakeRateSets(std::vector<RecoTrackSetDefinition>& FRset);
void bookFakeRateSet(RecoTrackSetDefinition& FRset);
void bookDuplicateRateSets(std::vector<RecoTrackSetDefinition>& DRset);
void bookDuplicateRateSet(RecoTrackSetDefinition& DRset);
void bookFakeOrDuplicateRateSets(std::vector<RecoTrackSetDefinition>& FDRset);
void bookFakeOrDuplicateRateSet(RecoTrackSetDefinition& FDRset);
void bookOTLengthSets(std::vector<RecoTrackSetDefinition>& OLsets);
void bookOTLengthSet(RecoTrackSetDefinition& OLset);

void fillEfficiencySets(std::vector<SimTrackSetDefinition>& effset);

void fillEfficiencySet(int isimtrk,
                       SimTrackSetDefinition& effset,
                       float pt,
                       float eta,
                       float dz,
                       float dxy,
                       float phi,
                       int pdgidtrk,
                       int q,
                       float vtx_x,
                       float vtx_y,
                       float vtx_z,
                       float deltaEta,
                       float deltaPhi,
                       float deltaR,
                       float genJetPt);
void fillEfficiencySet(int isimtrk,
                       SimTrackSetDefinition& effset,
                       float pt,
                       float eta,
                       float dz,
                       float dxy,
                       float phi,
                       int pdgidtrk,
                       int q,
                       float vtx_x,
                       float vtx_y,
                       float vtx_z);

void fillOTLengthSets(std::vector<RecoTrackSetDefinition>& OLsets);
void fillOTLengthSet(int itc, RecoTrackSetDefinition& OLset, float pt, float eta);
void fillFakeRateSets(std::vector<RecoTrackSetDefinition>& FRset);
void fillFakeRateSet(int isimtrk, RecoTrackSetDefinition& FRset, float pt, float eta, float phi);
void fillDuplicateRateSets(std::vector<RecoTrackSetDefinition>& DRset);
void fillDuplicateRateSet(int isimtrk, RecoTrackSetDefinition& DRset, float pt, float eta, float phi);
void fillFakeOrDuplicateRateSets(std::vector<RecoTrackSetDefinition>& FDRset);
void fillFakeOrDuplicateRateSet(int isimtrk, RecoTrackSetDefinition& FDRset, float pt, float eta, float phi);

#endif
