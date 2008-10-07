#ifndef ChargedHadronSpectra_Histograms
#define ChargedHadronSpectra_Histograms

#include <vector>

namespace edm { class ParameterSet ; }
class TFile;
class TTree;
class TH1F;
class TH2F;
class TH3F;

typedef struct
{
  int ids;
  float etas;
  float pts;
//  bool acc;
//  bool prim;
  int acc;
  int prim;
  int nrec;
  int ntrkr;
} SimTrack_t;

typedef struct
{
  int charge;
  float etar;
  float ptr;
  float phir;
  float zr;
  float logpr;
  float logde;
  int nhitr; 
//  bool prim;
  int prim;
  int nsim;
  int ids;
  int parids;
  float etas;
  float pts;
  int ntrkr;
} RecTrack_t;

typedef struct
{
  float etar;
  float ptr;
  float ima;
  float rhor;
} RecVzero_t;

typedef struct
{
  int proc;
  int strk;
  int ntrkr;
} EventInfo_t;

class Histograms
{
 public:
  Histograms(const edm::ParameterSet& pset);
  ~Histograms();

  void declareHistograms();
  void fillEventInfo(int proc,int strk, int ntrkr);
  void fillSimHistograms  (const SimTrack_t & s);
  void fillRecHistograms  (const RecTrack_t & r);
  void fillVzeroHistograms(const RecVzero_t & r, int part);
  void writeHistograms();

 private: 
   std::vector<TTree *> trackTrees;
   SimTrack_t simTrackValues;
   RecTrack_t recTrackValues;
   RecVzero_t recVzeroValues;
   EventInfo_t eventInfoValues;

   TFile * histoFile;
   TFile * ntupleFile;
   bool fillHistograms;
   bool fillNtuples;

   std::vector<double> etaBins, metaBins, ptBins, ratBins,
                       zBins, lpBins, ldeBins, nhitBins,
                       rhoBins, ntrkBins;

   int getParticle(int id);
   int getCharge(int charge);

   // SimTrack
   std::vector<TH1F *> heve;
   std::vector<TH2F *> hder;

   // SimTrack
   std::vector<TH3F *> hsim;
   std::vector<TH3F *> hacc;
   std::vector<TH3F *> href;
   std::vector<TH3F *> hmul;

   // RecTrack
   std::vector<TH3F *> hall;
//   std::vector<TH3F *> hdac;
   std::vector<TH2F *> hdac;

   // RecTrack, resolution, bias
   std::vector<TH3F *> hvpt;
   std::vector<TH3F *> hrpt;
   std::vector<TH2F *> hsp0;
   std::vector<TH2F *> hsp1;
   std::vector<TH2F *> hsp2;

   // RecTrack -- FakeRate
   std::vector<TH3F *> hfak;

   // RecTrack -- FeedDown
   std::vector<TH2F *> hpro;
   std::vector<TH2F *> hdec;

   // RecTrack -- EnergyLoss
   std::vector<TH3F *> helo;
   std::vector<TH3F *> hnhi;
   std::vector<TH2F *> held;

/*
   std::vector<TH3F *> selo;
   std::vector<TH3F *> snhi;
   std::vector<TH2F *> seld;
*/
 
   // RecVzero -- InvariantMass
   std::vector<TH3F *> hima;
   std::vector<TH3F *> hrho;
};

#endif
