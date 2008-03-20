#ifndef AnalysisRootpleProducer_H
#define AnalysisRootpleProducer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
//#include "TProfile.h"
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TProfile.h>
class TTree;
// forward declarations
class TFile;
class TH1D;

class AnalysisRootpleProducer : public edm::EDAnalyzer
{
  
public:
  
  //
  explicit AnalysisRootpleProducer( const edm::ParameterSet& ) ;
  virtual ~AnalysisRootpleProducer() {} // no need to delete ROOT stuff
  // as it'll be deleted upon closing TFile
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob( const edm::EventSetup& ) ;
  virtual void endJob() ;
  
  void fillEventInfo(int);
  void fillMCParticles(float, float, float, float);
  void fillTracks(float, float, float, float);
  void fillInclusiveJet(float, float, float, float);
  void fillChargedJet(float, float, float, float);
  void fillTracksJet(float, float, float, float);
  void fillCaloJet(float, float, float, float);
  void store();

private:
  
  //
  
  std::string fOutputFileName;
  bool onlyRECO;
  std::string mcEvent;
  std::string genJetCollName;
  std::string chgJetCollName;
  std::string tracksJetCollName;
  std::string recoCaloJetCollName;
  std::string chgGenPartCollName;
  std::string tracksCollName;
  
  float piG;
  TFile* hFile;
  TTree* AnalysisTree;

  static const int NMCPMAX = 10000;   
  static const int NTKMAX = 10000;
  static const int NIJMAX = 10000;
  static const int NCJMAX = 10000;
  static const int NTJMAX = 10000;
  static const int NEHJMAX = 10000;

  int EventKind,NumberMCParticles,NumberTracks,NumberInclusiveJet,NumberChargedJet,NumberTracksJet,NumberCaloJet;
  
  float MomentumMC[NMCPMAX],TransverseMomentumMC[NMCPMAX],EtaMC[NMCPMAX],PhiMC[NMCPMAX];
  float MomentumTK[NTKMAX],TransverseMomentumTK[NTKMAX],EtaTK[NTKMAX],PhiTK[NTKMAX];
  float MomentumIJ[NIJMAX],TransverseMomentumIJ[NIJMAX],EtaIJ[NIJMAX],PhiIJ[NIJMAX];
  float MomentumCJ[NCJMAX],TransverseMomentumCJ[NCJMAX],EtaCJ[NCJMAX],PhiCJ[NCJMAX];
  float MomentumTJ[NTJMAX],TransverseMomentumTJ[NTJMAX],EtaTJ[NTJMAX],PhiTJ[NTJMAX];
  float MomentumEHJ[NEHJMAX],TransverseMomentumEHJ[NEHJMAX],EtaEHJ[NEHJMAX],PhiEHJ[NEHJMAX];
};

#endif
