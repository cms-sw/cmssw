// -*- C++ -*-
//
// Class:      EcalTPGAnalyzer
// 
//
// Original Author:  Pascal Paganini
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <vector>
#include <string>
#include <TFile.h>
#include <TTree.h>

class CaloSubdetectorGeometry ;


// Auxiliary class
class towerEner {   
 public:
  float eRec_ ;
  int tpgEmul_[5] ;
  int tpgADC_; 
  int iphi_, ieta_, nbXtal_ ;
  towerEner()
    : eRec_(0), tpgADC_(0),  
      iphi_(-999), ieta_(-999), nbXtal_(0)
  { 
    for (int i=0 ; i<5 ; i ++) tpgEmul_[i] = 0 ; 
  }
};


class EcalTPGAnalyzer : public edm::EDAnalyzer {
public:
  explicit EcalTPGAnalyzer(const edm::ParameterSet&);
  ~EcalTPGAnalyzer() override;  
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override ;
  
private:
  struct EcalTPGVariables
  {
    // event variables
    unsigned int runNb ;
    unsigned int evtNb ;
    unsigned int bxNb ;
    unsigned int orbitNb ;
    unsigned int nbOfActiveTriggers ;
    int activeTriggers[128] ;

    // tower variables
    unsigned int nbOfTowers ; //max 4032 EB+EE
    int ieta[4032] ;
    int iphi[4032] ;
    int nbOfXtals[4032] ;
    int rawTPData[4032] ;
    int rawTPEmul1[4032] ;
    int rawTPEmul2[4032] ;
    int rawTPEmul3[4032] ;
    int rawTPEmul4[4032] ;
    int rawTPEmul5[4032] ;
    float eRec[4032] ;
  } ;

private:
  TFile * file_;
  TTree * tree_ ;
  EcalTPGVariables treeVariables_ ;

  edm::InputTag tpCollection_ ;
  edm::InputTag tpEmulatorCollection_ ;
  edm::InputTag digiCollectionEB_ ;
  edm::InputTag digiCollectionEE_ ;
  std::string gtRecordCollectionTag_ ;

  bool allowTP_ ;
  bool useEE_ ;
  bool print_ ;

  const CaloSubdetectorGeometry * theEndcapGeometry_ ;
  const CaloSubdetectorGeometry * theBarrelGeometry_ ;
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;


};

