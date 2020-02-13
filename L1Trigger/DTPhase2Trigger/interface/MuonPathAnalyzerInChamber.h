#ifndef L1Trigger_DTPhase2Trigger_MuonPathAnalyzerInChamber_cc
#define L1Trigger_DTPhase2Trigger_MuonPathAnalyzerInChamber_cc

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"
#include "L1Trigger/DTPhase2Trigger/interface/analtypedefs.h"
#include "L1Trigger/DTPhase2Trigger/interface/constants.h"
#include "L1Trigger/DTPhase2Trigger/interface/MuonPathAnalyzer.h" 

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================
const int NLayers = 8;
typedef std::array<LATERAL_CASES, NLayers> TLateralities;

// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathAnalyzerInChamber : public MuonPathAnalyzer {
 public:
  // Constructors and destructor
  MuonPathAnalyzerInChamber(const edm::ParameterSet& pset);
  virtual ~MuonPathAnalyzerInChamber();
  
  // Main methods
  void initialise(const edm::EventSetup& iEventSetup);
  void run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, std::vector<MuonPath*> &inMpath, std::vector<metaPrimitive> &metaPrimitives) {}
  void run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, std::vector<MuonPath*> &inMpath, std::vector<MuonPath*> &outMPath);

  void finish();
  
  // Other public methods
  void setBXTolerance(int t);
  int getBXTolerance(void);
  
  void setMinHits4Fit(int h) { minHits4Fit = h;}; 
  int getMinHits4Fit(void) { return minHits4Fit; }

  void setChiSquareThreshold(float ch2Thr);
  
  void setMinimumQuality(MP_QUALITY q);
  MP_QUALITY getMinimumQuality(void);

  bool hasPosRF(int wh,int sec) {    return  wh>0 || (wh==0 && sec%4>1);   };

  // Public attributes
  edm::ESHandle<DTGeometry> dtGeo;

  //ttrig
  std::map<int,float> ttriginfo;
  
  //z 
  std::map<int,float> zinfo;
  
  //shift
  std::map<int,float> shiftinfo;
 
 
 private:
  
  // Private methods
  //  void analyze(MuonPath *inMPath, std::vector<metaPrimitive> &metaPrimitives);
  void analyze(MuonPath *inMPath, std::vector<MuonPath*> &outMPaths);
  
  void setCellLayout(MuonPath *mpath);
  void buildLateralities(MuonPath *mpath);
  void setLateralitiesInMP(MuonPath *mpath,TLateralities lat);
  void setWirePosAndTimeInMP(MuonPath *mpath);
  void calculateFitParameters(MuonPath *mpath, TLateralities lat, int present_layer[8]);
  //void calculateFitParameters(MuonPath *mpath, TLateralities lat);
 
  /* Determina si los valores de 4 primitivas forman una trayectoria
     Los valores tienen que ir dispuestos en el orden de capa:
     0    -> Capa más próxima al centro del detector,
     1, 2 -> Siguientes capas
     3    -> Capa más externa */
  void evaluateQuality(MuonPath *mPath);
  // Private attributes

  /* Combinaciones verticales de 3 celdas sobre las que se va a aplicar el
     mean-timer */
  static const int LAYER_ARRANGEMENTS[4][3];
  
  /* El máximo de combinaciones de lateralidad para 4 celdas es 16 grupos
     Es feo reservar todo el posible bloque de memoria de golpe, puesto que
     algunas combinaciones no serán válidas, desperdiciando parte de la
     memoria de forma innecesaria, pero la alternativa es complicar el
     código con vectores y reserva dinámica de memoria y, ¡bueno! ¡si hay
     que ir se va, pero ir p'a n'á es tontería! */

  int totalNumValLateralities;  
  std::vector<TLateralities> lateralities;
  std::vector<LATQ_TYPE> latQuality;
  
  /* Posiciones horizontales de cada celda (una por capa), en unidades de
     semilongitud de celda, relativas a la celda de la capa inferior
     (capa 0). Pese a que la celda de la capa 0 siempre está en posición
     0 respecto de sí misma, se incluye en el array para que el código que
     hace el procesamiento sea más homogéneo y sencillo */

  Bool_t debug;
  double chi2Th;
  edm::FileInPath z_filename;
  edm::FileInPath shift_filename;
  int bxTolerance;
  MP_QUALITY minQuality;
  float chiSquareThreshold;
  short minHits4Fit;
  int cellLayout[8];
  
};


#endif
