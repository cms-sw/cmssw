#ifndef L1Trigger_DTTriggerPhase2_MuonPathAnalyzerInChamber_cc
#define L1Trigger_DTTriggerPhase2_MuonPathAnalyzerInChamber_cc

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

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzer.h"

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
namespace {
  constexpr int NLayers = 8;
  typedef std::array<LATERAL_CASES, NLayers> TLateralities;
}  // namespace
// ===============================================================================
// Class declarations
// ===============================================================================

class MuonPathAnalyzerInChamber : public MuonPathAnalyzer {
public:
  // Constructors and destructor
  MuonPathAnalyzerInChamber(const edm::ParameterSet &pset, edm::ConsumesCollector &iC);
  virtual ~MuonPathAnalyzerInChamber();

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup);
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           std::vector<metaPrimitive> &metaPrimitives) {}
  void run(edm::Event &iEvent, const edm::EventSetup &iEventSetup, MuonPathPtrs &inMpath, MuonPathPtrs &outMPath);

  void finish();

  // Other public methods
  void setBxTolerance(int t) { bxTolerance_ = t; };
  void setMinHits4Fit(int h) { minHits4Fit_ = h; };
  void setChiSquareThreshold(float ch2Thr) { chiSquareThreshold_ = ch2Thr; };
  void setMinimumQuality(MP_QUALITY q) {
    if (minQuality_ >= LOWQGHOST)
      minQuality_ = q;
  };

  int bxTolerance(void) { return bxTolerance_; };
  int minHits4Fit(void) { return minHits4Fit_; };
  MP_QUALITY minQuality(void) { return minQuality_; };

  bool hasPosRF(int wh, int sec) { return wh > 0 || (wh == 0 && sec % 4 > 1); };

  // Public attributes
  DTGeometry const *dtGeo_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;

  //ttrig
  std::map<int, float> ttriginfo_;

  //z
  std::map<int, float> zinfo_;

  //shift
  std::map<int, float> shiftinfo_;

private:
  // Private methods
  //  void analyze(MuonPathPtr &inMPath, std::vector<metaPrimitive> &metaPrimitives);
  void analyze(MuonPathPtr &inMPath, MuonPathPtrs &outMPaths);

  void setCellLayout(MuonPathPtr &mpath);
  void buildLateralities(MuonPathPtr &mpath);
  void setLateralitiesInMP(MuonPathPtr &mpath, TLateralities lat);
  void setWirePosAndTimeInMP(MuonPathPtr &mpath);
  void calculateFitParameters(MuonPathPtr &mpath, TLateralities lat, int present_layer[NLayers]);
  //void calculateFitParameters(MuonPath *mpath, TLateralities lat);

  /* Determina si los valores de 4 primitivas forman una trayectoria
     Los valores tienen que ir dispuestos en el orden de capa:
     0    -> Capa más próxima al centro del detector,
     1, 2 -> Siguientes capas
     3    -> Capa más externa */
  void evaluateQuality(MuonPathPtr &mPath);
  // Private attributes

  /* El máximo de combinaciones de lateralidad para 4 celdas es 16 grupos
     Es feo reservar todo el posible bloque de memoria de golpe, puesto que
     algunas combinaciones no serán válidas, desperdiciando parte de la
     memoria de forma innecesaria, pero la alternativa es complicar el
     código con vectores y reserva dinámica de memoria y, ¡bueno! ¡si hay
     que ir se va, pero ir p'a n'á es tontería! */

  int totalNumValLateralities_;
  std::vector<TLateralities> lateralities_;
  std::vector<LATQ_TYPE> latQuality_;

  /* Posiciones horizontales de cada celda (una por capa), en unidades de
     semilongitud de celda, relativas a la celda de la capa inferior
     (capa 0). Pese a que la celda de la capa 0 siempre está en posición
     0 respecto de sí misma, se incluye en el array para que el código que
     hace el procesamiento sea más homogéneo y sencillo */

  bool debug_;
  double chi2Th_;
  edm::FileInPath z_filename_;
  edm::FileInPath shift_filename_;
  int bxTolerance_;
  MP_QUALITY minQuality_;
  float chiSquareThreshold_;
  short minHits4Fit_;
  int cellLayout_[NLayers];
};

#endif
