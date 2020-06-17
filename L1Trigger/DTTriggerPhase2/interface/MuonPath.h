#ifndef L1Trigger_DTTriggerPhase2_MuonPath_h
#define L1Trigger_DTTriggerPhase2_MuonPath_h
#include <iostream>
#include <memory>

#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"

class MuonPath {
public:
  MuonPath();
  MuonPath(DTPrimitivePtrs &ptrPrimitive, int prup = 0, int prdw = 0);
  MuonPath(std::shared_ptr<MuonPath> &ptr);
  virtual ~MuonPath(){};

  // setter methods
  void setPrimitive(DTPrimitivePtr &ptr, int layer);
  void setNPrimitives(short nprim) { nprimitives_ = nprim; };
  void setNPrimitivesUp(short nprim) { nprimitivesUp_ = nprim; };
  void setNPrimitivesDown(short nprim) { nprimitivesDown_ = nprim; };
  void setCellHorizontalLayout(int layout[4]);
  void setCellHorizontalLayout(const int *layout);
  void setBaseChannelId(int bch) { baseChannelId_ = bch; };
  void setQuality(MP_QUALITY qty) { quality_ = qty; };
  void setBxTimeValue(int time);
  void setLateralComb(LATERAL_CASES latComb[4]);
  void setLateralComb(const LATERAL_CASES *latComb);
  void setLateralCombFromPrimitives(void);

  void setHorizPos(float pos) { horizPos_ = pos; };
  void setTanPhi(float tanPhi) { tanPhi_ = tanPhi; };
  void setChiSquare(float chi) { chiSquare_ = chi; };
  void setPhi(float phi) { phi_ = phi; };
  void setPhiB(float phib) { phiB_ = phib; };
  void setXCoorCell(float x, int cell) { xCoorCell_[cell] = x; };
  void setDriftDistance(float dx, int cell) { xDriftDistance_[cell] = dx; };
  void setXWirePos(float x, int cell) { xWirePos_[cell] = x; };
  void setZWirePos(float z, int cell) { zWirePos_[cell] = z; };
  void setTWireTDC(float t, int cell) { tWireTDC_[cell] = t; };
  void setRawId(uint32_t id) { rawId_ = id; };

  // getter methods
  DTPrimitivePtr primitive(int layer) { return prim_[layer]; };
  short nprimitives(void) { return nprimitives_; };
  short nprimitivesDown(void) { return nprimitivesDown_; };
  short nprimitivesUp(void) { return nprimitivesUp_; };
  const int *cellLayout(void) { return cellLayout_; };
  int baseChannelId(void) { return baseChannelId_; };
  MP_QUALITY quality(void) { return quality_; };
  int bxTimeValue(void) { return bxTimeValue_; };
  int bxNumId(void) { return bxNumId_; };
  float tanPhi(void) { return tanPhi_; };
  const LATERAL_CASES *lateralComb(void) { return (lateralComb_); };
  float horizPos(void) { return horizPos_; };
  float chiSquare(void) { return chiSquare_; };
  float phi(void) { return phi_; };
  float phiB(void) { return phiB_; };
  float xCoorCell(int cell) { return xCoorCell_[cell]; };
  float xDriftDistance(int cell) { return xDriftDistance_[cell]; };
  float xWirePos(int cell) { return xWirePos_[cell]; };
  float zWirePos(int cell) { return zWirePos_[cell]; };
  float tWireTDC(int cell) { return tWireTDC_[cell]; };
  uint32_t rawId() { return rawId_; };

  // Other methods
  bool isEqualTo(MuonPath *ptr);
  bool isAnalyzable(void);
  bool completeMP(void);

private:
  //------------------------------------------------------------------
  //--- Datos del MuonPath
  //------------------------------------------------------------------
  /*
      Primitivas que forman el path. En posición 0 está el dato del canal de la
      capa inferior, y de ahí hacia arriba. El orden es crítico.
  */
  DTPrimitivePtrs prim_;  //ENSURE that there are no more than 4-8 prims
  short nprimitives_;
  short nprimitivesUp_;
  short nprimitivesDown_;

  /* Posiciones horizontales de cada celda (una por capa), en unidades de
       semilongitud de celda, relativas a la celda de la capa inferior
       (capa 0). Pese a que la celda de la capa 0 siempre está en posición
       0 respecto de sí misma, se incluye en el array para que el código que
       hace el procesamiento sea más homogéneo y sencillo.
       Estos parámetros se habían definido, en la versión muy preliminar del
       código, en el 'PathAnalyzer'. Ahora se trasladan al 'MuonPath' para
       que el 'PathAnalyzer' sea un único componente (y no uno por posible
       ruta, como en la versión original) y se puede disponer en arquitectura
       tipo pipe-line */
  int cellLayout_[4];
  int baseChannelId_;

  //------------------------------------------------------------------
  //--- Resultados tras cálculos
  //------------------------------------------------------------------
  /* Calidad del path */
  MP_QUALITY quality_;

  /* Combinación de lateralidad */
  LATERAL_CASES lateralComb_[4];

  /* Tiempo del BX respecto del BX0 de la órbita en curso */
  int bxTimeValue_;

  /* Número del BX dentro de una órbita */
  int bxNumId_;

  /* Parámetros de celda */
  float xCoorCell_[8];       // Posicion horizontal del hit en la cámara
  float xDriftDistance_[8];  // Distancia de deriva en la celda (sin signo)
  float xWirePos_[8];
  float zWirePos_[8];
  float tWireTDC_[8];

  float tanPhi_;
  float horizPos_;
  float chiSquare_;
  float phi_;
  float phiB_;

  uint32_t rawId_;
};

typedef std::vector<MuonPath> MuonPaths;
typedef std::shared_ptr<MuonPath> MuonPathPtr;
typedef std::vector<MuonPathPtr> MuonPathPtrs;

#endif
