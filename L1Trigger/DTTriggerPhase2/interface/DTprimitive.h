#ifndef L1Trigger_DTTriggerPhase2_DTprimitive_h
#define L1Trigger_DTTriggerPhase2_DTprimitive_h

#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

using namespace cmsdt;

class DTPrimitive {
public:
  DTPrimitive();
  DTPrimitive(DTPrimitive *ptr);
  virtual ~DTPrimitive();

  /* Este método se implementará en la FPGA mediante la comprobación de un
       bit que indique la validez del valor. En el software lo hacemos
       representando como valor no válido, un número negativo cualquiera */
  bool isValidTime(void);
  float wireHorizPos(void);

  void setTimeCorrection(int time) { timeCorrection_ = time; };
  void setTDCTimeStamp(int tstamp) { tdcTimeStamp_ = tstamp; };
  void setOrbit(int orb) { orbit_ = orb; }
  void setPayload(double hitTag, int idx) { this->payLoad_[idx] = hitTag; };
  void setChannelId(int channel) { channelId_ = channel; };
  void setLayerId(int layer) { layerId_ = layer; };
  void setCameraId(int camera) { cameraId_ = camera; };
  void setSuperLayerId(int lay) { superLayerId_ = lay; };
  void setLaterality(LATERAL_CASES lat) { laterality_ = lat; };

  int timeCorrection(void) { return timeCorrection_; };
  int tdcTimeStamp(void) { return tdcTimeStamp_; };
  int orbit(void) { return orbit_; };
  int tdcTimeStampNoOffset(void) { return tdcTimeStamp_ - timeCorrection_; };
  double payLoad(int idx) { return payLoad_[idx]; };
  int channelId(void) { return channelId_; };
  int layerId(void) { return layerId_; };
  int cameraId(void) { return cameraId_; };
  int superLayerId(void) { return superLayerId_; };
  LATERAL_CASES laterality(void) { return laterality_; };

private:
  /* Estos identificadores no tienen nada que ver con el "número de canal"
       que se emplea en el analizador y el resto de componentes. Estos sirven
       para identificar, en el total de la cámara, cada canal individual, y el
       par "cameraId, channelId" (o equivalente) ha de ser único en todo el
       experimento.
       Aquellos sirven para identificar un canal concreto dentro de un
       analizador, recorren los valores de 0 a 9 (tantos como canales tiene
       un analizador) y se repiten entre analizadores */
  int cameraId_;              // Identificador de la cámara
  int superLayerId_;          // Identificador de la super-layer
  int layerId_;               // Identificador de la capa del canal
  int channelId_;             // Identificador del canal en la capa
  LATERAL_CASES laterality_;  // LEFT, RIGHT, NONE

  int timeCorrection_;  // Correccion temporal por electronica, etc...
  int tdcTimeStamp_;    // Tiempo medido por el TDC
  int orbit_;           // Número de órbita
  double payLoad_[PAYLOAD_ENTRIES];
};

#endif
