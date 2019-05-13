#ifndef DTPRIMITIVE_H
#define DTPRIMITIVE_H
/*
 * Bit mask to get BX counter part from TDC Time-Stamp
 * It's assumend that TDC value is 32 bits in length, and comprises two parts:
 *    Coarse counter -> bits [31:5]
 *    Fine part      -> bits [4:0]
 *
 * Coarse part is reseted periodically with BX Reset signal, so it represents
 * BX counter value.
 */
#define TDC_TIME_COARSE_MASK 0xFFFFFFE0
#define TDC_TIME_FINE_MASK   0x1F

#include "constants.h"
#include "analtypedefs.h"

class DTPrimitive {

  public:
    DTPrimitive();
    DTPrimitive(DTPrimitive *ptr);
    virtual ~DTPrimitive();

    /* Este método se implementará en la FPGA mediante la comprobación de un
       bit que indique la validez del valor. En el software lo hacemos
       representando como valor no válido, un número negativo cualquiera */
    bool isValidTime(void);

    /* Correcciones temporales debido a tiempo de vuelo y demora electrónica */
    int  getTimeOffsetCorrection(void);
    void setTimeOffsetCorrection(int time);

    int  getTDCTime(void);
    void setTDCTime(int tstamp);
    int  getOrbit(void);
    void setOrbit(int orb);
    int  getTDCTimeNoOffset(void);

    int getTDCCoarsePart(void);
    int getTDCFinePart(void);

    double getPayload(int idx);
    void   setPayload(double hitTag, int idx);

    int  getChannelId(void);
    void setChannelId(int channel);
    int  getLayerId(void);
    void setLayerId(int layer);
    int  getCameraId(void);
    void setCameraId(int camera);
    int  getSuperLayerId(void);
    void setSuperLayerId(int lay);
    
    LATERAL_CASES  getLaterality(void){ return laterality; };
    void setLaterality(LATERAL_CASES lat) { laterality = lat; };
    float getWireHorizPos(void);

  private:
    /* Estos identificadores no tienen nada que ver con el "número de canal"
       que se emplea en el analizador y el resto de componentes. Estos sirven
       para identificar, en el total de la cámara, cada canal individual, y el
       par "cameraId, channelId" (o equivalente) ha de ser único en todo el
       experimento.
       Aquellos sirven para identificar un canal concreto dentro de un
       analizador, recorren los valores de 0 a 9 (tantos como canales tiene
       un analizador) y se repiten entre analizadores */
    int cameraId;     // Identificador de la cámara
    int superLayerId; // Identificador de la super-layer
    int layerId;      // Identificador de la capa del canal
    int channelId;    // Identificador del canal en la capa
    LATERAL_CASES laterality;   // LEFT, RIGHT, NONE

    int timeCorrection;   // Correccion temporal por electronica, etc...
    int tdcTimeStamp;     // Tiempo medido por el TDC
    int orbit;            // Número de órbita
    double hitTag[PAYLOAD_ENTRIES];
};

#endif
