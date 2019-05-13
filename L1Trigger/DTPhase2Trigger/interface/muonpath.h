#ifndef MUONPATH_H
#define MUONPATH_H
#include <iostream> 
#include "analtypedefs.h"


#include "L1Trigger/DTPhase2Trigger/interface/dtprimitive.h"

class MuonPath {

  public:
    MuonPath(DTPrimitive *ptrPrimitive[4]);
    MuonPath(DTPrimitive *ptrPrimitive[8], short nprim);
    MuonPath(MuonPath *ptr);
    virtual ~MuonPath();

    void setPrimitive(DTPrimitive *ptr, int layer);
    DTPrimitive *getPrimitive(int layer);
    
    short getNPrimitives(void) { return nprimitives; }
    void setNPrimitives(short nprim) { nprimitives = nprim; }

    void setCellHorizontalLayout(int layout[4]);
    void setCellHorizontalLayout(const int *layout);
    const int* getCellHorizontalLayout(void);

    void setCellHorizontalLayout(int layout[4], short sl);
    void setCellHorizontalLayout(const int *layout, short sl);
    const int* getCellHorizontalLayout(short sl);

    int  getBaseChannelId(void);
    void setBaseChannelId(int bch);
    int  getBaseChannelId(short sl);
    void setBaseChannelId(int bch, short sl);

    void setQuality(MP_QUALITY qty);
    MP_QUALITY getQuality(void);
    void setQuality(MP_QUALITY qty, short sl);
    MP_QUALITY getQuality(short sl);

    bool isEqualTo(MuonPath *ptr);
    
    /* El MuonPath debe ser analizado si hay al menos 3 Primitivas válidas */
    bool isAnalyzable(void);
    bool isAnalyzable(short sl);
    /* Indica que hay 4 Primitivas con dato válido */
    bool completeMP(void);
    bool completeMP(short sl);

    void setBxTimeValue(int time);
    int  getBxTimeValue(void);
    void setBxTimeValue(int time, short sl);
    int  getBxTimeValue(short sl);

    int  getBxNumId(void);
    int  getBxNumId(short sl);

    void setLateralComb(LATERAL_CASES latComb[4]);
    void setLateralComb(const LATERAL_CASES *latComb);
    const LATERAL_CASES* getLateralComb(void);
    void setLateralComb(LATERAL_CASES latComb[4],short sl);
    void setLateralComb(const LATERAL_CASES *latComb,short sl);
    const LATERAL_CASES* getLateralComb(short sl);
    void setLateralCombFromPrimitives(void);

    void  setHorizPos(float pos);
    float getHorizPos(void);
    void  setHorizPos(float pos, short sl);
    float getHorizPos(short sl);

    void  setTanPhi(float tanPhi);
    float getTanPhi(void);
    void  setTanPhi(float tanPhi, short sl);
    float getTanPhi(short sl);

    void  setChiSq(float chi);
    float getChiSq(void);
    void  setChiSq(float chi, short sl);
    float getChiSq(short sl);

    void  setXCoorCell(float x, int cell);
    float getXCoorCell(int cell);

    void  setDriftDistance(float dx, int cell);
    float getDriftDistance(int cell);

  private:
    //------------------------------------------------------------------
    //--- Datos del MuonPath
    //------------------------------------------------------------------
    /*
      Primitivas que forman el path. En posición 0 está el dato del canal de la
      capa inferior, y de ahí hacia arriba. El orden es crítico.
     */
    DTPrimitive *prim[8];
    short nprimitives;
    
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
    int cellLayout[3][4];  // SLX=0, SL1=1, SL3=2;
    int baseChannelId[3];  // SLX=0, SL1=1, SL3=2;

    //------------------------------------------------------------------
    //--- Resultados tras cálculos
    //------------------------------------------------------------------
    /* Calidad del path */
    MP_QUALITY quality[3]; // SLX=0, SL1=1, SL3=2;
    
    /* Combinación de lateralidad */
    LATERAL_CASES lateralComb[3][4]; // SLX=0, SL1=1, SL3=2;

    /* Tiempo del BX respecto del BX0 de la órbita en curso */
    int bxTimeValue[3]; // SLX=0, SL1=1, SL3=2;

    /* Número del BX dentro de una órbita */
    int bxNumId[3];  // SLX=0, SL1=1, SL3=2;

    /* Parámetros de celda */
    float xCoorCell[8];         // Posicion horizontal del hit en la cámara
    float xDriftDistance[8];    // Distancia de deriva en la celda (sin signo)

    float tanPhi[3];   // SLX=0, SL1=1, SL3=2;
    float horizPos[3]; // SLX=0, SL1=1, SL3=2;

    float chiSquare[3]; // SLX=0, SL1=1, SL3=2;
};

#endif
