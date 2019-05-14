#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"

#include <cstring>  // Para función "memcpy"
#include "math.h"
#include <iostream>


MuonPath::MuonPath(DTPrimitive *ptrPrimitive[4]) {
  //    std::cout<<"Creando un 'MuonPath'"<<std::endl;
    
    quality[0]       = NOPATH;
    baseChannelId[0] = -1;

    for (int i = 0; i <= 3; i++) {
      if ( (prim[i] = ptrPrimitive[i]) == NULL )
	std::cout<<"Unable to create 'MuonPath'. Null 'Primitive'."<<std::endl;
    }
    
    nprimitives = 4;
    bxTimeValue[0] = -1;
    bxNumId[0]     = -1;
    tanPhi[0]      = 0;
    horizPos[0]    = 0;
    chiSquare[0]   = 0;
    for (int i = 0; i <= 3; i++) {
      lateralComb[0][i] = LEFT;
      setXCoorCell     ( 0, i );
      setDriftDistance ( 0, i );
    }
}

MuonPath::MuonPath(DTPrimitive *ptrPrimitive[8], short nprim) {
  //    std::cout<<"Creando un 'MuonPath'"<<std::endl;
  nprimitives = nprim;
  
  for (int i=1; i<=2; i++){
    quality[i]       = NOPATH;
    baseChannelId[i] = -1;
    bxTimeValue[i] = -1;
    bxNumId[i]     = -1;
    tanPhi[i]      = 0;
    horizPos[i]    = 0;
    chiSquare[i]   = 0;

    for (int l = 0; l <= 3; l++) {
      lateralComb[i][l] = LEFT;
    }
  }
  for (short i = 0; i <= nprim; i++) {
    if ( (prim[i] = ptrPrimitive[i]) == NULL )
      std::cout<<"Unable to create 'MuonPath'. Null 'Primitive'."<<std::endl;

    setXCoorCell     ( 0, i );
    setDriftDistance ( 0, i );
  }
  
}

MuonPath::MuonPath(MuonPath *ptr) {
  //  std::cout<<"Clonando un 'MuonPath'"<<std::endl;

    setQuality             ( ptr->getQuality()              );
    setBaseChannelId       ( ptr->getBaseChannelId()        );
    setCellHorizontalLayout( ptr->getCellHorizontalLayout() );
    setNPrimitives         ( ptr->getNPrimitives()          );

    for (int i = 0; i < ptr->getNPrimitives(); i++)
	setPrimitive( new DTPrimitive(ptr->getPrimitive(i)), i );

    setLateralComb ( ptr->getLateralComb() );
    setBxTimeValue ( ptr->getBxTimeValue() );
    setTanPhi      ( ptr->getTanPhi()      );
    setHorizPos    ( ptr->getHorizPos()    );
    setChiSq       ( ptr->getChiSq()       );

    for (int i = 0; i <  ptr->getNPrimitives(); i++) {
	setXCoorCell     ( ptr->getXCoorCell(i), i     );
	setDriftDistance ( ptr->getDriftDistance(i), i );
    }
}

MuonPath::~MuonPath() {
  //std::cout<<"Destruyendo un 'MuonPath'"<<std::endl;
  
  for (int i = 0; i < nprimitives; i++)
    if (prim[i] != NULL) delete prim[i];
}

//------------------------------------------------------------------
//--- Public
//------------------------------------------------------------------
/**
 * Añade una 'DTPrimitive'
 */
void MuonPath::setPrimitive(DTPrimitive *ptr, int layer){
    if (ptr == NULL) std::cout<<"NULL 'Primitive'."<<std::endl;
    prim[layer] = ptr;
}

DTPrimitive *MuonPath::getPrimitive(int layer) { return prim[layer]; }

void MuonPath::setCellHorizontalLayout(int layout[4]){
  //  std::cout << "setCellHorizontalLayout" << std::endl;
  for (int i=0; i<=3; i++)
    cellLayout[0][i] = layout[i];
}

void MuonPath::setCellHorizontalLayout(const int *layout){
  //  std::cout << "setCellHorizontalLayout2" << std::endl;
  for (int i=0; i<=3; i++)
    cellLayout[0][i] = layout[i];
}

void MuonPath::setCellHorizontalLayout(int layout[4], short sl){
  int idx=0;
  if      (sl==0) idx=1; 
  else if (sl==2) idx=2; 
  
  for (int i=0; i<=3; i++)
    cellLayout[idx][i] = layout[i];
}

void MuonPath::setCellHorizontalLayout(const int *layout, short sl){
  int idx=0;
  if      (sl==0) idx=1; 
  else if (sl==2) idx=2; 
  
  for (int i=0; i<=3; i++)
    cellLayout[idx][i] = layout[i];
}

const int* MuonPath::getCellHorizontalLayout(void) { return (cellLayout[0]); }
const int* MuonPath::getCellHorizontalLayout(short sl) { 
  if (sl==0) return (cellLayout[1]); 
  if (sl==2) return (cellLayout[2]); 

  std::cerr<<"Please provide a valid SL Id"<<std::endl;    
  return  getCellHorizontalLayout();
}

/**
 * Devuelve el identificador del canal que está en la base del BTI (en la capa
 * inferior del MuonPath)
 */
int MuonPath::getBaseChannelId(void) { return baseChannelId[0]; }
void MuonPath::setBaseChannelId(int bch) { baseChannelId[0] = bch; }

int MuonPath::getBaseChannelId(short sl) { 
  if (sl==0) return baseChannelId[1]; 
  if (sl==2) return baseChannelId[2]; 
  return getBaseChannelId();
}
void MuonPath::setBaseChannelId(int bch, short sl) { 
  if (sl==0) baseChannelId[1] = bch; 
  if (sl==2) baseChannelId[2] = bch; 
  setBaseChannelId(bch);
}

void MuonPath::setQuality(MP_QUALITY qty) { quality[0] = qty; }
MP_QUALITY MuonPath::getQuality(void) { return quality[0]; }

void MuonPath::setQuality(MP_QUALITY qty, short sl) { 
  // SL values are 0, 1 or 2
  if      (sl==0) quality[1] = qty;
  else if (sl==2) quality[2] = qty;
  else {
    std::cerr<<"Please provide a valid SL Id"<<std::endl;
    setQuality(qty);
  }
}

MP_QUALITY MuonPath::getQuality(short sl) { 
  if      (sl==0) return quality[1];
  else if (sl==2) return quality[2];
  else {
    std::cerr<<"Please provide a valid SL Id"<<std::endl;    
    return getQuality();
  }
}

/**
 * Devuelve TRUE si el objeto apuntado por "ptr" tiene los mismos valores de
 * campos que este objeto.
 * El criterio de "los mismos valores" puede ser adaptado con algunos
 * parámetros. De momento solo los valores significativos se comparan con
 * igualdad estricta.
 */
bool MuonPath::isEqualTo(MuonPath *ptr) {
    /*
     * Comparamos las primitivas (TDC TimeStamp y identificador de canal).
     * También se incluye la combinación de lateralidad.
     * 
     * Al criterio de combinación de lateralidad se añade una mejora para 
     * que sólo se consideren diferentes aquellos casos en los que, si bien hay
     * una lateralidad diferente, esta sólo se compara en caso que el valor
     * de TDC sea válido. Esto es así porque si el valor de TDC no es válido
     * (valor "dummy") no tiene sentido considerarlos diferentes porque 
     * haya cambiado la lateralidad.
     */
    for (int i = 0; i <= 7; i++) {
	/* Ambas válidas: comprobamos diferencias entre 'hits' */
	if (this->getPrimitive(i)->isValidTime() && ptr->getPrimitive(i)->isValidTime() )
	    {
		if (ptr->getPrimitive(i)->getSuperLayerId() != this->getPrimitive(i)->getSuperLayerId() ||
        
		    ptr->getPrimitive(i)->getChannelId() != this->getPrimitive(i)->getChannelId()    ||

		    ptr->getPrimitive(i)->getTDCTime() != this->getPrimitive(i)->getTDCTime()      ||

		    /* Esta condición no debería cumplirse nunca ya que en un 'Segment'
		     * no tendrían que aparecer jamás 'hits' de distintar orbitas.
		     * En ese caso habría un error en el mixer.
		     */
		    ptr->getPrimitive(i)->getOrbit() !=  this->getPrimitive(i)->getOrbit() || (ptr->getLateralComb())[i] != (this->getLateralComb())[i]
		    ) 
		    return false;
	    }
	else {
	    /* Ambas inválidas: pasamos al siguiente hit */
	    if (!this->getPrimitive(i)->isValidTime() && 
		!ptr->getPrimitive(i)->isValidTime()) continue;
	    /* Una válida y la otra no: son diferentes por definición */
	    else
		return false;
	}
    }
    /*
     * Si se parte de los mismos valores de TDC en las mismas celdas, y se somete
     * al mismo análisis, han de arrojar idénticos resultados.
     *
     * Esto es redundante pero lo añado.
     * Para ser iguales tienen que tener la misma celda base.
     */
  
    /*
     * La condición anterior ha demostrado dejar segmentos idénticos en la
     * salida, cuando todos los 'hits' son iguales, excepto el de la celda base
     * en aquellos casos en los que la celda base es inválida.
     * Así pues, comento esta condición porque no ayuda y falsea la salida.
     */
    // if ( ptr->getBaseChannelId() != this->getBaseChannelId() ) return false;

    return true;
}

/** Este método indica si, al menos, 3 primitivas tienen valor válido, por lo
 * que es el MuonPath es un candidato a ser analizado (sólo 2 primitivas
 * válidas no permiten aplicar el método de 'mean timer').
 * La implementación se realiza de forma booleana, para asemejar la posible
 * implementación con lógica combinacional que tendría en la FPGA.
 */
bool MuonPath::isAnalyzable(void) {
    return (
	    (prim[0]->isValidTime() && prim[1]->isValidTime() &&
	     prim[2]->isValidTime()) ||
	    (prim[0]->isValidTime() && prim[1]->isValidTime() &&
	     prim[3]->isValidTime()) ||
	    (prim[0]->isValidTime() && prim[2]->isValidTime() &&
	     prim[3]->isValidTime()) ||
	    (prim[1]->isValidTime() && prim[2]->isValidTime() &&
	     prim[3]->isValidTime())
	    );
}

bool MuonPath::isAnalyzable(short sl) {
  if (sl==0) {
    return (
	    (prim[0]->isValidTime() && prim[1]->isValidTime() &&  prim[2]->isValidTime()) ||
	    (prim[0]->isValidTime() && prim[1]->isValidTime() &&  prim[3]->isValidTime()) ||
	    (prim[0]->isValidTime() && prim[2]->isValidTime() &&  prim[3]->isValidTime()) ||
	    (prim[1]->isValidTime() && prim[2]->isValidTime() &&  prim[3]->isValidTime()) 
	    );
  }
  else if (sl==2){
    return (
	    (prim[4]->isValidTime() && prim[5]->isValidTime() &&  prim[6]->isValidTime()) ||
	    (prim[4]->isValidTime() && prim[5]->isValidTime() &&  prim[7]->isValidTime()) ||
	    (prim[4]->isValidTime() && prim[6]->isValidTime() &&  prim[7]->isValidTime()) ||
	    (prim[5]->isValidTime() && prim[6]->isValidTime() &&  prim[7]->isValidTime()) 
	    );
  }
  else 
    return isAnalyzable();
}
/**
 * Informa si las 4 DTPrimitives que formas el MuonPath tiene un 'TimeStamp'
 * válido.
 *
 */
bool MuonPath::completeMP(void) {
    return (prim[0]->isValidTime() && prim[1]->isValidTime() &&
	    prim[2]->isValidTime() && prim[3]->isValidTime());
}

bool MuonPath::completeMP(short sl) {
  if (sl==0) 
    return (prim[0]->isValidTime() && prim[1]->isValidTime() &&
	    prim[2]->isValidTime() && prim[3]->isValidTime());
  if (sl==2) 
    return (prim[4]->isValidTime() && prim[5]->isValidTime() &&
	    prim[6]->isValidTime() && prim[7]->isValidTime());
  
  return completeMP();
}

void MuonPath::setBxTimeValue(int time) {
    bxTimeValue[0] = time;

    float auxBxId = float(time) / LHC_CLK_FREQ;
    bxNumId[0] = int(auxBxId);
    if ( (auxBxId - int(auxBxId)) >= 0.5 ) bxNumId[0] = int(bxNumId[0] + 1);
}

void MuonPath::setBxTimeValue(int time, short sl) {
  int idx=0; 
  if (sl==0) idx=1;
  if (sl==2) idx=2;
  
  bxTimeValue[idx] = time;
  float auxBxId = float(time) / LHC_CLK_FREQ;
  bxNumId[idx] = int(auxBxId);
  if ( (auxBxId - int(auxBxId)) >= 0.5 ) bxNumId[idx] = int(bxNumId[idx] + 1);
}

int MuonPath::getBxTimeValue(void) { return bxTimeValue[0]; }
int MuonPath::getBxNumId(void) { return bxNumId[0]; }

int MuonPath::getBxTimeValue(short sl) { 
  if (sl==0) return bxTimeValue[1]; 
  if (sl==2) return bxTimeValue[2]; 
  return getBxTimeValue();
}
int MuonPath::getBxNumId(short sl) { 
  if (sl==0) return bxNumId[1]; 
  if (sl==2) return bxNumId[2]; 
  return getBxNumId();
}

/* Este método será invocado por el analizador para rellenar la información
   sobre la combinación de lateralidad que ha dado lugar a una trayectoria
   válida. Antes de ese momento, no tiene utilidad alguna */

void MuonPath::setLateralCombFromPrimitives(void) {
  for (int i=0; i<nprimitives; i++){
    if (!this->getPrimitive(i)->isValidTime()) continue;
    
    if (i<4)  lateralComb[1][i]             = this->getPrimitive(i)->getLaterality();
    if (i>=4) lateralComb[2][nprimitives-i] = this->getPrimitive(i)->getLaterality();
  }
}

void MuonPath::setLateralComb(LATERAL_CASES latComb[4]) {
  for (int i=0; i<=3; i++)
    lateralComb[0][i] = latComb[i];
}

void MuonPath::setLateralComb(const LATERAL_CASES *latComb) {
  for (int i=0; i<=3; i++)
    lateralComb[0][i] = latComb[i];
}

const LATERAL_CASES* MuonPath::getLateralComb(void) { 
    return (lateralComb[0]); 
}

void MuonPath::setLateralComb(LATERAL_CASES latComb[4],short sl) {
  if (sl==0)      memcpy(lateralComb[1], latComb, 4 * sizeof(LATERAL_CASES));
  else if (sl==2) memcpy(lateralComb[2], latComb, 4 * sizeof(LATERAL_CASES));
  else            memcpy(lateralComb[0], latComb, 4 * sizeof(LATERAL_CASES)); 
    
}

void MuonPath::setLateralComb(const LATERAL_CASES *latComb, short sl) {
  if (sl==0)      memcpy(lateralComb[1], latComb, 4 * sizeof(LATERAL_CASES));
  else if (sl==2) memcpy(lateralComb[2], latComb, 4 * sizeof(LATERAL_CASES));
  else            memcpy(lateralComb[0], latComb, 4 * sizeof(LATERAL_CASES)); 
}

const LATERAL_CASES* MuonPath::getLateralComb(short sl) { 
  if (sl==0) return (lateralComb[1]); 
  if (sl==2) return (lateralComb[2]); 
  return getLateralComb();
}

void  MuonPath::setHorizPos(float pos) { horizPos[0] = pos; }
float MuonPath::getHorizPos(void) { return horizPos[0]; }

void  MuonPath::setHorizPos(float pos, short sl) { 
  if      (sl==0) horizPos[1] = pos; 
  else if (sl==2) horizPos[2] = pos; 
  else            horizPos[0] = pos; 
}
float MuonPath::getHorizPos(short sl) { 
  if (sl==0) return horizPos[1];
  if (sl==2) return horizPos[2];
  return getHorizPos();
}

void  MuonPath::setTanPhi(float tanPhi) { this->tanPhi[0] = tanPhi; }
float MuonPath::getTanPhi(void) { return tanPhi[0]; }

void  MuonPath::setTanPhi(float tanPhi, short sl) { 
  if      (sl==0) this->tanPhi[1] = tanPhi;
  else if (sl==2) this->tanPhi[2] = tanPhi;
  else            this->tanPhi[0] = tanPhi;
}
float MuonPath::getTanPhi(short sl) { 
  if (sl==0) return tanPhi[1];
  if (sl==2) return tanPhi[2];
  return getTanPhi();
}

void  MuonPath::setChiSq(float chi) { chiSquare[0] = chi;  }
float MuonPath::getChiSq(void)      { return chiSquare[0]; }

void  MuonPath::setChiSq(float chi, short sl) {
  if      (sl==0) chiSquare[1] = chi;
  else if (sl==2) chiSquare[2] = chi;
  else            chiSquare[0] = chi;
}
float MuonPath::getChiSq(short sl)      { 
  if (sl==0) return chiSquare[1];
  if (sl==2) return chiSquare[2];
  return getChiSq();
}

void  MuonPath::setXCoorCell(float x, int cell) { xCoorCell[cell] = x;    }
float MuonPath::getXCoorCell(int cell)          { return xCoorCell[cell]; }

void  MuonPath::setDriftDistance(float dx, int cell) {
    xDriftDistance[cell] = dx;
}
float MuonPath::getDriftDistance(int cell) { return xDriftDistance[cell]; }
