#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"

#include <cstring>  // Para función "memcpy"
#include "math.h"
#include <iostream>

MuonPath::MuonPath() {
  //    std::cout<<"Creando un 'MuonPath'"<<std::endl;
    
    quality       = NOPATH;
    baseChannelId = -1;

    for (int i = 0; i <= 3; i++) {
      prim[i] = new DTPrimitive();     
    }
    
    nprimitives = 4;
    bxTimeValue = -1;
    bxNumId     = -1;
    tanPhi      = 0;
    horizPos    = 0;
    chiSquare   = 0;
    for (int i = 0; i <= 3; i++) {
      lateralComb[i] = LEFT;
      setXCoorCell     ( 0, i );
      setDriftDistance ( 0, i );
    }
}

MuonPath::MuonPath(DTPrimitive *ptrPrimitive[4]) {
  //    std::cout<<"Creando un 'MuonPath'"<<std::endl;
  
  quality       = NOPATH;
  baseChannelId = -1;

  for (int i = 0; i <= 3; i++) {
    if ( (prim[i] = ptrPrimitive[i]) == NULL )
      std::cout<<"Unable to create 'MuonPath'. Null 'Primitive'."<<std::endl;
  }
  
  nprimitives = 4;
  //Dummy values
  nprimitivesUp = 0;
  nprimitivesDown = 0;
  bxTimeValue = -1;
  bxNumId     = -1;
  tanPhi      = 0;
  horizPos    = 0;
  chiSquare   = 0;
  Phi         = 0;
  PhiB        = 0;
  rawId          = 0;
  for (int i = 0; i <= 3; i++) {
    lateralComb[i] = LEFT;
    setXCoorCell     ( 0, i );
    setDriftDistance ( 0, i );
    setXWirePos      ( 0, i ); 
    setZWirePos      ( 0, i ); 
    settWireTDC      ( 0, i ); 
  }
}

MuonPath::MuonPath(DTPrimitive *ptrPrimitive[8], int nprimUp, int nprimDown) {
  //    std::cout<<"Creando un 'MuonPath'"<<std::endl;
  nprimitives     = 8; //Instead of nprimUp + nprimDown;
  nprimitivesUp   = nprimUp;
  nprimitivesDown = nprimDown;
  rawId           = 0;
  quality         = NOPATH;
  baseChannelId   = -1;
  bxTimeValue     = -1;
  bxNumId         = -1;
  tanPhi          = 0;
  horizPos        = 0;
  chiSquare       = 0;
  Phi             = 0;
  PhiB            = 0;
  for (int l = 0; l <= 3; l++) {
    lateralComb[l] = LEFT;
  }
  
  for (short i = 0; i < nprimitives; i++) {
    if ( (prim[i] = ptrPrimitive[i]) == NULL ){
      std::cout<<"Unable to create 'MuonPath'. Null 'Primitive'."<<std::endl;
    }
    setXCoorCell     ( 0, i );
    setDriftDistance ( 0, i );
    setXWirePos      ( 0, i ); 
    setZWirePos      ( 0, i ); 
    settWireTDC      ( 0, i ); 
  }
  
}

MuonPath::MuonPath(MuonPath *ptr) {
  //  std::cout<<"Clonando un 'MuonPath'"<<std::endl;
  setRawId		 ( ptr->getRawId()		  );
  setPhi		 ( ptr->getPhi()		  );
  setPhiB		 ( ptr->getPhiB()		  );
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
    setXWirePos      ( ptr->getXWirePos(i), i ); 
    setZWirePos      ( ptr->getZWirePos(i), i ); 
    settWireTDC      ( ptr->gettWireTDC(i), i ); 

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
    cellLayout[i] = layout[i];
}

void MuonPath::setCellHorizontalLayout(const int *layout){
  //  std::cout << "setCellHorizontalLayout2" << std::endl;
  for (int i=0; i<=3; i++)
    cellLayout[i] = layout[i];
}


const int* MuonPath::getCellHorizontalLayout(void) { return (cellLayout); }

/**
 * Devuelve el identificador del canal que está en la base del BTI (en la capa
 * inferior del MuonPath)
 */
int MuonPath::getBaseChannelId(void) { return baseChannelId; }
void MuonPath::setBaseChannelId(int bch) { baseChannelId = bch; }


void MuonPath::setQuality(MP_QUALITY qty) { quality = qty; }
MP_QUALITY MuonPath::getQuality(void) { return quality; }


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
  for (int i = 0; i < ptr->getNPrimitives(); i++) {
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

  short countValidHits=0;
  for (int i = 0; i < this->getNPrimitives(); i++) {
    if (this->getPrimitive(i)->isValidTime()) countValidHits++;    
  }
  
  if (countValidHits >= 3) return true; 
  return false;
}
/* return (
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
*/
/**
 * Informa si las 4 DTPrimitives que formas el MuonPath tiene un 'TimeStamp'
 * válido.
 *
 */
bool MuonPath::completeMP(void) {
    return (prim[0]->isValidTime() && prim[1]->isValidTime() &&
	    prim[2]->isValidTime() && prim[3]->isValidTime());
}


void MuonPath::setBxTimeValue(int time) {
    bxTimeValue = time;

    float auxBxId = float(time) / LHC_CLK_FREQ;
    bxNumId = int(auxBxId);
    if ( (auxBxId - int(auxBxId)) >= 0.5 ) bxNumId = int(bxNumId + 1);
}


int MuonPath::getBxTimeValue(void) { return bxTimeValue; }
int MuonPath::getBxNumId(void) { return bxNumId; }


/* Este método será invocado por el analizador para rellenar la información
   sobre la combinación de lateralidad que ha dado lugar a una trayectoria
   válida. Antes de ese momento, no tiene utilidad alguna */

void MuonPath::setLateralCombFromPrimitives(void) {
  for (int i=0; i<nprimitives; i++){
    if (!this->getPrimitive(i)->isValidTime()) continue;
    lateralComb[i] = this->getPrimitive(i)->getLaterality();
  }
}

void MuonPath::setLateralComb(LATERAL_CASES latComb[4]) {
  for (int i=0; i<=3; i++)
    lateralComb[i] = latComb[i];
}

void MuonPath::setLateralComb(const LATERAL_CASES *latComb) {
  for (int i=0; i<=3; i++)
    lateralComb[i] = latComb[i];
}

const LATERAL_CASES* MuonPath::getLateralComb(void) { 
    return (lateralComb); 
}


void  MuonPath::setHorizPos(float pos) { horizPos = pos; }
float MuonPath::getHorizPos(void) { return horizPos; }


void  MuonPath::setTanPhi(float tanPhi) { this->tanPhi = tanPhi; }
float MuonPath::getTanPhi(void) { return tanPhi; }


void  MuonPath::setChiSq(float chi) { chiSquare = chi;  }
float MuonPath::getChiSq(void)      { return chiSquare; }

void  MuonPath::setPhi(float phi) { Phi = phi;  }
float MuonPath::getPhi(void)      { return Phi; }


void  MuonPath::setPhiB(float phib) { PhiB = phib;  }
float MuonPath::getPhiB(void)      { return PhiB; }


void  MuonPath::setXCoorCell(float x, int cell) { xCoorCell[cell] = x;    }
float MuonPath::getXCoorCell(int cell)          { return xCoorCell[cell]; }

void  MuonPath::setXWirePos(float x, int cell) { xWirePos[cell] = x;    }
float MuonPath::getXWirePos(int cell)          { return xWirePos[cell]; }
void  MuonPath::setZWirePos(float z, int cell) { zWirePos[cell] = z;    }
float MuonPath::getZWirePos(int cell)          { return zWirePos[cell]; }
void  MuonPath::settWireTDC(float t, int cell) { tWireTDC[cell] = t;    }
float MuonPath::gettWireTDC(int cell)          { return tWireTDC[cell]; }

void  MuonPath::setDriftDistance(float dx, int cell) {
    xDriftDistance[cell] = dx;
}
float MuonPath::getDriftDistance(int cell) { return xDriftDistance[cell]; }
