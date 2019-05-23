#include "L1Trigger/DTPhase2Trigger/interface/MPRedundantFilter.h"

using namespace edm;
using namespace std;



// ============================================================================
// Constructors and destructor
// ============================================================================
MPRedundantFilter::MPRedundantFilter(const ParameterSet& pset):
  MPFilter(pset),
  MaxBufferSize(8)
{
  // Obtention of parameters
  debug         = pset.getUntrackedParameter<Bool_t>("debug");
  if (debug) cout <<"MPRedundantFilter: constructor" << endl;
}


MPRedundantFilter::~MPRedundantFilter() {
  if (debug) cout <<"MPRedundantFilter: destructor" << endl;
}



// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPRedundantFilter::initialise(const edm::EventSetup& iEventSetup) {
  if(debug) cout << "MPRedundantFilter::initialiase" << endl;
  buffer.clear();
  
  //  for (unsigned int i=0; i<MaxBufferSize; i++) buffer.push_back(new MuonPath());
}


void MPRedundantFilter::run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, 
			 std::vector<MuonPath*> &inMPaths, 
			 std::vector<MuonPath*> &outMPaths) 
{
  buffer.clear();
  for(auto muonpath = inMPaths.begin();muonpath!=inMPaths.end();++muonpath) {
    filter(*muonpath,outMPaths);
  }
  if (debug) cout <<"MPRedundantFilter: done" << endl;
  
  buffer.clear();
}

void MPRedundantFilter::filter(MuonPath *mPath, std::vector<MuonPath*> &outMPaths) {
  
  /*
    Esta línea se incluye para evitar que, tras un 'stop', que fuerza la
    liberación del mutex de la fifo de entrada, devuelva un puntero nulo, lo que
    a su vez, induce un error en la ejecución al intentar acceder a cualquiera
    de los métodos de la clase 'DTPrimitive'
  */
  if (mPath == NULL) return;

  // En caso que no esté en el buffer, será enviado al exterior.
  if ( !isInBuffer(mPath) ) {
    // Borramos el primer elemento que se insertó (el más antiguo).
    if (buffer.size() == MaxBufferSize) buffer.pop_front();
    // Insertamos el ultimo "path" como nuevo elemento.
    buffer.push_back(mPath);
    
    // Enviamos una copia
    MuonPath *mpAux = new MuonPath(mPath);
    outMPaths.push_back( mpAux );
  }
}

bool MPRedundantFilter::isInBuffer(MuonPath* mPath) {
  bool ans = false;

  if ( !buffer.empty() ){
    for (unsigned int i = 0; i < buffer.size(); i++)
      /*
       * Recorremos el buffer is si detectamos un elemento igual al de prueba
       * salimos, indicando el resultado.
       * No se siguen procesando los restantes elementos.
       */
      if ( mPath->isEqualTo( (MuonPath*) buffer.at(i) )) {
        ans = true;
        break;
      }
  }
  return ans;
}
