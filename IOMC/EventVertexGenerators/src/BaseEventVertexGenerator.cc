#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"

BaseEventVertexGenerator::BaseEventVertexGenerator(const edm::ParameterSet & p,
                                                   const long& seed) 
    : m_pBaseEventVertexGenerator(p)
{
   m_Engine = new HepJamesRandom(seed) ;
}

BaseEventVertexGenerator::~BaseEventVertexGenerator() 
{
   if ( m_Engine != NULL ) delete m_Engine ;
   m_Engine = 0 ;
}

