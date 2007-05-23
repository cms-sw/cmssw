// $Id: SprMyWriter.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
#ifndef STATPATTERNRECOGNITION_SPRMYWRITER_HH 
#define STATPATTERNRECOGNITION_SPRMYWRITER_HH 1

#ifdef HEPTUPLE
#include "PhysicsTools/StatPatternRecognition/interface/SprTupleWriter.hh"
typedef SprTupleWriter SprMyWriter;
#elif ROOTTUPLE
#include "PhysicsTools/StatPatternRecognition/interface/SprRootWriter.hh"
typedef SprRootWriter SprMyWriter;
#else
#include "PhysicsTools/StatPatternRecognition/interface/SprAsciiWriter.hh"
typedef SprAsciiWriter SprMyWriter;
#endif

#endif // STATPATTERNRECOGNITION_SPRMYWRITER_HH
