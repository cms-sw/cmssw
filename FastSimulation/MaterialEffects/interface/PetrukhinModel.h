//////////////////////////////////////////////////////////
// PETRUKHINMODEL_h 
//
// Improvements: Function of Muom Brem using  nuclear screening correction
// Description: Muon bremsstrahlung using the Petrukhin's model in FASTSIM
// Authors: Sandro Fonseca de Souza and Andre Sznajder (UERJ/Brazil)
// Date: 23-Nov-2010
////////////////////////////////////////////////////////////////////

#ifndef PETRUKHINMODEL_h
#define PETRUKHINMODEL_h

#include "TMath.h"
#include "TF1.h"


double PetrukhinFunc (double *x, double *p );


#endif
