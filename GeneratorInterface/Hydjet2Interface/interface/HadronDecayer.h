/*                                                                            
                                                                           
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#ifndef HADRONDECAYER_INCLUDED
#define HADRONDECAYER_INCLUDED

#include "DatabasePDG.h"
#include "Particle.h"

double GetDecayTime(const Particle &p, double weakDecayLimit);
void Decay(List_t &output, Particle &p, ParticleAllocator &allocator, DatabasePDG* database);

#endif
