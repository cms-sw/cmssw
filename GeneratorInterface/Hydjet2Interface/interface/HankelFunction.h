/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#ifndef HANKELFUNCTION_INCLUDED
#define HANKELFUNCTION_INCLUDED

#include <Rtypes.h>

double HankelK0(double x);
double HankelK1(double x);
// compute modified Hankel function of the second,...,order
double HankelKn(int n, double x);

#endif
