/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                
*/

#ifndef UKUTILITY_INCLUDED
#define UKUTILITY_INCLUDED
 
class TLorentzVector;
class TVector3;
class TH1F;

class Particle;

void IsotropicR3(double r, double *pX, double *pY, double *pZ);
void IsotropicR3(double r, TVector3 &pos);
void MomAntiMom(TLorentzVector &mom, double mass, TLorentzVector &antiMom,
		double antiMass, double initialMass);

extern const double GeV;
extern const double MeV;
extern const double fermi;
extern const double mbarn;
extern const double hbarc; 
extern const double w;
extern const double hbarc_squared; 

#endif
