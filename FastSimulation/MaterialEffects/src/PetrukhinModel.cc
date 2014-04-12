//////////////////////////////////////////////////////////
// PetrukhinModel.cc Class:
//
// Improvements: Function of Muom Brem using  nuclear screening correction
// Description: Muon bremsstrahlung using the Petrukhin's model in FASTSIM
// Authors: Sandro Fonseca de Souza and Andre Sznajder (UERJ/Brazil)
// Date: 23-Nov-2010
////////////////////////////////////////////////////////////////////

#include <fstream>
#include "TF1.h"
#include "FastSimulation/MaterialEffects/interface/PetrukhinModel.h"
#include <math.h>
using namespace std;

//=====================================================================

///////////////////////////////////////////////////
//Function of Muom Brem using  nuclear-electron screening correction from G4 style
//

 double PetrukhinFunc (double *x, double *p ){
 
   //Function independent variable 
   double nu = x[0]; //fraction of muon's energy transferred to the photon
   
  // Parameters
   double E=p[0]; //Muon Energy (in GeV)
   double A=p[1];// Atomic weight
   double Z=p[2];// Atomic number

 /*

//////////////////////////////////////////////////
//Function of Muom Brem using  nuclear screening correction
//Ref: http://pdg.lbl.gov/2008/AtomicNuclearProperties/adndt.pdf

   //Physical constants
   double B = 182.7;
   double ee = sqrt(2.7181) ; // sqrt(e)
   double ZZ=  pow( Z,-1./3.); // Z^-1/3
   ///////////////////////////////////////////////////////////////////
  double emass = 0.0005109990615;  // electron mass (GeV/c^2)
  double mumass = 0.105658367;//mu mass  (GeV/c^2)

   double re = 2.817940285e-13;// Classical electron radius (Units: cm)
   double alpha = 1./137.03599976; // fine structure constant
   double Dn = 1.54* (pow(A,0.27));
   double constant =  pow((2.0 * Z * emass/mumass * re ),2.0);
   //////////////////////////////////////////////
   
   double delta = (mumass * mumass * nu) /(2.* E * (1.- nu)); 
    
   double Delta_n = TMath::Log(Dn / (1.+ delta *( Dn * ee -2.)/ mumass)); //nuclear screening correction 
    
   double Phi = TMath::Log((B * mumass * ZZ / emass)/ (1.+ delta * ee * B * ZZ  / emass)) - Delta_n;//phi(delta)
   
    //Diff. Cross Section for Muon Brem from a screened nuclear (Equation 16: REF: LBNL-44742)
   double f = alpha * constant *(4./3.-4./3.*nu + nu*nu)*Phi/nu;
*/

//////////////////////////////////////////////////
// Function for Muon Brem Xsec from G4
//////////////////////////////////////////////////
//Physical constants
   double B = 183.;
   double Bl= 1429.;
   double ee = 1.64872 ; // sqrt(e)
   double Z13=  pow( Z,-1./3.); // Z^-1/3
   double Z23=  pow( Z,-2./3.); // Z^-2/3

   //Original values of paper
   double emass = 0.0005109990615;  // electron mass (GeV/c^2)
   double mumass = 0.105658367;     // muon mass  (GeV/c^2)
   // double re = 2.817940285e-13;     // Classical electron radius (Units: cm)
   double alpha =  0.00729735;      // 1./137.03599976; // fine structure constant
   double constant = 1.85736e-30;   // pow( ( emass / mumass * re ) , 2.0);

   double Dn = 1.54*(pow(A,0.27));
   double Dnl= pow(Dn,(1.-1./Z));

   double delta = (mumass * mumass * nu)/(2.* E * (1.- nu)); 
    
    
   double Phi_n = TMath::Log( B * Z13 *( mumass + delta * ( Dnl * ee -2 ))
                  / ( Dnl * ( emass + delta * ee * B * Z13 ) ) );

    
   double Phi_e = TMath::Log( ( Bl * Z23 * mumass )
                  / ( 1.+ delta * mumass / ( emass*emass * ee ) )
                  / ( emass + delta *ee * Bl * Z23 ) );



//Diff. Cross Section for Muon Brem from G4 ( without NA/A factor )
   double f =  16./3. * alpha * constant* Z * ( Z * Phi_n + Phi_e ) * (1./nu) * (1. - nu + 0.75*nu*nu) ;

   return f;
 


 }



