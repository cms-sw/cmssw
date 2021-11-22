// This is file contains the code to generate a dark sector shower in a strongly coupled, quasi-conformal hidden valley, often 
// referred to as "soft unclustered energy patterns (SUEP)" or "softbomb" events. The shower is generated in its rest frame and 
// for a realistic simulation this class needs to be interfaced with an event generator such as madgraph, pythia or herwig.
//
// The algorithm relies on arXiv:1305.5226. See arXiv:1612.00850 for a description of the model.
// Please cite both papers when using this code.  

// Written by Simon Knapen on 12/22/2019

//following 1305.5226
#include <cmath>
#include <boost/math/tools/roots.hpp>
#include <boost/bind.hpp>

#include "GeneratorInterface/Pythia8Interface/interface/suep_shower.h"

using namespace std;                          // Help ADL of std functions.
using namespace boost::math::tools;           // For bracket_and_solve_root.

 // constructor
 Suep_shower::Suep_shower(double mass, double temperature, double energy, Pythia8::Rndm* rndmPtr) {
        m = mass;
        Temp=temperature;
        Etot=energy;
        fRndmPtr=rndmPtr;

        A=m/Temp;
        p_m=sqrt(2/(A*A)*(1+sqrt(1+A*A)));

        double pmax=sqrt(2+2*sqrt(1+A*A))/A; // compute the location of the maximum, to split the range 

        tolerance tol = 0.00001;

        p_plus = (bisect(boost::bind(&Suep_shower::test_fun, this, _1),pmax,50.0, tol)).first; // first root
        p_minus = (bisect(boost::bind(&Suep_shower::test_fun, this, _1), 0.0,pmax, tol)).first; // second root

        lambda_plus = - f(p_plus)/fp(p_plus);
        lambda_minus = f(p_minus)/fp(p_minus);
        q_plus = lambda_plus / (p_plus - p_minus);
        q_minus = lambda_minus / (p_plus - p_minus);
        q_m = 1- (q_plus + q_minus);

    }

// maxwell-boltzman distribution, slightly massaged
double  Suep_shower::f(double p){
    return p*p*exp(-A*p*p/(1+sqrt(1+p*p)));
}

// derivative of maxwell-boltzmann
double  Suep_shower::fp(double p){
    return exp(-A*p*p/(1+sqrt(1+p*p)))*p*(2-A*p*p/sqrt(1+p*p));
}

// test function to be solved for p_plusminus
double  Suep_shower::test_fun(double p){
    return log(f(p)/f(p_m))+1.0;
}

// generate one random 4 vector from the thermal distribution
vector<double> Suep_shower::generate_fourvector(){

    vector<double> fourvec;
    double en, phi, theta, p;//kinematic variables of the 4 vector

    // first do momentum, following arxiv:1305.5226
    double U, V, X, Y, E;
    int i=0;      
    while(i<100){
        U = fRndmPtr->flat();
        V = fRndmPtr->flat();

        if(U < q_m){
            Y=U/q_m;
            X=( 1 - Y )*( p_minus + lambda_minus )+Y*( p_plus - lambda_plus );
            if(V < f(X) / f(p_m) && X>0){
                break;
                }
            }
        else{if(U < q_m + q_plus){
            E = -log((U-q_m)/q_plus);
            X = p_plus - lambda_plus*(1-E);
            if(V<exp(E)*f(X)/f(p_m) && X>0){
                break;
                }
            }
            else{
                E = - log((U-(q_m+q_plus))/q_minus);
                X = p_minus + lambda_minus * (1 - E);
                if(V < exp(E)*f(X)/f(p_m) && X>0){
                    break;
                    }
                }
            }
        }
    p=X*(this->m); // X is the dimensionless momentum, p/m

    // now do the angles
    phi = 2.0*M_PI*(fRndmPtr->flat());
    theta = acos(2.0*(fRndmPtr->flat())-1.0);

    // compose the 4 vector
    en = sqrt(p*p+(this->m)*(this->m));
    fourvec.push_back(en);
    fourvec.push_back(p*cos(phi)*sin(theta));
    fourvec.push_back(p*sin(phi)*sin(theta));
    fourvec.push_back(p*cos(theta));

    return fourvec; 
}

// auxiliary function which computes the total energy difference as a function of the momentum vectors and a scale factor "a"
// to ballance energy, we solve for "a" by demanding that this function vanishes
// By rescaling the momentum rather than the energy, I avoid having to do these annoying rotations from the previous version 
double Suep_shower::reballance_func(double a, const vector< vector <double> >& event){
    double result =0.0;
    double p2;
    for(unsigned n = 0; n<event.size();n++){
        p2 = event[n][1]*event[n][1] + event[n][2]*event[n][2] + event[n][3]*event[n][3];
        result += sqrt(a*a*p2 + (this->m)* (this->m));
    }
    return result - (this->Etot);
}


// generate a shower event, in the rest frame of the shower
vector< vector <double> > Suep_shower::generate_shower(){

    vector<vector<double> > event;
    double sum_E = 0.0;

    // fill up event record
    while(sum_E<(this->Etot)){
        event.push_back(this->generate_fourvector());
        sum_E += (event.back()).at(0);     
    }

    // reballance momenta
    int len = event.size();
    double sum_p, correction;
    for(int i = 1;i<4;i++){ // loop over 3 carthesian directions

        sum_p = 0.0;
        for(int n=0;n<len;n++){
            sum_p+=event[n][i];
        }
        correction=-1.0*sum_p/len;

        for(int n=0;n<len;n++){
            event[n][i] += correction;
        } 
    }

    // finally, ballance the total energy, without destroying momentum conservation
    tolerance tol = 0.00001;
    double p_scale;
    p_scale = (bisect(boost::bind(&Suep_shower::reballance_func, this, _1, event),0.0,2.0, tol)).first;

    for(int n=0;n<len;n++){
            event[n][1] = p_scale*event[n][1];
            event[n][2] = p_scale*event[n][2];
            event[n][3] = p_scale*event[n][3];
            // force the energy with the on-shell condition
            event[n][0] = sqrt(event[n][1]*event[n][1] + event[n][2]*event[n][2] + event[n][3]*event[n][3] + (this->m)*(this->m));
        }

    return event;


}
