// -*- C++ -*-
//
// Package:    EcalDeadChannelRecoveryAlgos
// Class:      CorrectEBDeadChannelsNN
//
/**\class CorrectEBDeadChannelsNN CorrectEBDeadChannelsNN.cc RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/CorrectEBDeadChannelsNN.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
     
     Return Value:  1)  Normal execution returns a positive number ("double"), coresponding to the ANN estimate for the energy of the "dead" cell.
                    2)  Non-normal execution returns a negative number ("double") with the following meaning:
                            -1000000.0      Zero DC's were detected
                            -1000001.0      More than one DC's detected.
                            -2000000.0      Non-positive (i.e negative or zero) cell energy detected within at least one "live" cell
                            -3000000.0      Detector region provided was EB but no match with a "dead" cell case was detected
                            -3000001.0      Detector region provided was EE but no match with a "dead" cell case was detected
                        To avoid future conflicts the return values have been set to very-high unphysical values
*/
// 
//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Wed Nov 21 11:24:39 EET 2012
// 
//      Nov 21 2012:   First version of the code. Based on the old "CorrectDeadChannelsNN.cc" code
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TMath.h>

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/xyNNEB.h"
    
using namespace std;

double CorrectEBDeadChannelsNN(double *M3x3Input, double epsilon=0.0000001) {

    Double_t NNResult ;
    
    //  Arrangement within the M3x3Input matrix
    //
    //                  M3x3
    //   -----------------------------------
    //   
    //   
    //   LU  UU  RU             04  01  07
    //   LL  CC  RR      or     03  00  06
    //   LD  DD  RD             05  02  08
    //   
    //   


    //  Enumeration to switch from custom names within the 3x3 matrix.
    enum { CC=0, UU=1, DD=2, LL=3, LU=4, LD=5, RR=6, RU=7, RD=8 } ;

    //  Conversion between M3x3Input matrix and M5x5 matrix needed by NN:

    Double_t M3x3[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } ;          //  This is the 3x3 around the Max Containment Crystal.
    
    M3x3[LU] = M3x3Input[LU];   M3x3[UU] = M3x3Input[UU];   M3x3[RU] = M3x3Input[RU];
    M3x3[LL] = M3x3Input[LL];   M3x3[CC] = M3x3Input[CC];   M3x3[RR] = M3x3Input[RR];
    M3x3[LD] = M3x3Input[LD];   M3x3[DD] = M3x3Input[DD];   M3x3[RD] = M3x3Input[RD];

    //  Find the Dead Channels inside the 3x3 matrix
    std::vector<Int_t> idxDC_v;
    
    for (Int_t i=0; i<9; i++) {
        if ( TMath::Abs(M3x3[i]) < epsilon ) { idxDC_v.push_back(i) ; }
    }

    //  Currently EXACTLY ONE AND ONLY ONE dead cell is corrected. Return -1000000.0 if zero DC's detected and -101.0 if more than one DC's exist.
    Int_t idxDC = -1 ;
    if ( idxDC_v.size() == 0 ) { NNResult = -1000000.0 ; return NNResult ; }    //  Zero DC's were detected
    if ( idxDC_v.size()  > 1 ) { NNResult = -1000001.0 ; return NNResult ; }    //  More than one DC's detected.
    if ( idxDC_v.size() == 1 ) { idxDC = idxDC_v.at(0) ; } 

    //  Generally the "dead" cells are allowed to have negative energies (since they will be estimated by the ANN anyway).
    //  But all the remaining "live" ones must have positive values otherwise the logarithm fails.
    for (Int_t i=0; i<9; i++) {
        if ( M3x3[i] < 0.0 && TMath::Abs(M3x3[i]) >= epsilon ) { NNResult = -2000000.0 ; return NNResult ; }
    }
    
    //  Call ANN code depending the detector region. Intermediate varibles "lnXX" are created individualy inside the switch statement
    //  instead of globaly outside the "if-block" to avoid the case where some cell value is zero (that would be the "dead" cell).
    //
    Double_t lnLU ; Double_t lnUU ; Double_t lnRU ;
    Double_t lnLL ; Double_t lnCC ; Double_t lnRR ;
    Double_t lnLD ; Double_t lnDD ; Double_t lnRD ;

    //  Select the case to apply the appropriate NN and return the result.
    if        ( idxDC == CC ) {
    
        lnLU = TMath::Log( M3x3[LU] ) ; lnUU = TMath::Log( M3x3[UU] ) ; lnRU = TMath::Log( M3x3[RU] ) ;
        lnLL = TMath::Log( M3x3[LL] ) ;                                 lnRR = TMath::Log( M3x3[RR] ) ;
        lnLD = TMath::Log( M3x3[LD] ) ; lnDD = TMath::Log( M3x3[DD] ) ; lnRD = TMath::Log( M3x3[RD] ) ;
                
        ccNNEB* ccNNObjEB = new ccNNEB();   NNResult = TMath::Exp( ccNNObjEB->Value( 0, lnRR, lnLL, lnUU, lnDD, lnRU, lnRD, lnLU, lnLD ) ) ;    delete ccNNObjEB ;
                
        M3x3Input[CC] = NNResult ;
        
    } else if ( idxDC == RR ) {

        lnLU = TMath::Log( M3x3[LU] ) ; lnUU = TMath::Log( M3x3[UU] ) ; lnRU = TMath::Log( M3x3[RU] ) ;
        lnLL = TMath::Log( M3x3[LL] ) ; lnCC = TMath::Log( M3x3[CC] ) ;
        lnLD = TMath::Log( M3x3[LD] ) ; lnDD = TMath::Log( M3x3[DD] ) ; lnRD = TMath::Log( M3x3[RD] ) ;
                
        rrNNEB* rrNNObjEB = new rrNNEB();   NNResult = TMath::Exp( rrNNObjEB->Value( 0, lnCC, lnLL, lnUU, lnDD, lnRU, lnRD, lnLU, lnLD ) ) ;    delete rrNNObjEB ;

        M3x3Input[RR] = NNResult ;
                
    } else if ( idxDC == LL ) {

        lnLU = TMath::Log( M3x3[LU] ) ; lnUU = TMath::Log( M3x3[UU] ) ; lnRU = TMath::Log( M3x3[RU] ) ;
                                        lnCC = TMath::Log( M3x3[CC] ) ; lnRR = TMath::Log( M3x3[RR] ) ;
        lnLD = TMath::Log( M3x3[LD] ) ; lnDD = TMath::Log( M3x3[DD] ) ; lnRD = TMath::Log( M3x3[RD] ) ;
                
        llNNEB* llNNObjEB = new llNNEB();   NNResult = TMath::Exp( llNNObjEB->Value( 0, lnCC, lnRR, lnUU, lnDD, lnRU, lnRD, lnLU, lnLD ) ) ;    delete llNNObjEB ;
                
        M3x3Input[LL] = NNResult ;
        
    } else if ( idxDC == UU ) {

        lnLU = TMath::Log( M3x3[LU] ) ;                                 lnRU = TMath::Log( M3x3[RU] ) ;
        lnLL = TMath::Log( M3x3[LL] ) ; lnCC = TMath::Log( M3x3[CC] ) ; lnRR = TMath::Log( M3x3[RR] ) ;
        lnLD = TMath::Log( M3x3[LD] ) ; lnDD = TMath::Log( M3x3[DD] ) ; lnRD = TMath::Log( M3x3[RD] ) ;
                
        uuNNEB* uuNNObjEB = new uuNNEB();   NNResult = TMath::Exp( uuNNObjEB->Value( 0, lnCC, lnRR, lnLL, lnDD, lnRU, lnRD, lnLU, lnLD ) ) ;    delete uuNNObjEB ;

        M3x3Input[UU] = NNResult ;
                
    } else if ( idxDC == DD ) {

        lnLU = TMath::Log( M3x3[LU] ) ; lnUU = TMath::Log( M3x3[UU] ) ; lnRU = TMath::Log( M3x3[RU] ) ;
        lnLL = TMath::Log( M3x3[LL] ) ; lnCC = TMath::Log( M3x3[CC] ) ; lnRR = TMath::Log( M3x3[RR] ) ;
        lnLD = TMath::Log( M3x3[LD] ) ;                                 lnRD = TMath::Log( M3x3[RD] ) ;
                
        ddNNEB* ddNNObjEB = new ddNNEB();   NNResult = TMath::Exp( ddNNObjEB->Value( 0, lnCC, lnRR, lnLL, lnUU, lnRU, lnRD, lnLU, lnLD ) ) ;    delete ddNNObjEB ;
                
        M3x3Input[DD] = NNResult ;
        
    } else if ( idxDC == RU ) {

        lnLU = TMath::Log( M3x3[LU] ) ; lnUU = TMath::Log( M3x3[UU] ) ; 
        lnLL = TMath::Log( M3x3[LL] ) ; lnCC = TMath::Log( M3x3[CC] ) ; lnRR = TMath::Log( M3x3[RR] ) ;
        lnLD = TMath::Log( M3x3[LD] ) ; lnDD = TMath::Log( M3x3[DD] ) ; lnRD = TMath::Log( M3x3[RD] ) ;
                
        ruNNEB* ruNNObjEB = new ruNNEB();   NNResult = TMath::Exp( ruNNObjEB->Value( 0, lnCC, lnRR, lnLL, lnUU, lnDD, lnRD, lnLU, lnLD ) ) ;    delete ruNNObjEB ;

        M3x3Input[RU] = NNResult ;
                
    } else if ( idxDC == RD ) {

        lnLU = TMath::Log( M3x3[LU] ) ; lnUU = TMath::Log( M3x3[UU] ) ; lnRU = TMath::Log( M3x3[RU] ) ;
        lnLL = TMath::Log( M3x3[LL] ) ; lnCC = TMath::Log( M3x3[CC] ) ; lnRR = TMath::Log( M3x3[RR] ) ;
        lnLD = TMath::Log( M3x3[LD] ) ; lnDD = TMath::Log( M3x3[DD] ) ; 
                
        rdNNEB* rdNNObjEB = new rdNNEB();   NNResult = TMath::Exp( rdNNObjEB->Value( 0, lnCC, lnRR, lnLL, lnUU, lnDD, lnRU, lnLU, lnLD ) ) ;    delete rdNNObjEB ;
                
        M3x3Input[RD] = NNResult ;
                        
    } else if ( idxDC == LU ) {

                                        lnUU = TMath::Log( M3x3[UU] ) ; lnRU = TMath::Log( M3x3[RU] ) ;
        lnLL = TMath::Log( M3x3[LL] ) ; lnCC = TMath::Log( M3x3[CC] ) ; lnRR = TMath::Log( M3x3[RR] ) ;
        lnLD = TMath::Log( M3x3[LD] ) ; lnDD = TMath::Log( M3x3[DD] ) ; lnRD = TMath::Log( M3x3[RD] ) ;
                
        luNNEB* luNNObjEB = new luNNEB();   NNResult = TMath::Exp( luNNObjEB->Value( 0, lnCC, lnRR, lnLL, lnUU, lnDD, lnRU, lnRD, lnLD ) ) ;    delete luNNObjEB ;

        M3x3Input[LU] = NNResult ;

    } else if ( idxDC == LD ) {

        lnLU = TMath::Log( M3x3[LU] ) ; lnUU = TMath::Log( M3x3[UU] ) ; lnRU = TMath::Log( M3x3[RU] ) ;
        lnLL = TMath::Log( M3x3[LL] ) ; lnCC = TMath::Log( M3x3[CC] ) ; lnRR = TMath::Log( M3x3[RR] ) ;
                                        lnDD = TMath::Log( M3x3[DD] ) ; lnRD = TMath::Log( M3x3[RD] ) ;
                                                         
        ldNNEB* ldNNObjEB = new ldNNEB();   NNResult = TMath::Exp( ldNNObjEB->Value( 0, lnCC, lnRR, lnLL, lnUU, lnDD, lnRU, lnRD, lnLU ) ) ;    delete ldNNObjEB ;
                
        M3x3Input[LD] = NNResult ;
        
    } else {
        //  Error
        NNResult = -3000000.0;
    }

    return NNResult ;
}
