#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_CorrectEBDeadChannelsNN_H
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_CorrectEBDeadChannelsNN_H

// -*- C++ -*-
//
// Package:    EcalDeadChannelRecoveryAlgos
// Class:      CorrectEBDeadChannelsNN
//
/**\class CorrectEBDeadChannelsNN CorrectEBDeadChannelsNN.cc RecoLocalCalo/EcalDeadChannelRecoveryAlgos/src/CorrectEBDeadChannelsNN.cc

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

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/DeadChannelNNContext.h"

double CorrectEBDeadChannelsNN(DeadChannelNNContext &ctx, double *M3x3Input, double epsilon=0.0000001);

#endif
