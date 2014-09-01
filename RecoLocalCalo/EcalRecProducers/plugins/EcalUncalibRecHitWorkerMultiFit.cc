#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerMultiFit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"

EcalUncalibRecHitWorkerMultiFit::EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&ps,edm::ConsumesCollector& c) :
  EcalUncalibRecHitWorkerBaseClass(ps,c),
  noisecorEBg12(10), noisecorEEg12(10),
  noisecorEBg6(10), noisecorEEg6(10),
  noisecorEBg1(10), noisecorEEg1(10),
  fullpulseEB(12),fullpulseEE(12),fullpulsecovEB(12),fullpulsecovEE(12)  
{
  
        //multifit method parameters
        const int nnoise = 10;
        
        double noisecorvEBg12[nnoise] = {1.00000, 0.71073, 0.55721, 0.46089, 0.40449, 0.35931, 0.33924, 0.32439, 0.31581, 0.30481};
        double noisecorvEEg12[nnoise] = {1.00000, 0.71373, 0.44825, 0.30152, 0.21609, 0.14786, 0.11772, 0.10165, 0.09465, 0.08098};
        
        double noisecorvEBg6[nnoise] = {1.00000, 0.70946, 0.58021, 0.49846, 0.45006, 0.41366, 0.39699, 0.38478, 0.37847, 0.37055};
        double noisecorvEEg6[nnoise] = {1.00000, 0.71217, 0.47464, 0.34056, 0.26282, 0.20287, 0.17734, 0.16256, 0.15618, 0.14443};
        
        double noisecorvEBg1[nnoise] = {1.00000, 0.73354, 0.64442, 0.58851, 0.55425, 0.53082, 0.51916, 0.51097, 0.50732, 0.50409};
        double noisecorvEEg1[nnoise] = {1.00000, 0.72698, 0.62048, 0.55691, 0.51848, 0.49147, 0.47813, 0.47007, 0.46621, 0.46265};
        
        //fill correlation matrices
        for (int i=0; i<nnoise; ++i) {
          for (int j=0; j<nnoise; ++j) {
            int vidx = std::abs(j-i);
            noisecorEBg12(i,j) = noisecorvEBg12[vidx];
            noisecorEEg12(i,j) = noisecorvEEg12[vidx];
            noisecorEBg6(i,j) = noisecorvEBg6[vidx];
            noisecorEEg6(i,j) = noisecorvEEg6[vidx];
            noisecorEBg1(i,j) = noisecorvEBg1[vidx];
            noisecorEEg1(i,j) = noisecorvEEg1[vidx];        
          }
        }
        
        fullpulseEB(0) = 1.123570e-02;
        fullpulseEB(1) = 7.572697e-01;
        fullpulseEB(2) = 1.000000e+00;
        fullpulseEB(3) = 8.880847e-01;
        fullpulseEB(4) = 6.739063e-01;
        fullpulseEB(5) = 4.746290e-01;
        fullpulseEB(6) = 3.198094e-01;
        fullpulseEB(7) = 2.002313e-01;
        fullpulseEB(8) = 1.240913e-01;
        fullpulseEB(9) = 7.523601e-02;
        fullpulseEB(10) = 4.482069e-02;
        fullpulseEB(11) = 2.637229e-02;
        fullpulseEE(0) = 1.155830e-01;
        fullpulseEE(1) = 7.554980e-01;
        fullpulseEE(2) = 1.000000e+00;
        fullpulseEE(3) = 8.975266e-01;
        fullpulseEE(4) = 6.872156e-01;
        fullpulseEE(5) = 4.918896e-01;
        fullpulseEE(6) = 3.444126e-01;
        fullpulseEE(7) = 2.120742e-01;
        fullpulseEE(8) = 1.318843e-01;
        fullpulseEE(9) = 8.005721e-02;
        fullpulseEE(10) = 4.765987e-02;
        fullpulseEE(11) = 2.797843e-02;
        fullpulsecovEB(0,0) = 3.089231e-06;
        fullpulsecovEB(0,1) = 1.364223e-05;
        fullpulsecovEB(0,2) = 0.000000e+00;
        fullpulsecovEB(0,3) = -4.841374e-06;
        fullpulsecovEB(0,4) = -5.016645e-06;
        fullpulsecovEB(0,5) = -3.978544e-06;
        fullpulsecovEB(0,6) = -2.954626e-06;
        fullpulsecovEB(0,7) = 0.000000e+00;
        fullpulsecovEB(0,8) = 0.000000e+00;
        fullpulsecovEB(0,9) = 0.000000e+00;
        fullpulsecovEB(0,10) = 0.000000e+00;
        fullpulsecovEB(0,11) = 0.000000e+00;
        fullpulsecovEB(1,0) = 1.364223e-05;
        fullpulsecovEB(1,1) = 6.723361e-05;
        fullpulsecovEB(1,2) = 0.000000e+00;
        fullpulsecovEB(1,3) = -2.390276e-05;
        fullpulsecovEB(1,4) = -2.487319e-05;
        fullpulsecovEB(1,5) = -1.987776e-05;
        fullpulsecovEB(1,6) = -1.482751e-05;
        fullpulsecovEB(1,7) = 0.000000e+00;
        fullpulsecovEB(1,8) = 0.000000e+00;
        fullpulsecovEB(1,9) = 0.000000e+00;
        fullpulsecovEB(1,10) = 0.000000e+00;
        fullpulsecovEB(1,11) = 0.000000e+00;
        fullpulsecovEB(2,0) = 0.000000e+00;
        fullpulsecovEB(2,1) = 0.000000e+00;
        fullpulsecovEB(2,2) = 0.000000e+00;
        fullpulsecovEB(2,3) = 0.000000e+00;
        fullpulsecovEB(2,4) = 0.000000e+00;
        fullpulsecovEB(2,5) = 0.000000e+00;
        fullpulsecovEB(2,6) = 0.000000e+00;
        fullpulsecovEB(2,7) = 0.000000e+00;
        fullpulsecovEB(2,8) = 0.000000e+00;
        fullpulsecovEB(2,9) = 0.000000e+00;
        fullpulsecovEB(2,10) = 0.000000e+00;
        fullpulsecovEB(2,11) = 0.000000e+00;
        fullpulsecovEB(3,0) = -4.841374e-06;
        fullpulsecovEB(3,1) = -2.390276e-05;
        fullpulsecovEB(3,2) = 0.000000e+00;
        fullpulsecovEB(3,3) = 8.821379e-06;
        fullpulsecovEB(3,4) = 9.053254e-06;
        fullpulsecovEB(3,5) = 7.222126e-06;
        fullpulsecovEB(3,6) = 5.379169e-06;
        fullpulsecovEB(3,7) = 0.000000e+00;
        fullpulsecovEB(3,8) = 0.000000e+00;
        fullpulsecovEB(3,9) = 0.000000e+00;
        fullpulsecovEB(3,10) = 0.000000e+00;
        fullpulsecovEB(3,11) = 0.000000e+00;
        fullpulsecovEB(4,0) = -5.016645e-06;
        fullpulsecovEB(4,1) = -2.487319e-05;
        fullpulsecovEB(4,2) = 0.000000e+00;
        fullpulsecovEB(4,3) = 9.053254e-06;
        fullpulsecovEB(4,4) = 9.555901e-06;
        fullpulsecovEB(4,5) = 7.581942e-06;
        fullpulsecovEB(4,6) = 5.657722e-06;
        fullpulsecovEB(4,7) = 0.000000e+00;
        fullpulsecovEB(4,8) = 0.000000e+00;
        fullpulsecovEB(4,9) = 0.000000e+00;
        fullpulsecovEB(4,10) = 0.000000e+00;
        fullpulsecovEB(4,11) = 0.000000e+00;
        fullpulsecovEB(5,0) = -3.978544e-06;
        fullpulsecovEB(5,1) = -1.987776e-05;
        fullpulsecovEB(5,2) = 0.000000e+00;
        fullpulsecovEB(5,3) = 7.222126e-06;
        fullpulsecovEB(5,4) = 7.581942e-06;
        fullpulsecovEB(5,5) = 6.252068e-06;
        fullpulsecovEB(5,6) = 4.612691e-06;
        fullpulsecovEB(5,7) = 0.000000e+00;
        fullpulsecovEB(5,8) = 0.000000e+00;
        fullpulsecovEB(5,9) = 0.000000e+00;
        fullpulsecovEB(5,10) = 0.000000e+00;
        fullpulsecovEB(5,11) = 0.000000e+00;
        fullpulsecovEB(6,0) = -2.954626e-06;
        fullpulsecovEB(6,1) = -1.482751e-05;
        fullpulsecovEB(6,2) = 0.000000e+00;
        fullpulsecovEB(6,3) = 5.379169e-06;
        fullpulsecovEB(6,4) = 5.657722e-06;
        fullpulsecovEB(6,5) = 4.612691e-06;
        fullpulsecovEB(6,6) = 3.627807e-06;
        fullpulsecovEB(6,7) = 0.000000e+00;
        fullpulsecovEB(6,8) = 0.000000e+00;
        fullpulsecovEB(6,9) = 0.000000e+00;
        fullpulsecovEB(6,10) = 0.000000e+00;
        fullpulsecovEB(6,11) = 0.000000e+00;
        fullpulsecovEB(7,0) = 0.000000e+00;
        fullpulsecovEB(7,1) = 0.000000e+00;
        fullpulsecovEB(7,2) = 0.000000e+00;
        fullpulsecovEB(7,3) = 0.000000e+00;
        fullpulsecovEB(7,4) = 0.000000e+00;
        fullpulsecovEB(7,5) = 0.000000e+00;
        fullpulsecovEB(7,6) = 0.000000e+00;
        fullpulsecovEB(7,7) = 3.627807e-06;
        fullpulsecovEB(7,8) = 0.000000e+00;
        fullpulsecovEB(7,9) = 0.000000e+00;
        fullpulsecovEB(7,10) = 0.000000e+00;
        fullpulsecovEB(7,11) = 0.000000e+00;
        fullpulsecovEB(8,0) = 0.000000e+00;
        fullpulsecovEB(8,1) = 0.000000e+00;
        fullpulsecovEB(8,2) = 0.000000e+00;
        fullpulsecovEB(8,3) = 0.000000e+00;
        fullpulsecovEB(8,4) = 0.000000e+00;
        fullpulsecovEB(8,5) = 0.000000e+00;
        fullpulsecovEB(8,6) = 0.000000e+00;
        fullpulsecovEB(8,7) = 0.000000e+00;
        fullpulsecovEB(8,8) = 3.627807e-06;
        fullpulsecovEB(8,9) = 0.000000e+00;
        fullpulsecovEB(8,10) = 0.000000e+00;
        fullpulsecovEB(8,11) = 0.000000e+00;
        fullpulsecovEB(9,0) = 0.000000e+00;
        fullpulsecovEB(9,1) = 0.000000e+00;
        fullpulsecovEB(9,2) = 0.000000e+00;
        fullpulsecovEB(9,3) = 0.000000e+00;
        fullpulsecovEB(9,4) = 0.000000e+00;
        fullpulsecovEB(9,5) = 0.000000e+00;
        fullpulsecovEB(9,6) = 0.000000e+00;
        fullpulsecovEB(9,7) = 0.000000e+00;
        fullpulsecovEB(9,8) = 0.000000e+00;
        fullpulsecovEB(9,9) = 3.627807e-06;
        fullpulsecovEB(9,10) = 0.000000e+00;
        fullpulsecovEB(9,11) = 0.000000e+00;
        fullpulsecovEB(10,0) = 0.000000e+00;
        fullpulsecovEB(10,1) = 0.000000e+00;
        fullpulsecovEB(10,2) = 0.000000e+00;
        fullpulsecovEB(10,3) = 0.000000e+00;
        fullpulsecovEB(10,4) = 0.000000e+00;
        fullpulsecovEB(10,5) = 0.000000e+00;
        fullpulsecovEB(10,6) = 0.000000e+00;
        fullpulsecovEB(10,7) = 0.000000e+00;
        fullpulsecovEB(10,8) = 0.000000e+00;
        fullpulsecovEB(10,9) = 0.000000e+00;
        fullpulsecovEB(10,10) = 3.627807e-06;
        fullpulsecovEB(10,11) = 0.000000e+00;
        fullpulsecovEB(11,0) = 0.000000e+00;
        fullpulsecovEB(11,1) = 0.000000e+00;
        fullpulsecovEB(11,2) = 0.000000e+00;
        fullpulsecovEB(11,3) = 0.000000e+00;
        fullpulsecovEB(11,4) = 0.000000e+00;
        fullpulsecovEB(11,5) = 0.000000e+00;
        fullpulsecovEB(11,6) = 0.000000e+00;
        fullpulsecovEB(11,7) = 0.000000e+00;
        fullpulsecovEB(11,8) = 0.000000e+00;
        fullpulsecovEB(11,9) = 0.000000e+00;
        fullpulsecovEB(11,10) = 0.000000e+00;
        fullpulsecovEB(11,11) = 3.627807e-06;
        fullpulsecovEE(0,0) = 4.488648e-05;
        fullpulsecovEE(0,1) = 3.855150e-05;
        fullpulsecovEE(0,2) = 0.000000e+00;
        fullpulsecovEE(0,3) = -1.716703e-05;
        fullpulsecovEE(0,4) = -1.966737e-05;
        fullpulsecovEE(0,5) = -1.729944e-05;
        fullpulsecovEE(0,6) = -1.469454e-05;
        fullpulsecovEE(0,7) = 0.000000e+00;
        fullpulsecovEE(0,8) = 0.000000e+00;
        fullpulsecovEE(0,9) = 0.000000e+00;
        fullpulsecovEE(0,10) = 0.000000e+00;
        fullpulsecovEE(0,11) = 0.000000e+00;
        fullpulsecovEE(1,0) = 3.855150e-05;
        fullpulsecovEE(1,1) = 3.373966e-05;
        fullpulsecovEE(1,2) = 0.000000e+00;
        fullpulsecovEE(1,3) = -1.497342e-05;
        fullpulsecovEE(1,4) = -1.720638e-05;
        fullpulsecovEE(1,5) = -1.522689e-05;
        fullpulsecovEE(1,6) = -1.307713e-05;
        fullpulsecovEE(1,7) = 0.000000e+00;
        fullpulsecovEE(1,8) = 0.000000e+00;
        fullpulsecovEE(1,9) = 0.000000e+00;
        fullpulsecovEE(1,10) = 0.000000e+00;
        fullpulsecovEE(1,11) = 0.000000e+00;
        fullpulsecovEE(2,0) = 0.000000e+00;
        fullpulsecovEE(2,1) = 0.000000e+00;
        fullpulsecovEE(2,2) = 0.000000e+00;
        fullpulsecovEE(2,3) = 0.000000e+00;
        fullpulsecovEE(2,4) = 0.000000e+00;
        fullpulsecovEE(2,5) = 0.000000e+00;
        fullpulsecovEE(2,6) = 0.000000e+00;
        fullpulsecovEE(2,7) = 0.000000e+00;
        fullpulsecovEE(2,8) = 0.000000e+00;
        fullpulsecovEE(2,9) = 0.000000e+00;
        fullpulsecovEE(2,10) = 0.000000e+00;
        fullpulsecovEE(2,11) = 0.000000e+00;
        fullpulsecovEE(3,0) = -1.716703e-05;
        fullpulsecovEE(3,1) = -1.497342e-05;
        fullpulsecovEE(3,2) = 0.000000e+00;
        fullpulsecovEE(3,3) = 7.317861e-06;
        fullpulsecovEE(3,4) = 8.272783e-06;
        fullpulsecovEE(3,5) = 7.267976e-06;
        fullpulsecovEE(3,6) = 6.225963e-06;
        fullpulsecovEE(3,7) = 0.000000e+00;
        fullpulsecovEE(3,8) = 0.000000e+00;
        fullpulsecovEE(3,9) = 0.000000e+00;
        fullpulsecovEE(3,10) = 0.000000e+00;
        fullpulsecovEE(3,11) = 0.000000e+00;
        fullpulsecovEE(4,0) = -1.966737e-05;
        fullpulsecovEE(4,1) = -1.720638e-05;
        fullpulsecovEE(4,2) = 0.000000e+00;
        fullpulsecovEE(4,3) = 8.272783e-06;
        fullpulsecovEE(4,4) = 9.960259e-06;
        fullpulsecovEE(4,5) = 8.757415e-06;
        fullpulsecovEE(4,6) = 7.487101e-06;
        fullpulsecovEE(4,7) = 0.000000e+00;
        fullpulsecovEE(4,8) = 0.000000e+00;
        fullpulsecovEE(4,9) = 0.000000e+00;
        fullpulsecovEE(4,10) = 0.000000e+00;
        fullpulsecovEE(4,11) = 0.000000e+00;
        fullpulsecovEE(5,0) = -1.729944e-05;
        fullpulsecovEE(5,1) = -1.522689e-05;
        fullpulsecovEE(5,2) = 0.000000e+00;
        fullpulsecovEE(5,3) = 7.267976e-06;
        fullpulsecovEE(5,4) = 8.757415e-06;
        fullpulsecovEE(5,5) = 8.286420e-06;
        fullpulsecovEE(5,6) = 7.079047e-06;
        fullpulsecovEE(5,7) = 0.000000e+00;
        fullpulsecovEE(5,8) = 0.000000e+00;
        fullpulsecovEE(5,9) = 0.000000e+00;
        fullpulsecovEE(5,10) = 0.000000e+00;
        fullpulsecovEE(5,11) = 0.000000e+00;
        fullpulsecovEE(6,0) = -1.469454e-05;
        fullpulsecovEE(6,1) = -1.307713e-05;
        fullpulsecovEE(6,2) = 0.000000e+00;
        fullpulsecovEE(6,3) = 6.225963e-06;
        fullpulsecovEE(6,4) = 7.487101e-06;
        fullpulsecovEE(6,5) = 7.079047e-06;
        fullpulsecovEE(6,6) = 6.623356e-06;
        fullpulsecovEE(6,7) = 0.000000e+00;
        fullpulsecovEE(6,8) = 0.000000e+00;
        fullpulsecovEE(6,9) = 0.000000e+00;
        fullpulsecovEE(6,10) = 0.000000e+00;
        fullpulsecovEE(6,11) = 0.000000e+00;
        fullpulsecovEE(7,0) = 0.000000e+00;
        fullpulsecovEE(7,1) = 0.000000e+00;
        fullpulsecovEE(7,2) = 0.000000e+00;
        fullpulsecovEE(7,3) = 0.000000e+00;
        fullpulsecovEE(7,4) = 0.000000e+00;
        fullpulsecovEE(7,5) = 0.000000e+00;
        fullpulsecovEE(7,6) = 0.000000e+00;
        fullpulsecovEE(7,7) = 6.623356e-06;
        fullpulsecovEE(7,8) = 0.000000e+00;
        fullpulsecovEE(7,9) = 0.000000e+00;
        fullpulsecovEE(7,10) = 0.000000e+00;
        fullpulsecovEE(7,11) = 0.000000e+00;
        fullpulsecovEE(8,0) = 0.000000e+00;
        fullpulsecovEE(8,1) = 0.000000e+00;
        fullpulsecovEE(8,2) = 0.000000e+00;
        fullpulsecovEE(8,3) = 0.000000e+00;
        fullpulsecovEE(8,4) = 0.000000e+00;
        fullpulsecovEE(8,5) = 0.000000e+00;
        fullpulsecovEE(8,6) = 0.000000e+00;
        fullpulsecovEE(8,7) = 0.000000e+00;
        fullpulsecovEE(8,8) = 6.623356e-06;
        fullpulsecovEE(8,9) = 0.000000e+00;
        fullpulsecovEE(8,10) = 0.000000e+00;
        fullpulsecovEE(8,11) = 0.000000e+00;
        fullpulsecovEE(9,0) = 0.000000e+00;
        fullpulsecovEE(9,1) = 0.000000e+00;
        fullpulsecovEE(9,2) = 0.000000e+00;
        fullpulsecovEE(9,3) = 0.000000e+00;
        fullpulsecovEE(9,4) = 0.000000e+00;
        fullpulsecovEE(9,5) = 0.000000e+00;
        fullpulsecovEE(9,6) = 0.000000e+00;
        fullpulsecovEE(9,7) = 0.000000e+00;
        fullpulsecovEE(9,8) = 0.000000e+00;
        fullpulsecovEE(9,9) = 6.623356e-06;
        fullpulsecovEE(9,10) = 0.000000e+00;
        fullpulsecovEE(9,11) = 0.000000e+00;
        fullpulsecovEE(10,0) = 0.000000e+00;
        fullpulsecovEE(10,1) = 0.000000e+00;
        fullpulsecovEE(10,2) = 0.000000e+00;
        fullpulsecovEE(10,3) = 0.000000e+00;
        fullpulsecovEE(10,4) = 0.000000e+00;
        fullpulsecovEE(10,5) = 0.000000e+00;
        fullpulsecovEE(10,6) = 0.000000e+00;
        fullpulsecovEE(10,7) = 0.000000e+00;
        fullpulsecovEE(10,8) = 0.000000e+00;
        fullpulsecovEE(10,9) = 0.000000e+00;
        fullpulsecovEE(10,10) = 6.623356e-06;
        fullpulsecovEE(10,11) = 0.000000e+00;
        fullpulsecovEE(11,0) = 0.000000e+00;
        fullpulsecovEE(11,1) = 0.000000e+00;
        fullpulsecovEE(11,2) = 0.000000e+00;
        fullpulsecovEE(11,3) = 0.000000e+00;
        fullpulsecovEE(11,4) = 0.000000e+00;
        fullpulsecovEE(11,5) = 0.000000e+00;
        fullpulsecovEE(11,6) = 0.000000e+00;
        fullpulsecovEE(11,7) = 0.000000e+00;
        fullpulsecovEE(11,8) = 0.000000e+00;
        fullpulsecovEE(11,9) = 0.000000e+00;
        fullpulsecovEE(11,10) = 0.000000e+00;
        fullpulsecovEE(11,11) = 6.623356e-06;        
  
        activeBX.insert(-5);
        activeBX.insert(-4);
        activeBX.insert(-3);
        activeBX.insert(-2);
        activeBX.insert(-1);
        activeBX.insert(0);
        activeBX.insert(1);
        activeBX.insert(2);
        activeBX.insert(3);
        activeBX.insert(4);        
        
        // ratio method parameters
        EBtimeFitParameters_ = ps.getParameter<std::vector<double> >("EBtimeFitParameters"); 
        EEtimeFitParameters_ = ps.getParameter<std::vector<double> >("EEtimeFitParameters"); 
        EBamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EBamplitudeFitParameters");
        EEamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EEamplitudeFitParameters");
        EBtimeFitLimits_.first  = ps.getParameter<double>("EBtimeFitLimits_Lower");
        EBtimeFitLimits_.second = ps.getParameter<double>("EBtimeFitLimits_Upper");
        EEtimeFitLimits_.first  = ps.getParameter<double>("EEtimeFitLimits_Lower");
        EEtimeFitLimits_.second = ps.getParameter<double>("EEtimeFitLimits_Upper");
        EBtimeConstantTerm_=ps.getParameter<double>("EBtimeConstantTerm");
        EEtimeConstantTerm_=ps.getParameter<double>("EEtimeConstantTerm");

        // leading edge parameters
        ebPulseShape_ = ps.getParameter<std::vector<double> >("ebPulseShape");
        eePulseShape_ = ps.getParameter<std::vector<double> >("eePulseShape");

        // chi2 parameters for flags determination
        kPoorRecoFlagEB_ = ps.getParameter<bool>("kPoorRecoFlagEB");
        kPoorRecoFlagEE_ = ps.getParameter<bool>("kPoorRecoFlagEE");;
        chi2ThreshEB_=ps.getParameter<double>("chi2ThreshEB_");
        chi2ThreshEE_=ps.getParameter<double>("chi2ThreshEE_");

        // significance of the additional OOT pulses 
        significanceOutOfTime_ = ps.getParameter<double>("significanceOutOfTime");
 }



// EcalUncalibRecHitWorkerMultiFit::EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&ps) :
//   EcalUncalibRecHitWorkerBaseClass(ps)
// {
//         // ratio method parameters
//         EBtimeFitParameters_ = ps.getParameter<std::vector<double> >("EBtimeFitParameters"); 
//         EEtimeFitParameters_ = ps.getParameter<std::vector<double> >("EEtimeFitParameters"); 
//         EBamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EBamplitudeFitParameters");
//         EEamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EEamplitudeFitParameters");
//         EBtimeFitLimits_.first  = ps.getParameter<double>("EBtimeFitLimits_Lower");
//         EBtimeFitLimits_.second = ps.getParameter<double>("EBtimeFitLimits_Upper");
//         EEtimeFitLimits_.first  = ps.getParameter<double>("EEtimeFitLimits_Lower");
//         EEtimeFitLimits_.second = ps.getParameter<double>("EEtimeFitLimits_Upper");
//         EBtimeConstantTerm_=ps.getParameter<double>("EBtimeConstantTerm");
//         EEtimeConstantTerm_=ps.getParameter<double>("EEtimeConstantTerm");
// 
//         // leading edge parameters
//         ebPulseShape_ = ps.getParameter<std::vector<double> >("ebPulseShape");
//         eePulseShape_ = ps.getParameter<std::vector<double> >("eePulseShape");
// 
// }













void
EcalUncalibRecHitWorkerMultiFit::set(const edm::EventSetup& es)
{
        // common setup
        es.get<EcalGainRatiosRcd>().get(gains);
        es.get<EcalPedestalsRcd>().get(peds);


	// which of the samples need be used
	es.get<EcalSampleMaskRcd>().get(sampleMaskHand_);

        // for the ratio method

        // for the leading edge method
        es.get<EcalTimeCalibConstantsRcd>().get(itime);
        es.get<EcalTimeOffsetConstantRcd>().get(offtime);

		// for the time correction methods
		es.get<EcalTimeBiasCorrectionsRcd>().get(timeCorrBias_);
}

/**
 * Amplitude-dependent time corrections; EE and EB have separate corrections:
 * EXtimeCorrAmplitudes (ADC) and EXtimeCorrShifts (ns) need to have the same number of elements
 * Bins must be ordered in amplitude. First-last bins take care of under-overflows.
 *
 * The algorithm is the same for EE and EB, only the correction vectors are different.
 *
 * @return Jitter (in clock cycles) which will be added to UncalibRechit.setJitter(), 0 if no correction is applied.
 */
double EcalUncalibRecHitWorkerMultiFit::timeCorrection(
    float ampli,
	const std::vector<float>& amplitudeBins,
    const std::vector<float>& shiftBins) {

  // computed initially in ns. Than turned in the BX's, as
  // EcalUncalibratedRecHit need be.
  double theCorrection = 0;

  // sanity check for arrays
  if (amplitudeBins.size() == 0) {
    edm::LogError("EcalRecHitError")
        << "timeCorrAmplitudeBins is empty, forcing no time bias corrections.";

    return 0;
  }

  if (amplitudeBins.size() != shiftBins.size()) {
    edm::LogError("EcalRecHitError")
        << "Size of timeCorrAmplitudeBins different from "
           "timeCorrShiftBins. Forcing no time bias corrections. ";

    return 0;
  }

  int myBin = -1;
  for (int bin = 0; bin < (int) amplitudeBins.size(); bin++) {
    if (ampli > amplitudeBins.at(bin)) {
      myBin = bin;
    } else {
      break;
	}
  }

  if (myBin == -1) {
    theCorrection = shiftBins.at(0);
  } else if (myBin == ((int)(amplitudeBins.size() - 1))) {
    theCorrection = shiftBins.at(myBin);
  } else if (-1 < myBin && myBin < ((int) amplitudeBins.size() - 1)) {
    // interpolate linearly between two assingned points
    theCorrection = (shiftBins.at(myBin + 1) - shiftBins.at(myBin));
    theCorrection *= (((double) ampli) - amplitudeBins.at(myBin)) /
                     (amplitudeBins.at(myBin + 1) - amplitudeBins.at(myBin));
    theCorrection += shiftBins.at(myBin);
  } else {
    edm::LogError("EcalRecHitError")
        << "Assigning time correction impossible. Setting it to 0 ";
    theCorrection = 0.;
  }

  // convert ns into clocks
  return theCorrection / 25.;
}



bool
EcalUncalibRecHitWorkerMultiFit::run( const edm::Event & evt,
                const EcalDigiCollection::const_iterator & itdg,
                EcalUncalibratedRecHitCollection & result )
{
        DetId detid(itdg->id());

        const EcalSampleMask  *sampleMask_ = sampleMaskHand_.product();                
        
        // intelligence for recHit computation
        EcalUncalibratedRecHit uncalibRecHit;
        
        
        const EcalPedestals::Item * aped = 0;
        const EcalMGPAGainRatio * aGain = 0;

        if (detid.subdetId()==EcalEndcap) {
                unsigned int hashedIndex = EEDetId(detid).hashedIndex();
                aped  = &peds->endcap(hashedIndex);
                aGain = &gains->endcap(hashedIndex);
        } else {
                unsigned int hashedIndex = EBDetId(detid).hashedIndex();
                aped  = &peds->barrel(hashedIndex);
                aGain = &gains->barrel(hashedIndex);
        }

        pedVec[0] = aped->mean_x12;
        pedVec[1] = aped->mean_x6;
        pedVec[2] = aped->mean_x1;
        pedRMSVec[0] = aped->rms_x12;
        pedRMSVec[1] = aped->rms_x6;
        pedRMSVec[2] = aped->rms_x1;
        gainRatios[0] = 1.;
        gainRatios[1] = aGain->gain12Over6();
        gainRatios[2] = aGain->gain6Over1()*aGain->gain12Over6();

        
        // === amplitude computation ===
        int leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();

        if ( leadingSample >= 0 ) { // saturation
                if ( leadingSample != 4 ) {
                        // all samples different from the fifth are not reliable for the amplitude estimation
                        // put by default the energy at the saturation threshold and flag as saturated
                        float sratio = 1;
                        if ( detid.subdetId()==EcalBarrel) {
                                sratio = ebPulseShape_[5] / ebPulseShape_[4];
                        } else {
                                sratio = eePulseShape_[5] / eePulseShape_[4];
                        }
			uncalibRecHit = EcalUncalibratedRecHit( (*itdg).id(), 4095*12*sratio, 0, 0, 0);
                        uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kSaturated );
                } else {
                        // float clockToNsConstant = 25.;
                        // reconstruct the rechit
                        if (detid.subdetId()==EcalEndcap) {
                                leadingEdgeMethod_endcap_.setPulseShape( eePulseShape_ );
                                // float mult = (float)eePulseShape_.size() / (float)(*itdg).size();
                                // bin (or some analogous mapping) will be used instead of the leadingSample
                                //int bin  = (int)(( (mult * leadingSample + mult/2) * clockToNsConstant + itimeconst ) / clockToNsConstant);
                                // bin is not uset for the moment
                                leadingEdgeMethod_endcap_.setLeadingEdgeSample( leadingSample );
                                uncalibRecHit = leadingEdgeMethod_endcap_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
                                uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kLeadingEdgeRecovered );
                                leadingEdgeMethod_endcap_.setLeadingEdgeSample( -1 );
                        } else {
                                leadingEdgeMethod_barrel_.setPulseShape( ebPulseShape_ );
                                // float mult = (float)ebPulseShape_.size() / (float)(*itdg).size();
                                // bin (or some analogous mapping) will be used instead of the leadingSample
                                //int bin  = (int)(( (mult * leadingSample + mult/2) * clockToNsConstant + itimeconst ) / clockToNsConstant);
                                // bin is not uset for the moment
                                leadingEdgeMethod_barrel_.setLeadingEdgeSample( leadingSample );
                                uncalibRecHit = leadingEdgeMethod_barrel_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
                                uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kLeadingEdgeRecovered );
                                leadingEdgeMethod_barrel_.setLeadingEdgeSample( -1 );
                        }
                }
		// do not propagate the default chi2 = -1 value to the calib rechit (mapped to 64), set it to 0 when saturation
                uncalibRecHit.setChi2(0);
        } else {
                // multifit
                bool barrel = detid.subdetId()==EcalBarrel;
                int gain = 12;
                if (((EcalDataFrame)(*itdg)).hasSwitchToGain6()) {
                  gain = 6;
                }
                if (((EcalDataFrame)(*itdg)).hasSwitchToGain1()) {
                  gain = 1;
                }
                const TMatrixDSym &noisecormat = noisecor(barrel,gain);
                const TVectorD &fullpulse = barrel ? fullpulseEB : fullpulseEE;
                const TMatrixDSym &fullpulsecov = barrel ? fullpulsecovEB : fullpulsecovEE;
                                
                uncalibRecHit = multiFitMethod_.makeRecHit(*itdg, aped, aGain, noisecormat,fullpulse,fullpulsecov,activeBX);
                
                // out of time flags (not proper flag with this method, but says if the Detid is populated just by OOT PU
                // not active for tests of the clustering
                /*
                if(uncalibRecHit.amplitude()/uncalibRecHit.amplitudeError() < significanceOutOfTime_) {
                  bool significantOOTPulse=false;
                  for(int ibx=0; ibx<EcalDataFrame::MAXSAMPLES; ++ibx) {
                    if(uncalibRecHit.outOfTimeAmplitudeErr(ibx) > 0. && 
                       uncalibRecHit.outOfTimeAmplitude(ibx)/uncalibRecHit.outOfTimeAmplitudeErr(ibx) > significanceOutOfTime_) significantOOTPulse=true;
                  }
                  if(significantOOTPulse) uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kOutOfTime );
                }
                */

                // === time computation ===
                // ratio method
                float const clockToNsConstant = 25.;
                if (detid.subdetId()==EcalEndcap) {
    		                ratioMethod_endcap_.init( *itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios );
                                ratioMethod_endcap_.computeTime( EEtimeFitParameters_, EEtimeFitLimits_, EEamplitudeFitParameters_ );
                                ratioMethod_endcap_.computeAmplitude( EEamplitudeFitParameters_);
                                EcalUncalibRecHitRatioMethodAlgo<EEDataFrame>::CalculatedRecHit crh = ratioMethod_endcap_.getCalculatedRecHit();
				double theTimeCorrectionEE = timeCorrection(uncalibRecHit.amplitude(),
					timeCorrBias_->EETimeCorrAmplitudeBins, timeCorrBias_->EETimeCorrShiftBins);

                                uncalibRecHit.setJitter( crh.timeMax - 5 + theTimeCorrectionEE);
                                uncalibRecHit.setJitterError( std::sqrt(pow(crh.timeError,2) + std::pow(EEtimeConstantTerm_,2)/std::pow(clockToNsConstant,2)) );
				
                } else {
 		                ratioMethod_barrel_.init( *itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios );
				ratioMethod_barrel_.fixMGPAslew(*itdg);
                                ratioMethod_barrel_.computeTime( EBtimeFitParameters_, EBtimeFitLimits_, EBamplitudeFitParameters_ );
                                ratioMethod_barrel_.computeAmplitude( EBamplitudeFitParameters_);
                                EcalUncalibRecHitRatioMethodAlgo<EBDataFrame>::CalculatedRecHit crh = ratioMethod_barrel_.getCalculatedRecHit();

				double theTimeCorrectionEB = timeCorrection(uncalibRecHit.amplitude(),
					timeCorrBias_->EBTimeCorrAmplitudeBins, timeCorrBias_->EBTimeCorrShiftBins);

				uncalibRecHit.setJitter( crh.timeMax - 5 + theTimeCorrectionEB);

                                uncalibRecHit.setJitterError( std::sqrt(std::pow(crh.timeError,2) + std::pow(EBtimeConstantTerm_,2)/std::pow(clockToNsConstant,2)) );
		}
		
        }

	// set flags if gain switch has occurred
	if( ((EcalDataFrame)(*itdg)).hasSwitchToGain6()  ) uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kHasSwitchToGain6 );
	if( ((EcalDataFrame)(*itdg)).hasSwitchToGain1()  ) uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kHasSwitchToGain1 );

        // set quality flags based on chi2 of the the fit
        /*
        if(detid.subdetId()==EcalEndcap) { 
          if(kPoorRecoFlagEE_ && uncalibRecHit.chi2()>chi2ThreshEE_) {
          bool samplesok = true;
          for (int sample =0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
            if (!sampleMask_->useSampleEE(sample)) {
              samplesok = false;
              break;
            }
          }
          if (samplesok) uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kPoorReco);
          }
        } else {
          if(kPoorRecoFlagEB_ && uncalibRecHit.chi2()>chi2ThreshEB_) {
          bool samplesok = true;
          for (int sample =0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
            if (!sampleMask_->useSampleEB(sample)) {
              samplesok = false;
              break;
            }
          }
          if (samplesok) uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kPoorReco);
          }
        }
        */
        

        // put the recHit in the collection
        if (detid.subdetId()==EcalEndcap) {
                result.push_back( uncalibRecHit );
        } else {
                result.push_back( uncalibRecHit );
        }

        return true;
}


const TMatrixDSym &EcalUncalibRecHitWorkerMultiFit::noisecor(bool barrel, int gain) const {
  if (barrel) {
    if (gain==6) {
      return noisecorEBg6;
    }
    else if (gain==1) {
      return noisecorEBg1;
    }
    else {
      return noisecorEBg12;
    }    
  }
  else {
    if (gain==6) {
      return noisecorEEg6;
    }
    else if (gain==1) {
      return noisecorEEg1;
    }
    else {
      return noisecorEEg12;
    }        
  }
  
  return noisecorEBg12;
  
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerMultiFit, "EcalUncalibRecHitWorkerMultiFit" );
