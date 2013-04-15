#include "MuonTaggerMLP.h"
#include <cmath>

double MuonTaggerMLP::Value(int index,double in0,double in1,double in2,double in3) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   switch(index) {
     case 0:
         return neuron0x67dc690();
     default:
         return 0.;
   }
}

double MuonTaggerMLP::Value(int index, double* input) {
   input0 = (input[0] - 0)/1;
   input1 = (input[1] - 0)/1;
   input2 = (input[2] - 0)/1;
   input3 = (input[3] - 0)/1;
   switch(index) {
     case 0:
         return neuron0x67dc690();
     default:
         return 0.;
   }
}

double MuonTaggerMLP::neuron0x67da870() {
   return input0;
}

double MuonTaggerMLP::neuron0x67dabb0() {
   return input1;
}

double MuonTaggerMLP::neuron0x67daef0() {
   return input2;
}

double MuonTaggerMLP::neuron0x67db230() {
   return input3;
}

double MuonTaggerMLP::input0x67db6a0() {
   double input = 1.7302;
   input += synapse0x67b3ae0();
   input += synapse0x67b39a0();
   input += synapse0x67db950();
   input += synapse0x67db990();
   return input;
}

double MuonTaggerMLP::neuron0x67db6a0() {
   double input = input0x67db6a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double MuonTaggerMLP::input0x67db9d0() {
   double input = -2.53635;
   input += synapse0x67dbd10();
   input += synapse0x67dbd50();
   input += synapse0x67dbd90();
   input += synapse0x67dbdd0();
   return input;
}

double MuonTaggerMLP::neuron0x67db9d0() {
   double input = input0x67db9d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double MuonTaggerMLP::input0x67dbe10() {
   double input = 0.435731;
   input += synapse0x67dc150();
   input += synapse0x67dc190();
   input += synapse0x67dc1d0();
   input += synapse0x67dc210();
   return input;
}

double MuonTaggerMLP::neuron0x67dbe10() {
   double input = input0x67dbe10();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double MuonTaggerMLP::input0x67dc250() {
   double input = -0.787089;
   input += synapse0x67dc590();
   input += synapse0x67dc5d0();
   input += synapse0x67dc610();
   input += synapse0x67dc650();
   return input;
}

double MuonTaggerMLP::neuron0x67dc250() {
   double input = input0x67dc250();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double MuonTaggerMLP::input0x67dc690() {
   double input = 0.0472732;
   input += synapse0x67dc9d0();
   input += synapse0x677bb20();
   input += synapse0x6745bb0();
   input += synapse0x6745bf0();
   return input;
}

double MuonTaggerMLP::neuron0x67dc690() {
   double input = input0x67dc690();
   return (input * 1)+0;
}

double MuonTaggerMLP::synapse0x67b3ae0() {
   return (neuron0x67da870()*7.45967);
}

double MuonTaggerMLP::synapse0x67b39a0() {
   return (neuron0x67dabb0()*1.40943);
}

double MuonTaggerMLP::synapse0x67db950() {
   return (neuron0x67daef0()*-14.7121);
}

double MuonTaggerMLP::synapse0x67db990() {
   return (neuron0x67db230()*-2.04334);
}

double MuonTaggerMLP::synapse0x67dbd10() {
   return (neuron0x67da870()*-0.341096);
}

double MuonTaggerMLP::synapse0x67dbd50() {
   return (neuron0x67dabb0()*1.11996);
}

double MuonTaggerMLP::synapse0x67dbd90() {
   return (neuron0x67daef0()*3.96074);
}

double MuonTaggerMLP::synapse0x67dbdd0() {
   return (neuron0x67db230()*6.91115);
}

double MuonTaggerMLP::synapse0x67dc150() {
   return (neuron0x67da870()*1.81332);
}

double MuonTaggerMLP::synapse0x67dc190() {
   return (neuron0x67dabb0()*0.0399628);
}

double MuonTaggerMLP::synapse0x67dc1d0() {
   return (neuron0x67daef0()*-9.92208);
}

double MuonTaggerMLP::synapse0x67dc210() {
   return (neuron0x67db230()*26.1899);
}

double MuonTaggerMLP::synapse0x67dc590() {
   return (neuron0x67da870()*-4.73569);
}

double MuonTaggerMLP::synapse0x67dc5d0() {
   return (neuron0x67dabb0()*0.0406085);
}

double MuonTaggerMLP::synapse0x67dc610() {
   return (neuron0x67daef0()*2.35069);
}

double MuonTaggerMLP::synapse0x67dc650() {
   return (neuron0x67db230()*-8.46218);
}

double MuonTaggerMLP::synapse0x67dc9d0() {
   return (neuron0x67db6a0()*-0.273679);
}

double MuonTaggerMLP::synapse0x677bb20() {
   return (neuron0x67db9d0()*0.379686);
}

double MuonTaggerMLP::synapse0x6745bb0() {
   return (neuron0x67dbe10()*0.878413);
}

double MuonTaggerMLP::synapse0x6745bf0() {
   return (neuron0x67dc250()*-1.015);
}

