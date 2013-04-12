#include "MuonTaggerMLP.h"
#include <cmath>

double MuonTaggerMLP::value(int index,double in0,double in1,double in2,double in3) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   switch(index) {
     case 0:
         return neuron0x16f7a880();
     default:
         return 0.;
   }
}

double MuonTaggerMLP::value(int index, double* input) {
   input0 = (input[0] - 0)/1;
   input1 = (input[1] - 0)/1;
   input2 = (input[2] - 0)/1;
   input3 = (input[3] - 0)/1;
   switch(index) {
     case 0:
         return neuron0x16f7a880();
     default:
         return 0.;
   }
}

double MuonTaggerMLP::neuron0x16f30a90() {
   return input0;
}

double MuonTaggerMLP::neuron0x16f30dd0() {
   return input1;
}

double MuonTaggerMLP::neuron0x16f6f2a0() {
   return input2;
}

double MuonTaggerMLP::neuron0x16f6f5e0() {
   return input3;
}

double MuonTaggerMLP::input0x1711c4b0() {
   double input = -10.3587;
   input += synapse0x16f06360();
   input += synapse0x16f09170();
   input += synapse0x1711d270();
   input += synapse0x171084a0();
   return input;
}

double MuonTaggerMLP::neuron0x1711c4b0() {
   double input = input0x1711c4b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double MuonTaggerMLP::input0x1711c760() {
   double input = 7.63154;
   input += synapse0x16f7a340();
   input += synapse0x16f7a380();
   input += synapse0x16f7a3c0();
   input += synapse0x16f7a400();
   return input;
}

double MuonTaggerMLP::neuron0x1711c760() {
   double input = input0x1711c760();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double MuonTaggerMLP::input0x16f7a440() {
   double input = -6.84926;
   input += synapse0x16f7a780();
   input += synapse0x16f7a7c0();
   input += synapse0x16f7a800();
   input += synapse0x16f7a840();
   return input;
}

double MuonTaggerMLP::neuron0x16f7a440() {
   double input = input0x16f7a440();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double MuonTaggerMLP::input0x16f7a880() {
   double input = -1.22461;
   input += synapse0x16f7abc0();
   input += synapse0x16f7ac00();
   input += synapse0x16f7ac40();
   return input;
}

double MuonTaggerMLP::neuron0x16f7a880() {
   double input = input0x16f7a880();
   return (input * 1)+0;
}

double MuonTaggerMLP::synapse0x16f06360() {
   return (neuron0x16f30a90()*-8.97054);
}

double MuonTaggerMLP::synapse0x16f09170() {
   return (neuron0x16f30dd0()*20.3888);
}

double MuonTaggerMLP::synapse0x1711d270() {
   return (neuron0x16f6f2a0()*18.6933);
}

double MuonTaggerMLP::synapse0x171084a0() {
   return (neuron0x16f6f5e0()*-6.20455);
}

double MuonTaggerMLP::synapse0x16f7a340() {
   return (neuron0x16f30a90()*4.93426);
}

double MuonTaggerMLP::synapse0x16f7a380() {
   return (neuron0x16f30dd0()*-11.2857);
}

double MuonTaggerMLP::synapse0x16f7a3c0() {
   return (neuron0x16f6f2a0()*-11.2742);
}

double MuonTaggerMLP::synapse0x16f7a400() {
   return (neuron0x16f6f5e0()*1.92205);
}

double MuonTaggerMLP::synapse0x16f7a780() {
   return (neuron0x16f30a90()*-5.12296);
}

double MuonTaggerMLP::synapse0x16f7a7c0() {
   return (neuron0x16f30dd0()*0.0021079);
}

double MuonTaggerMLP::synapse0x16f7a800() {
   return (neuron0x16f6f2a0()*10.6783);
}

double MuonTaggerMLP::synapse0x16f7a840() {
   return (neuron0x16f6f5e0()*0.416692);
}

double MuonTaggerMLP::synapse0x16f7abc0() {
   return (neuron0x1711c4b0()*1.28848);
}

double MuonTaggerMLP::synapse0x16f7ac00() {
   return (neuron0x1711c760()*1.2765);
}

double MuonTaggerMLP::synapse0x16f7ac40() {
   return (neuron0x16f7a440()*0.888399);
}

