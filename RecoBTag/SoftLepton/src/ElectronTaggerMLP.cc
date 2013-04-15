#include "ElectronTaggerMLP.h"
#include <cmath>

double ElectronTaggerMLP::Value(int index,double in0,double in1,double in2,double in3) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   switch(index) {
     case 0:
         return neuron0x222109d0();
     default:
         return 0.;
   }
}

double ElectronTaggerMLP::Value(int index, double* input) {
   input0 = (input[0] - 0)/1;
   input1 = (input[1] - 0)/1;
   input2 = (input[2] - 0)/1;
   input3 = (input[3] - 0)/1;
   switch(index) {
     case 0:
         return neuron0x222109d0();
     default:
         return 0.;
   }
}

double ElectronTaggerMLP::neuron0x2220ebb0() {
   return input0;
}

double ElectronTaggerMLP::neuron0x2220eef0() {
   return input1;
}

double ElectronTaggerMLP::neuron0x2220f230() {
   return input2;
}

double ElectronTaggerMLP::neuron0x2220f570() {
   return input3;
}

double ElectronTaggerMLP::input0x2220f9e0() {
   double input = 0.445654;
   input += synapse0x221e7e20();
   input += synapse0x221e7ce0();
   input += synapse0x2220fc90();
   input += synapse0x2220fcd0();
   return input;
}

double ElectronTaggerMLP::neuron0x2220f9e0() {
   double input = input0x2220f9e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0x2220fd10() {
   double input = 3.0541;
   input += synapse0x22210050();
   input += synapse0x22210090();
   input += synapse0x222100d0();
   input += synapse0x22210110();
   return input;
}

double ElectronTaggerMLP::neuron0x2220fd10() {
   double input = input0x2220fd10();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0x22210150() {
   double input = -0.888504;
   input += synapse0x22210490();
   input += synapse0x222104d0();
   input += synapse0x22210510();
   input += synapse0x22210550();
   return input;
}

double ElectronTaggerMLP::neuron0x22210150() {
   double input = input0x22210150();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0x22210590() {
   double input = 0.469273;
   input += synapse0x222108d0();
   input += synapse0x22210910();
   input += synapse0x22210950();
   input += synapse0x22210990();
   return input;
}

double ElectronTaggerMLP::neuron0x22210590() {
   double input = input0x22210590();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0x222109d0() {
   double input = -1.04287;
   input += synapse0x22210d10();
   input += synapse0x221afe70();
   input += synapse0x221a2dd0();
   input += synapse0x221a2e10();
   return input;
}

double ElectronTaggerMLP::neuron0x222109d0() {
   double input = input0x222109d0();
   return (input * 1)+0;
}

double ElectronTaggerMLP::synapse0x221e7e20() {
   return (neuron0x2220ebb0()*-0.428812);
}

double ElectronTaggerMLP::synapse0x221e7ce0() {
   return (neuron0x2220eef0()*-0.506051);
}

double ElectronTaggerMLP::synapse0x2220fc90() {
   return (neuron0x2220f230()*-0.120691);
}

double ElectronTaggerMLP::synapse0x2220fcd0() {
   return (neuron0x2220f570()*-3.54343);
}

double ElectronTaggerMLP::synapse0x22210050() {
   return (neuron0x2220ebb0()*-0.254167);
}

double ElectronTaggerMLP::synapse0x22210090() {
   return (neuron0x2220eef0()*0.0663286);
}

double ElectronTaggerMLP::synapse0x222100d0() {
   return (neuron0x2220f230()*-4.81951);
}

double ElectronTaggerMLP::synapse0x22210110() {
   return (neuron0x2220f570()*2.201);
}

double ElectronTaggerMLP::synapse0x22210490() {
   return (neuron0x2220ebb0()*-0.237537);
}

double ElectronTaggerMLP::synapse0x222104d0() {
   return (neuron0x2220eef0()*-0.627087);
}

double ElectronTaggerMLP::synapse0x22210510() {
   return (neuron0x2220f230()*-4.22539);
}

double ElectronTaggerMLP::synapse0x22210550() {
   return (neuron0x2220f570()*2.0423);
}

double ElectronTaggerMLP::synapse0x222108d0() {
   return (neuron0x2220ebb0()*3.21231);
}

double ElectronTaggerMLP::synapse0x22210910() {
   return (neuron0x2220eef0()*-0.0201844);
}

double ElectronTaggerMLP::synapse0x22210950() {
   return (neuron0x2220f230()*-4.27204);
}

double ElectronTaggerMLP::synapse0x22210990() {
   return (neuron0x2220f570()*2.08387);
}

double ElectronTaggerMLP::synapse0x22210d10() {
   return (neuron0x2220f9e0()*-1.38387);
}

double ElectronTaggerMLP::synapse0x221afe70() {
   return (neuron0x2220fd10()*0.789103);
}

double ElectronTaggerMLP::synapse0x221a2dd0() {
   return (neuron0x22210150()*1.21951);
}

double ElectronTaggerMLP::synapse0x221a2e10() {
   return (neuron0x22210590()*1.28071);
}

