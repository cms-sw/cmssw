#include "ElectronTaggerMLP.h"
#include <cmath>

double ElectronTaggerMLP::value(int index,double in0,double in1,double in2,double in3) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   switch(index) {
     case 0:
         return neuron0xe93f7e0();
     default:
         return 0.;
   }
}

double ElectronTaggerMLP::value(int index, double* input) {
   input0 = (input[0] - 0)/1;
   input1 = (input[1] - 0)/1;
   input2 = (input[2] - 0)/1;
   input3 = (input[3] - 0)/1;
   switch(index) {
     case 0:
         return neuron0xe93f7e0();
     default:
         return 0.;
   }
}

double ElectronTaggerMLP::neuron0xe93d1c0() {
   return input0;
}

double ElectronTaggerMLP::neuron0xe93d500() {
   return input1;
}

double ElectronTaggerMLP::neuron0xe93d840() {
   return input2;
}

double ElectronTaggerMLP::neuron0xe93db80() {
   return input3;
}

double ElectronTaggerMLP::input0xe93dff0() {
   double input = -1.08531;
   input += synapse0xe916430();
   input += synapse0xe9162f0();
   input += synapse0xe93e2a0();
   input += synapse0xe93e2e0();
   return input;
}

double ElectronTaggerMLP::neuron0xe93dff0() {
   double input = input0xe93dff0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0xe93e320() {
   double input = 2.22983;
   input += synapse0xe93e660();
   input += synapse0xe93e6a0();
   input += synapse0xe93e6e0();
   input += synapse0xe93e720();
   return input;
}

double ElectronTaggerMLP::neuron0xe93e320() {
   double input = input0xe93e320();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0xe93e760() {
   double input = 0.394642;
   input += synapse0xe93eaa0();
   input += synapse0xe93eae0();
   input += synapse0xe93eb20();
   input += synapse0xe93eb60();
   return input;
}

double ElectronTaggerMLP::neuron0xe93e760() {
   double input = input0xe93e760();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0xe93eba0() {
   double input = 0.39117;
   input += synapse0xe93eee0();
   input += synapse0xe93ef20();
   input += synapse0xe93ef60();
   return input;
}

double ElectronTaggerMLP::neuron0xe93eba0() {
   double input = input0xe93eba0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0xe93efa0() {
   double input = -0.764948;
   input += synapse0xe93f2e0();
   input += synapse0xe93f320();
   input += synapse0xe8de480();
   return input;
}

double ElectronTaggerMLP::neuron0xe93efa0() {
   double input = input0xe93efa0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0xe93f470() {
   double input = -1.0173;
   input += synapse0xe93f720();
   input += synapse0xe93f760();
   input += synapse0xe93f7a0();
   return input;
}

double ElectronTaggerMLP::neuron0xe93f470() {
   double input = input0xe93f470();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ElectronTaggerMLP::input0xe93f7e0() {
   double input = 0.23116;
   input += synapse0xe93fb20();
   input += synapse0xe93fb60();
   input += synapse0xe93fba0();
   return input;
}

double ElectronTaggerMLP::neuron0xe93f7e0() {
   double input = input0xe93f7e0();
   return (input * 1)+0;
}

double ElectronTaggerMLP::synapse0xe916430() {
   return (neuron0xe93d1c0()*-4.1499);
}

double ElectronTaggerMLP::synapse0xe9162f0() {
   return (neuron0xe93d500()*0.0314529);
}

double ElectronTaggerMLP::synapse0xe93e2a0() {
   return (neuron0xe93d840()*2.53892);
}

double ElectronTaggerMLP::synapse0xe93e2e0() {
   return (neuron0xe93db80()*9.39677);
}

double ElectronTaggerMLP::synapse0xe93e660() {
   return (neuron0xe93d1c0()*-0.202208);
}

double ElectronTaggerMLP::synapse0xe93e6a0() {
   return (neuron0xe93d500()*0.533988);
}

double ElectronTaggerMLP::synapse0xe93e6e0() {
   return (neuron0xe93d840()*2.71641);
}

double ElectronTaggerMLP::synapse0xe93e720() {
   return (neuron0xe93db80()*4.59282);
}

double ElectronTaggerMLP::synapse0xe93eaa0() {
   return (neuron0xe93d1c0()*0.0203936);
}

double ElectronTaggerMLP::synapse0xe93eae0() {
   return (neuron0xe93d500()*0.181501);
}

double ElectronTaggerMLP::synapse0xe93eb20() {
   return (neuron0xe93d840()*2.9895);
}

double ElectronTaggerMLP::synapse0xe93eb60() {
   return (neuron0xe93db80()*0.213088);
}

double ElectronTaggerMLP::synapse0xe93eee0() {
   return (neuron0xe93dff0()*0.246792);
}

double ElectronTaggerMLP::synapse0xe93ef20() {
   return (neuron0xe93e320()*0.993431);
}

double ElectronTaggerMLP::synapse0xe93ef60() {
   return (neuron0xe93e760()*0.615563);
}

double ElectronTaggerMLP::synapse0xe93f2e0() {
   return (neuron0xe93dff0()*-6.1435);
}

double ElectronTaggerMLP::synapse0xe93f320() {
   return (neuron0xe93e320()*-5.735);
}

double ElectronTaggerMLP::synapse0xe8de480() {
   return (neuron0xe93e760()*9.33819);
}

double ElectronTaggerMLP::synapse0xe93f720() {
   return (neuron0xe93dff0()*-3.56662);
}

double ElectronTaggerMLP::synapse0xe93f760() {
   return (neuron0xe93e320()*-0.0794513);
}

double ElectronTaggerMLP::synapse0xe93f7a0() {
   return (neuron0xe93e760()*-0.851104);
}

double ElectronTaggerMLP::synapse0xe93fb20() {
   return (neuron0xe93eba0()*-0.274638);
}

double ElectronTaggerMLP::synapse0xe93fb60() {
   return (neuron0xe93efa0()*1.21068);
}

double ElectronTaggerMLP::synapse0xe93fba0() {
   return (neuron0xe93f470()*-1.31265);
}

