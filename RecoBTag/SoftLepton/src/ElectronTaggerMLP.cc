#include "ElectronTaggerMLP.h"
#include <cmath>

double ElectronTaggerMLP::value(int index,double in0,double in1,double in2,double in3) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   switch(index) {
     case 0:
         return ((neuron0x9e6bf10()*1)+0);
     default:
         return 0.;
   }
}

double ElectronTaggerMLP::neuron0x9dfc2f0() {
   return input0;
}

double ElectronTaggerMLP::neuron0x9dfc438() {
   return input1;
}

double ElectronTaggerMLP::neuron0x9e6bb10() {
   return input2;
}

double ElectronTaggerMLP::neuron0x9e6bd10() {
   return input3;
}

double ElectronTaggerMLP::input0x9e6c030() {
   double input = 0.366429;
   input += synapse0x9e1f488();
   input += synapse0x9e1f3d0();
   input += synapse0x9ddd668();
   input += synapse0x9ddd6a8();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6c030() {
   double input = input0x9e6c030();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronTaggerMLP::input0x9e6c1e0() {
   double input = -5.97521;
   input += synapse0x9e6c3d8();
   input += synapse0x9e6c400();
   input += synapse0x9e6c428();
   input += synapse0x9e6c450();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6c1e0() {
   double input = input0x9e6c1e0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronTaggerMLP::input0x9e6c478() {
   double input = -0.604356;
   input += synapse0x9e6c670();
   input += synapse0x9e6c698();
   input += synapse0x9e6c6c0();
   input += synapse0x9e6c6e8();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6c478() {
   double input = input0x9e6c478();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronTaggerMLP::input0x9e6c710() {
   double input = 0.491337;
   input += synapse0x9e6c908();
   input += synapse0x9e6c930();
   input += synapse0x9e6c958();
   input += synapse0x9e6c980();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6c710() {
   double input = input0x9e6c710();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronTaggerMLP::input0x9e6c9a8() {
   double input = -2.42706;
   input += synapse0x9e6cba0();
   input += synapse0x9e6cc50();
   input += synapse0x9e6cc78();
   input += synapse0x9e6cca0();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6c9a8() {
   double input = input0x9e6c9a8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronTaggerMLP::input0x9e6ccc8() {
   double input = -4.0112;
   input += synapse0x9e6ce78();
   input += synapse0x9e6cea0();
   input += synapse0x9e6cec8();
   input += synapse0x9e6cef0();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6ccc8() {
   double input = input0x9e6ccc8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronTaggerMLP::input0x9e6cf18() {
   double input = -1.80822;
   input += synapse0x9e6d110();
   input += synapse0x9e6d138();
   input += synapse0x9e6d160();
   input += synapse0x9e6d188();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6cf18() {
   double input = input0x9e6cf18();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronTaggerMLP::input0x9e6d1b0() {
   double input = -0.0152699;
   input += synapse0x9e6d3a8();
   input += synapse0x9e6d3d0();
   input += synapse0x9e6d3f8();
   input += synapse0x9e6d420();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6d1b0() {
   double input = input0x9e6d1b0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronTaggerMLP::input0x9e6bf10() {
   double input = 0.911885;
   input += synapse0x9e1f4d8();
   input += synapse0x9ddd4f8();
   input += synapse0x9dfc680();
   input += synapse0x9dfc6a8();
   input += synapse0x9dfc6d0();
   input += synapse0x9e6cbc8();
   input += synapse0x9e6cbf0();
   input += synapse0x9e6cc18();
   return input;
}

double ElectronTaggerMLP::neuron0x9e6bf10() {
   double input = input0x9e6bf10();
   return (input * 1)+0;
}

double ElectronTaggerMLP::synapse0x9e1f488() {
   return (neuron0x9dfc2f0()*-0.571591);
}

double ElectronTaggerMLP::synapse0x9e1f3d0() {
   return (neuron0x9dfc438()*1.42567);
}

double ElectronTaggerMLP::synapse0x9ddd668() {
   return (neuron0x9e6bb10()*2.05304);
}

double ElectronTaggerMLP::synapse0x9ddd6a8() {
   return (neuron0x9e6bd10()*-2.98556);
}

double ElectronTaggerMLP::synapse0x9e6c3d8() {
   return (neuron0x9dfc2f0()*3.83393);
}

double ElectronTaggerMLP::synapse0x9e6c400() {
   return (neuron0x9dfc438()*0.251168);
}

double ElectronTaggerMLP::synapse0x9e6c428() {
   return (neuron0x9e6bb10()*6.98154);
}

double ElectronTaggerMLP::synapse0x9e6c450() {
   return (neuron0x9e6bd10()*0.632472);
}

double ElectronTaggerMLP::synapse0x9e6c670() {
   return (neuron0x9dfc2f0()*0.32016);
}

double ElectronTaggerMLP::synapse0x9e6c698() {
   return (neuron0x9dfc438()*-0.032084);
}

double ElectronTaggerMLP::synapse0x9e6c6c0() {
   return (neuron0x9e6bb10()*6.7486);
}

double ElectronTaggerMLP::synapse0x9e6c6e8() {
   return (neuron0x9e6bd10()*3.63591);
}

double ElectronTaggerMLP::synapse0x9e6c908() {
   return (neuron0x9dfc2f0()*0.348717);
}

double ElectronTaggerMLP::synapse0x9e6c930() {
   return (neuron0x9dfc438()*-0.57267);
}

double ElectronTaggerMLP::synapse0x9e6c958() {
   return (neuron0x9e6bb10()*-1.44255);
}

double ElectronTaggerMLP::synapse0x9e6c980() {
   return (neuron0x9e6bd10()*0.35619);
}

double ElectronTaggerMLP::synapse0x9e6cba0() {
   return (neuron0x9dfc2f0()*-0.720165);
}

double ElectronTaggerMLP::synapse0x9e6cc50() {
   return (neuron0x9dfc438()*-0.810228);
}

double ElectronTaggerMLP::synapse0x9e6cc78() {
   return (neuron0x9e6bb10()*3.67346);
}

double ElectronTaggerMLP::synapse0x9e6cca0() {
   return (neuron0x9e6bd10()*0.875474);
}

double ElectronTaggerMLP::synapse0x9e6ce78() {
   return (neuron0x9dfc2f0()*-0.051992);
}

double ElectronTaggerMLP::synapse0x9e6cea0() {
   return (neuron0x9dfc438()*-0.141767);
}

double ElectronTaggerMLP::synapse0x9e6cec8() {
   return (neuron0x9e6bb10()*-7.23828);
}

double ElectronTaggerMLP::synapse0x9e6cef0() {
   return (neuron0x9e6bd10()*8.05255);
}

double ElectronTaggerMLP::synapse0x9e6d110() {
   return (neuron0x9dfc2f0()*-0.925528);
}

double ElectronTaggerMLP::synapse0x9e6d138() {
   return (neuron0x9dfc438()*-0.000121187);
}

double ElectronTaggerMLP::synapse0x9e6d160() {
   return (neuron0x9e6bb10()*11.1898);
}

double ElectronTaggerMLP::synapse0x9e6d188() {
   return (neuron0x9e6bd10()*1.85754);
}

double ElectronTaggerMLP::synapse0x9e6d3a8() {
   return (neuron0x9dfc2f0()*0.430314);
}

double ElectronTaggerMLP::synapse0x9e6d3d0() {
   return (neuron0x9dfc438()*-0.0365016);
}

double ElectronTaggerMLP::synapse0x9e6d3f8() {
   return (neuron0x9e6bb10()*-0.554831);
}

double ElectronTaggerMLP::synapse0x9e6d420() {
   return (neuron0x9e6bd10()*1.79062);
}

double ElectronTaggerMLP::synapse0x9e1f4d8() {
   return (neuron0x9e6c030()*-0.172724);
}

double ElectronTaggerMLP::synapse0x9ddd4f8() {
   return (neuron0x9e6c1e0()*0.262318);
}

double ElectronTaggerMLP::synapse0x9dfc680() {
   return (neuron0x9e6c478()*1.57443);
}

double ElectronTaggerMLP::synapse0x9dfc6a8() {
   return (neuron0x9e6c710()*-0.738794);
}

double ElectronTaggerMLP::synapse0x9dfc6d0() {
   return (neuron0x9e6c9a8()*0.465604);
}

double ElectronTaggerMLP::synapse0x9e6cbc8() {
   return (neuron0x9e6ccc8()*0.532105);
}

double ElectronTaggerMLP::synapse0x9e6cbf0() {
   return (neuron0x9e6cf18()*-1.21725);
}

double ElectronTaggerMLP::synapse0x9e6cc18() {
   return (neuron0x9e6d1b0()*-1.71053);
}

