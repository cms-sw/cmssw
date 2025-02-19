#include "ElectronIdMLP.h"
#include <cmath>

double ElectronIdMLP::value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7,double in8) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   input4 = (in4 - 0)/1;
   input5 = (in5 - 0)/1;
   input6 = (in6 - 0)/1;
   input7 = (in7 - 0)/1;
   input8 = (in8 - 0)/1;
   switch(index) {
     case 0:
         return ((neuron0x9f46aa8()*1)+0);
     default:
         return 0.;
   }
}

double ElectronIdMLP::neuron0x9f43138() {
   return input0;
}

double ElectronIdMLP::neuron0x9f432c8() {
   return input1;
}

double ElectronIdMLP::neuron0x9f434a0() {
   return input2;
}

double ElectronIdMLP::neuron0x9f43698() {
   return input3;
}

double ElectronIdMLP::neuron0x9f43870() {
   return input4;
}

double ElectronIdMLP::neuron0x9f43a48() {
   return input5;
}

double ElectronIdMLP::neuron0x9f43c20() {
   return input6;
}

double ElectronIdMLP::neuron0x9f43e10() {
   return input7;
}

double ElectronIdMLP::neuron0x9f44008() {
   return input8;
}

double ElectronIdMLP::input0x9f44320() {
   double input = -1.22145;
   input += synapse0x9f24e98();
   input += synapse0x9f444b0();
   input += synapse0x9f444d8();
   input += synapse0x9f44500();
   input += synapse0x9f44528();
   input += synapse0x9f44550();
   input += synapse0x9f44578();
   input += synapse0x9f445a0();
   input += synapse0x9f445c8();
   return input;
}

double ElectronIdMLP::neuron0x9f44320() {
   double input = input0x9f44320();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f445f0() {
   double input = -2.3169;
   input += synapse0x9f447c8();
   input += synapse0x9f447f0();
   input += synapse0x9f44818();
   input += synapse0x9f44840();
   input += synapse0x9f44868();
   input += synapse0x9f44890();
   input += synapse0x9f448b8();
   input += synapse0x9f448e0();
   input += synapse0x9f44990();
   return input;
}

double ElectronIdMLP::neuron0x9f445f0() {
   double input = input0x9f445f0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f449b8() {
   double input = 0.497047;
   input += synapse0x9f44b48();
   input += synapse0x9f44b70();
   input += synapse0x9f44b98();
   input += synapse0x9f44bc0();
   input += synapse0x9f44be8();
   input += synapse0x9f44c10();
   input += synapse0x9f44c38();
   input += synapse0x9f44c60();
   input += synapse0x9f44c88();
   return input;
}

double ElectronIdMLP::neuron0x9f449b8() {
   double input = input0x9f449b8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f44cb0() {
   double input = -3.54065;
   input += synapse0x9f44e88();
   input += synapse0x9f44eb0();
   input += synapse0x9f44ed8();
   input += synapse0x9f44f00();
   input += synapse0x9f44f28();
   input += synapse0x9f44f50();
   input += synapse0x9f44908();
   input += synapse0x9f44930();
   input += synapse0x9f44958();
   return input;
}

double ElectronIdMLP::neuron0x9f44cb0() {
   double input = input0x9f44cb0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f45080() {
   double input = -2.10164;
   input += synapse0x9f45258();
   input += synapse0x9f45280();
   input += synapse0x9f452a8();
   input += synapse0x9f452d0();
   input += synapse0x9f452f8();
   input += synapse0x9f45320();
   input += synapse0x9f45348();
   input += synapse0x9f45370();
   input += synapse0x9f45398();
   return input;
}

double ElectronIdMLP::neuron0x9f45080() {
   double input = input0x9f45080();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f453c0() {
   double input = 0.540837;
   input += synapse0x9f455b8();
   input += synapse0x9f455e0();
   input += synapse0x9f45608();
   input += synapse0x9f45630();
   input += synapse0x9f45658();
   input += synapse0x9f45680();
   input += synapse0x9f456a8();
   input += synapse0x9f456d0();
   input += synapse0x9f456f8();
   return input;
}

double ElectronIdMLP::neuron0x9f453c0() {
   double input = input0x9f453c0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f45720() {
   double input = 0.568144;
   input += synapse0x9f45918();
   input += synapse0x9f45940();
   input += synapse0x9f45968();
   input += synapse0x9f45990();
   input += synapse0x9f459b8();
   input += synapse0x9f459e0();
   input += synapse0x9f45a08();
   input += synapse0x9f45a30();
   input += synapse0x9f45a58();
   return input;
}

double ElectronIdMLP::neuron0x9f45720() {
   double input = input0x9f45720();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f45a80() {
   double input = -0.779783;
   input += synapse0x9f45d00();
   input += synapse0x9f45d28();
   input += synapse0x9f24ec0();
   input += synapse0x9ea6a58();
   input += synapse0x9ea69e0();
   input += synapse0x9ca2f20();
   input += synapse0x9e985c0();
   input += synapse0x9e985e8();
   input += synapse0x9e98610();
   return input;
}

double ElectronIdMLP::neuron0x9f45a80() {
   double input = input0x9f45a80();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f44f78() {
   double input = 0.146916;
   input += synapse0x9ea6b70();
   input += synapse0x9f45f58();
   input += synapse0x9f45f80();
   input += synapse0x9f45fa8();
   input += synapse0x9f45fd0();
   input += synapse0x9f45ff8();
   input += synapse0x9f46020();
   input += synapse0x9f46048();
   input += synapse0x9f46070();
   return input;
}

double ElectronIdMLP::neuron0x9f44f78() {
   double input = input0x9f44f78();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f46098() {
   double input = -0.350685;
   input += synapse0x9f46270();
   input += synapse0x9f46298();
   input += synapse0x9f462c0();
   input += synapse0x9f462e8();
   input += synapse0x9f46310();
   input += synapse0x9f46338();
   input += synapse0x9f46360();
   input += synapse0x9f46388();
   input += synapse0x9f463b0();
   return input;
}

double ElectronIdMLP::neuron0x9f46098() {
   double input = input0x9f46098();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f463d8() {
   double input = 1.71591;
   input += synapse0x9f465d8();
   input += synapse0x9f46600();
   input += synapse0x9f46628();
   input += synapse0x9f46650();
   input += synapse0x9f46678();
   input += synapse0x9f466a0();
   input += synapse0x9f466c8();
   input += synapse0x9f466f0();
   input += synapse0x9f46718();
   return input;
}

double ElectronIdMLP::neuron0x9f463d8() {
   double input = input0x9f463d8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f46740() {
   double input = -1.70461;
   input += synapse0x9f46940();
   input += synapse0x9f46968();
   input += synapse0x9f46990();
   input += synapse0x9f469b8();
   input += synapse0x9f469e0();
   input += synapse0x9f46a08();
   input += synapse0x9f46a30();
   input += synapse0x9f46a58();
   input += synapse0x9f46a80();
   return input;
}

double ElectronIdMLP::neuron0x9f46740() {
   double input = input0x9f46740();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double ElectronIdMLP::input0x9f46aa8() {
   double input = 0.345943;
   input += synapse0x9f46ca0();
   input += synapse0x9f46cc8();
   input += synapse0x9f46cf0();
   input += synapse0x9f46d18();
   input += synapse0x9f46d40();
   input += synapse0x9f46d68();
   input += synapse0x9f46d90();
   input += synapse0x9f46db8();
   input += synapse0x9f46de0();
   input += synapse0x9f46e08();
   input += synapse0x9f46e30();
   input += synapse0x9f46e58();
   return input;
}

double ElectronIdMLP::neuron0x9f46aa8() {
   double input = input0x9f46aa8();
   return (input * 1)+0;
}

double ElectronIdMLP::synapse0x9f24e98() {
   return (neuron0x9f43138()*0.879971);
}

double ElectronIdMLP::synapse0x9f444b0() {
   return (neuron0x9f432c8()*0.0961671);
}

double ElectronIdMLP::synapse0x9f444d8() {
   return (neuron0x9f434a0()*1.66153);
}

double ElectronIdMLP::synapse0x9f44500() {
   return (neuron0x9f43698()*0.321611);
}

double ElectronIdMLP::synapse0x9f44528() {
   return (neuron0x9f43870()*1.12196);
}

double ElectronIdMLP::synapse0x9f44550() {
   return (neuron0x9f43a48()*-5.17401);
}

double ElectronIdMLP::synapse0x9f44578() {
   return (neuron0x9f43c20()*4.4513);
}

double ElectronIdMLP::synapse0x9f445a0() {
   return (neuron0x9f43e10()*-1.27001);
}

double ElectronIdMLP::synapse0x9f445c8() {
   return (neuron0x9f44008()*-10.3839);
}

double ElectronIdMLP::synapse0x9f447c8() {
   return (neuron0x9f43138()*0.284838);
}

double ElectronIdMLP::synapse0x9f447f0() {
   return (neuron0x9f432c8()*0.0613191);
}

double ElectronIdMLP::synapse0x9f44818() {
   return (neuron0x9f434a0()*0.0915991);
}

double ElectronIdMLP::synapse0x9f44840() {
   return (neuron0x9f43698()*-0.549085);
}

double ElectronIdMLP::synapse0x9f44868() {
   return (neuron0x9f43870()*-0.388214);
}

double ElectronIdMLP::synapse0x9f44890() {
   return (neuron0x9f43a48()*1.82147);
}

double ElectronIdMLP::synapse0x9f448b8() {
   return (neuron0x9f43c20()*4.79833);
}

double ElectronIdMLP::synapse0x9f448e0() {
   return (neuron0x9f43e10()*-1.77169);
}

double ElectronIdMLP::synapse0x9f44990() {
   return (neuron0x9f44008()*-3.25212);
}

double ElectronIdMLP::synapse0x9f44b48() {
   return (neuron0x9f43138()*1.08682);
}

double ElectronIdMLP::synapse0x9f44b70() {
   return (neuron0x9f432c8()*0.0571666);
}

double ElectronIdMLP::synapse0x9f44b98() {
   return (neuron0x9f434a0()*-1.90535);
}

double ElectronIdMLP::synapse0x9f44bc0() {
   return (neuron0x9f43698()*-2.15635);
}

double ElectronIdMLP::synapse0x9f44be8() {
   return (neuron0x9f43870()*-1.183);
}

double ElectronIdMLP::synapse0x9f44c10() {
   return (neuron0x9f43a48()*-1.53323);
}

double ElectronIdMLP::synapse0x9f44c38() {
   return (neuron0x9f43c20()*1.59348);
}

double ElectronIdMLP::synapse0x9f44c60() {
   return (neuron0x9f43e10()*-0.834021);
}

double ElectronIdMLP::synapse0x9f44c88() {
   return (neuron0x9f44008()*-1.08357);
}

double ElectronIdMLP::synapse0x9f44e88() {
   return (neuron0x9f43138()*-0.959033);
}

double ElectronIdMLP::synapse0x9f44eb0() {
   return (neuron0x9f432c8()*0.150702);
}

double ElectronIdMLP::synapse0x9f44ed8() {
   return (neuron0x9f434a0()*0.603047);
}

double ElectronIdMLP::synapse0x9f44f00() {
   return (neuron0x9f43698()*0.766931);
}

double ElectronIdMLP::synapse0x9f44f28() {
   return (neuron0x9f43870()*1.33327);
}

double ElectronIdMLP::synapse0x9f44f50() {
   return (neuron0x9f43a48()*0.397085);
}

double ElectronIdMLP::synapse0x9f44908() {
   return (neuron0x9f43c20()*2.83518);
}

double ElectronIdMLP::synapse0x9f44930() {
   return (neuron0x9f43e10()*2.53477);
}

double ElectronIdMLP::synapse0x9f44958() {
   return (neuron0x9f44008()*-1.20442);
}

double ElectronIdMLP::synapse0x9f45258() {
   return (neuron0x9f43138()*-0.0646114);
}

double ElectronIdMLP::synapse0x9f45280() {
   return (neuron0x9f432c8()*-0.330255);
}

double ElectronIdMLP::synapse0x9f452a8() {
   return (neuron0x9f434a0()*-1.87624);
}

double ElectronIdMLP::synapse0x9f452d0() {
   return (neuron0x9f43698()*-0.872878);
}

double ElectronIdMLP::synapse0x9f452f8() {
   return (neuron0x9f43870()*0.789821);
}

double ElectronIdMLP::synapse0x9f45320() {
   return (neuron0x9f43a48()*2.97713);
}

double ElectronIdMLP::synapse0x9f45348() {
   return (neuron0x9f43c20()*3.70241);
}

double ElectronIdMLP::synapse0x9f45370() {
   return (neuron0x9f43e10()*2.25345);
}

double ElectronIdMLP::synapse0x9f45398() {
   return (neuron0x9f44008()*9.12485);
}

double ElectronIdMLP::synapse0x9f455b8() {
   return (neuron0x9f43138()*-1.76594);
}

double ElectronIdMLP::synapse0x9f455e0() {
   return (neuron0x9f432c8()*0.325989);
}

double ElectronIdMLP::synapse0x9f45608() {
   return (neuron0x9f434a0()*-1.57176);
}

double ElectronIdMLP::synapse0x9f45630() {
   return (neuron0x9f43698()*-1.49181);
}

double ElectronIdMLP::synapse0x9f45658() {
   return (neuron0x9f43870()*-0.436626);
}

double ElectronIdMLP::synapse0x9f45680() {
   return (neuron0x9f43a48()*-0.415296);
}

double ElectronIdMLP::synapse0x9f456a8() {
   return (neuron0x9f43c20()*0.686166);
}

double ElectronIdMLP::synapse0x9f456d0() {
   return (neuron0x9f43e10()*1.197);
}

double ElectronIdMLP::synapse0x9f456f8() {
   return (neuron0x9f44008()*1.08228);
}

double ElectronIdMLP::synapse0x9f45918() {
   return (neuron0x9f43138()*0.6264);
}

double ElectronIdMLP::synapse0x9f45940() {
   return (neuron0x9f432c8()*-0.0527889);
}

double ElectronIdMLP::synapse0x9f45968() {
   return (neuron0x9f434a0()*0.0420227);
}

double ElectronIdMLP::synapse0x9f45990() {
   return (neuron0x9f43698()*-0.503423);
}

double ElectronIdMLP::synapse0x9f459b8() {
   return (neuron0x9f43870()*0.015745);
}

double ElectronIdMLP::synapse0x9f459e0() {
   return (neuron0x9f43a48()*-2.27786);
}

double ElectronIdMLP::synapse0x9f45a08() {
   return (neuron0x9f43c20()*-2.78804);
}

double ElectronIdMLP::synapse0x9f45a30() {
   return (neuron0x9f43e10()*0.278226);
}

double ElectronIdMLP::synapse0x9f45a58() {
   return (neuron0x9f44008()*1.43897);
}

double ElectronIdMLP::synapse0x9f45d00() {
   return (neuron0x9f43138()*-0.113819);
}

double ElectronIdMLP::synapse0x9f45d28() {
   return (neuron0x9f432c8()*-0.526026);
}

double ElectronIdMLP::synapse0x9f24ec0() {
   return (neuron0x9f434a0()*0.280854);
}

double ElectronIdMLP::synapse0x9ea6a58() {
   return (neuron0x9f43698()*-0.565832);
}

double ElectronIdMLP::synapse0x9ea69e0() {
   return (neuron0x9f43870()*-4.1776);
}

double ElectronIdMLP::synapse0x9ca2f20() {
   return (neuron0x9f43a48()*2.4344);
}

double ElectronIdMLP::synapse0x9e985c0() {
   return (neuron0x9f43c20()*1.19602);
}

double ElectronIdMLP::synapse0x9e985e8() {
   return (neuron0x9f43e10()*3.02645);
}

double ElectronIdMLP::synapse0x9e98610() {
   return (neuron0x9f44008()*-0.38682);
}

double ElectronIdMLP::synapse0x9ea6b70() {
   return (neuron0x9f43138()*-2.22478);
}

double ElectronIdMLP::synapse0x9f45f58() {
   return (neuron0x9f432c8()*0.203476);
}

double ElectronIdMLP::synapse0x9f45f80() {
   return (neuron0x9f434a0()*2.45698);
}

double ElectronIdMLP::synapse0x9f45fa8() {
   return (neuron0x9f43698()*0.938005);
}

double ElectronIdMLP::synapse0x9f45fd0() {
   return (neuron0x9f43870()*-0.155581);
}

double ElectronIdMLP::synapse0x9f45ff8() {
   return (neuron0x9f43a48()*-1.84449);
}

double ElectronIdMLP::synapse0x9f46020() {
   return (neuron0x9f43c20()*1.62976);
}

double ElectronIdMLP::synapse0x9f46048() {
   return (neuron0x9f43e10()*-1.3476);
}

double ElectronIdMLP::synapse0x9f46070() {
   return (neuron0x9f44008()*-5.67653);
}

double ElectronIdMLP::synapse0x9f46270() {
   return (neuron0x9f43138()*-1.1783);
}

double ElectronIdMLP::synapse0x9f46298() {
   return (neuron0x9f432c8()*0.919324);
}

double ElectronIdMLP::synapse0x9f462c0() {
   return (neuron0x9f434a0()*0.100647);
}

double ElectronIdMLP::synapse0x9f462e8() {
   return (neuron0x9f43698()*1.14597);
}

double ElectronIdMLP::synapse0x9f46310() {
   return (neuron0x9f43870()*-0.270844);
}

double ElectronIdMLP::synapse0x9f46338() {
   return (neuron0x9f43a48()*0.982664);
}

double ElectronIdMLP::synapse0x9f46360() {
   return (neuron0x9f43c20()*1.50571);
}

double ElectronIdMLP::synapse0x9f46388() {
   return (neuron0x9f43e10()*-1.14533);
}

double ElectronIdMLP::synapse0x9f463b0() {
   return (neuron0x9f44008()*4.69291);
}

double ElectronIdMLP::synapse0x9f465d8() {
   return (neuron0x9f43138()*0.105787);
}

double ElectronIdMLP::synapse0x9f46600() {
   return (neuron0x9f432c8()*0.460172);
}

double ElectronIdMLP::synapse0x9f46628() {
   return (neuron0x9f434a0()*0.138901);
}

double ElectronIdMLP::synapse0x9f46650() {
   return (neuron0x9f43698()*1.72831);
}

double ElectronIdMLP::synapse0x9f46678() {
   return (neuron0x9f43870()*-1.36692);
}

double ElectronIdMLP::synapse0x9f466a0() {
   return (neuron0x9f43a48()*3.09794);
}

double ElectronIdMLP::synapse0x9f466c8() {
   return (neuron0x9f43c20()*0.170757);
}

double ElectronIdMLP::synapse0x9f466f0() {
   return (neuron0x9f43e10()*3.36054);
}

double ElectronIdMLP::synapse0x9f46718() {
   return (neuron0x9f44008()*5.85824);
}

double ElectronIdMLP::synapse0x9f46940() {
   return (neuron0x9f43138()*0.77873);
}

double ElectronIdMLP::synapse0x9f46968() {
   return (neuron0x9f432c8()*0.0264436);
}

double ElectronIdMLP::synapse0x9f46990() {
   return (neuron0x9f434a0()*0.86123);
}

double ElectronIdMLP::synapse0x9f469b8() {
   return (neuron0x9f43698()*1.8242);
}

double ElectronIdMLP::synapse0x9f469e0() {
   return (neuron0x9f43870()*3.66502);
}

double ElectronIdMLP::synapse0x9f46a08() {
   return (neuron0x9f43a48()*-0.182076);
}

double ElectronIdMLP::synapse0x9f46a30() {
   return (neuron0x9f43c20()*0.59582);
}

double ElectronIdMLP::synapse0x9f46a58() {
   return (neuron0x9f43e10()*0.667454);
}

double ElectronIdMLP::synapse0x9f46a80() {
   return (neuron0x9f44008()*1.1529);
}

double ElectronIdMLP::synapse0x9f46ca0() {
   return (neuron0x9f44320()*-3.27612);
}

double ElectronIdMLP::synapse0x9f46cc8() {
   return (neuron0x9f445f0()*-2.11621);
}

double ElectronIdMLP::synapse0x9f46cf0() {
   return (neuron0x9f449b8()*-4.88221);
}

double ElectronIdMLP::synapse0x9f46d18() {
   return (neuron0x9f44cb0()*1.89605);
}

double ElectronIdMLP::synapse0x9f46d40() {
   return (neuron0x9f45080()*-1.18343);
}

double ElectronIdMLP::synapse0x9f46d68() {
   return (neuron0x9f453c0()*5.41139);
}

double ElectronIdMLP::synapse0x9f46d90() {
   return (neuron0x9f45720()*-3.83221);
}

double ElectronIdMLP::synapse0x9f46db8() {
   return (neuron0x9f45a80()*-0.743994);
}

double ElectronIdMLP::synapse0x9f46de0() {
   return (neuron0x9f44f78()*7.23992);
}

double ElectronIdMLP::synapse0x9f46e08() {
   return (neuron0x9f46098()*2.98399);
}

double ElectronIdMLP::synapse0x9f46e30() {
   return (neuron0x9f463d8()*-1.92682);
}

double ElectronIdMLP::synapse0x9f46e58() {
   return (neuron0x9f46740()*-3.21289);
}


