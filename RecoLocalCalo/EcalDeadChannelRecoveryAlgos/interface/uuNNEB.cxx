#include "uuNNEB.h"
#include <cmath>

double uuNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.15492)/1.68616;
   input2 = (in2 - 1.14956)/1.69316;
   input3 = (in3 - 1.90666)/1.49442;
   input4 = (in4 - 0.3065)/1.46726;
   input5 = (in5 - 0.318454)/1.50742;
   input6 = (in6 - 0.305354)/1.51455;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x215a5090();
     default:
         return 0.;
   }
}

double uuNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.15492)/1.68616;
   input2 = (input[2] - 1.14956)/1.69316;
   input3 = (input[3] - 1.90666)/1.49442;
   input4 = (input[4] - 0.3065)/1.46726;
   input5 = (input[5] - 0.318454)/1.50742;
   input6 = (input[6] - 0.305354)/1.51455;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x215a5090();
     default:
         return 0.;
   }
}

double uuNNEB::neuron0x215a0910() {
   return input0;
}

double uuNNEB::neuron0x215a0c50() {
   return input1;
}

double uuNNEB::neuron0x215a0f90() {
   return input2;
}

double uuNNEB::neuron0x215a12d0() {
   return input3;
}

double uuNNEB::neuron0x215a1610() {
   return input4;
}

double uuNNEB::neuron0x215a1950() {
   return input5;
}

double uuNNEB::neuron0x215a1c90() {
   return input6;
}

double uuNNEB::neuron0x215a1fd0() {
   return input7;
}

double uuNNEB::input0x215a2440() {
   double input = -3.46668;
   input += synapse0x21501310();
   input += synapse0x215a9510();
   input += synapse0x215a26f0();
   input += synapse0x215a2730();
   input += synapse0x215a2770();
   input += synapse0x215a27b0();
   input += synapse0x215a27f0();
   input += synapse0x215a2830();
   return input;
}

double uuNNEB::neuron0x215a2440() {
   double input = input0x215a2440();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a2870() {
   double input = -0.216712;
   input += synapse0x215a2bb0();
   input += synapse0x215a2bf0();
   input += synapse0x215a2c30();
   input += synapse0x215a2c70();
   input += synapse0x215a2cb0();
   input += synapse0x215a2cf0();
   input += synapse0x215a2d30();
   input += synapse0x215a2d70();
   return input;
}

double uuNNEB::neuron0x215a2870() {
   double input = input0x215a2870();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a2db0() {
   double input = -1.25069;
   input += synapse0x215a30f0();
   input += synapse0x20dfffd0();
   input += synapse0x20e00010();
   input += synapse0x215a3240();
   input += synapse0x215a3280();
   input += synapse0x215a32c0();
   input += synapse0x215a3300();
   input += synapse0x215a3340();
   return input;
}

double uuNNEB::neuron0x215a2db0() {
   double input = input0x215a2db0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a3380() {
   double input = 0.164265;
   input += synapse0x215a36c0();
   input += synapse0x215a3700();
   input += synapse0x215a3740();
   input += synapse0x215a3780();
   input += synapse0x215a37c0();
   input += synapse0x215a3800();
   input += synapse0x215a3840();
   input += synapse0x215a3880();
   return input;
}

double uuNNEB::neuron0x215a3380() {
   double input = input0x215a3380();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a38c0() {
   double input = 1.20615;
   input += synapse0x215a3c00();
   input += synapse0x215a0840();
   input += synapse0x215a9550();
   input += synapse0x20dfeb10();
   input += synapse0x215a3130();
   input += synapse0x215a3170();
   input += synapse0x215a31b0();
   input += synapse0x215a31f0();
   return input;
}

double uuNNEB::neuron0x215a38c0() {
   double input = input0x215a38c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a3c40() {
   double input = -1.00109;
   input += synapse0x215a3f80();
   input += synapse0x215a3fc0();
   input += synapse0x215a4000();
   input += synapse0x215a4040();
   input += synapse0x215a4080();
   input += synapse0x215a40c0();
   input += synapse0x215a4100();
   input += synapse0x215a4140();
   return input;
}

double uuNNEB::neuron0x215a3c40() {
   double input = input0x215a3c40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a4180() {
   double input = -1.43638;
   input += synapse0x215a44c0();
   input += synapse0x215a4500();
   input += synapse0x215a4540();
   input += synapse0x215a4580();
   input += synapse0x215a45c0();
   input += synapse0x215a4600();
   input += synapse0x215a4640();
   input += synapse0x215a4680();
   return input;
}

double uuNNEB::neuron0x215a4180() {
   double input = input0x215a4180();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a46c0() {
   double input = 1.30954;
   input += synapse0x215a4a00();
   input += synapse0x215a4a40();
   input += synapse0x215a4a80();
   input += synapse0x215a4ac0();
   input += synapse0x215a4b00();
   input += synapse0x215a4b40();
   input += synapse0x215a4b80();
   input += synapse0x215a4bc0();
   return input;
}

double uuNNEB::neuron0x215a46c0() {
   double input = input0x215a46c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a4c00() {
   double input = -0.75568;
   input += synapse0x20cedf30();
   input += synapse0x20cedf70();
   input += synapse0x20e198e0();
   input += synapse0x20e19920();
   input += synapse0x20e19960();
   input += synapse0x20e199a0();
   input += synapse0x20e199e0();
   input += synapse0x20e19a20();
   return input;
}

double uuNNEB::neuron0x215a4c00() {
   double input = input0x215a4c00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a5460() {
   double input = 0.349564;
   input += synapse0x215a5710();
   input += synapse0x215a5750();
   input += synapse0x215a5790();
   input += synapse0x215a57d0();
   input += synapse0x215a5810();
   input += synapse0x215a5850();
   input += synapse0x215a5890();
   input += synapse0x215a58d0();
   return input;
}

double uuNNEB::neuron0x215a5460() {
   double input = input0x215a5460();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a5910() {
   double input = -2.54272;
   input += synapse0x215a5c50();
   input += synapse0x215a5c90();
   input += synapse0x215a5cd0();
   input += synapse0x215a5d10();
   input += synapse0x215a5d50();
   input += synapse0x215a5d90();
   input += synapse0x215a5dd0();
   input += synapse0x215a5e10();
   input += synapse0x215a5e50();
   input += synapse0x215a5e90();
   return input;
}

double uuNNEB::neuron0x215a5910() {
   double input = input0x215a5910();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a5ed0() {
   double input = -1.6243;
   input += synapse0x215a6210();
   input += synapse0x215a6250();
   input += synapse0x215a6290();
   input += synapse0x215a62d0();
   input += synapse0x215a6310();
   input += synapse0x215a6350();
   input += synapse0x215a6390();
   input += synapse0x215a63d0();
   input += synapse0x215a6410();
   input += synapse0x215a6450();
   return input;
}

double uuNNEB::neuron0x215a5ed0() {
   double input = input0x215a5ed0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a6490() {
   double input = -2.41565;
   input += synapse0x215a67d0();
   input += synapse0x215a6810();
   input += synapse0x215a6850();
   input += synapse0x215a6890();
   input += synapse0x215a68d0();
   input += synapse0x215a6910();
   input += synapse0x215a6950();
   input += synapse0x215a6990();
   input += synapse0x215a69d0();
   input += synapse0x215a6a10();
   return input;
}

double uuNNEB::neuron0x215a6490() {
   double input = input0x215a6490();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a6a50() {
   double input = 1.3051;
   input += synapse0x215a6d90();
   input += synapse0x215a6dd0();
   input += synapse0x215a6e10();
   input += synapse0x215a6e50();
   input += synapse0x215a6e90();
   input += synapse0x215a6ed0();
   input += synapse0x215a6f10();
   input += synapse0x215a6f50();
   input += synapse0x215a6f90();
   input += synapse0x215a6fd0();
   return input;
}

double uuNNEB::neuron0x215a6a50() {
   double input = input0x215a6a50();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a7010() {
   double input = -0.872798;
   input += synapse0x215a7350();
   input += synapse0x215a7390();
   input += synapse0x215a73d0();
   input += synapse0x215a7410();
   input += synapse0x215a7450();
   input += synapse0x215a7490();
   input += synapse0x215a74d0();
   input += synapse0x215a7510();
   input += synapse0x215a7550();
   input += synapse0x215a5050();
   return input;
}

double uuNNEB::neuron0x215a7010() {
   double input = input0x215a7010();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEB::input0x215a5090() {
   double input = -0.827423;
   input += synapse0x215a53d0();
   input += synapse0x215a5410();
   input += synapse0x215a2310();
   input += synapse0x215a2350();
   input += synapse0x215a2390();
   return input;
}

double uuNNEB::neuron0x215a5090() {
   double input = input0x215a5090();
   return (input * 1)+0;
}

double uuNNEB::synapse0x21501310() {
   return (neuron0x215a0910()*1.44277);
}

double uuNNEB::synapse0x215a9510() {
   return (neuron0x215a0c50()*-0.602439);
}

double uuNNEB::synapse0x215a26f0() {
   return (neuron0x215a0f90()*-0.664344);
}

double uuNNEB::synapse0x215a2730() {
   return (neuron0x215a12d0()*0.259659);
}

double uuNNEB::synapse0x215a2770() {
   return (neuron0x215a1610()*0.521977);
}

double uuNNEB::synapse0x215a27b0() {
   return (neuron0x215a1950()*-0.139725);
}

double uuNNEB::synapse0x215a27f0() {
   return (neuron0x215a1c90()*0.462369);
}

double uuNNEB::synapse0x215a2830() {
   return (neuron0x215a1fd0()*-0.144491);
}

double uuNNEB::synapse0x215a2bb0() {
   return (neuron0x215a0910()*-0.690881);
}

double uuNNEB::synapse0x215a2bf0() {
   return (neuron0x215a0c50()*3.00857);
}

double uuNNEB::synapse0x215a2c30() {
   return (neuron0x215a0f90()*0.584154);
}

double uuNNEB::synapse0x215a2c70() {
   return (neuron0x215a12d0()*-0.443833);
}

double uuNNEB::synapse0x215a2cb0() {
   return (neuron0x215a1610()*-1.12004);
}

double uuNNEB::synapse0x215a2cf0() {
   return (neuron0x215a1950()*0.385117);
}

double uuNNEB::synapse0x215a2d30() {
   return (neuron0x215a1c90()*0.828344);
}

double uuNNEB::synapse0x215a2d70() {
   return (neuron0x215a1fd0()*0.0518985);
}

double uuNNEB::synapse0x215a30f0() {
   return (neuron0x215a0910()*2.46324);
}

double uuNNEB::synapse0x20dfffd0() {
   return (neuron0x215a0c50()*-1.27957);
}

double uuNNEB::synapse0x20e00010() {
   return (neuron0x215a0f90()*1.52325);
}

double uuNNEB::synapse0x215a3240() {
   return (neuron0x215a12d0()*0.324036);
}

double uuNNEB::synapse0x215a3280() {
   return (neuron0x215a1610()*-0.772301);
}

double uuNNEB::synapse0x215a32c0() {
   return (neuron0x215a1950()*-0.288646);
}

double uuNNEB::synapse0x215a3300() {
   return (neuron0x215a1c90()*-1.59626);
}

double uuNNEB::synapse0x215a3340() {
   return (neuron0x215a1fd0()*-0.333816);
}

double uuNNEB::synapse0x215a36c0() {
   return (neuron0x215a0910()*-0.775788);
}

double uuNNEB::synapse0x215a3700() {
   return (neuron0x215a0c50()*-0.33891);
}

double uuNNEB::synapse0x215a3740() {
   return (neuron0x215a0f90()*0.413686);
}

double uuNNEB::synapse0x215a3780() {
   return (neuron0x215a12d0()*-0.103716);
}

double uuNNEB::synapse0x215a37c0() {
   return (neuron0x215a1610()*2.59851);
}

double uuNNEB::synapse0x215a3800() {
   return (neuron0x215a1950()*-0.0130854);
}

double uuNNEB::synapse0x215a3840() {
   return (neuron0x215a1c90()*-0.428691);
}

double uuNNEB::synapse0x215a3880() {
   return (neuron0x215a1fd0()*-0.0479283);
}

double uuNNEB::synapse0x215a3c00() {
   return (neuron0x215a0910()*-2.37794);
}

double uuNNEB::synapse0x215a0840() {
   return (neuron0x215a0c50()*-1.41136);
}

double uuNNEB::synapse0x215a9550() {
   return (neuron0x215a0f90()*2.07387);
}

double uuNNEB::synapse0x20dfeb10() {
   return (neuron0x215a12d0()*-0.30988);
}

double uuNNEB::synapse0x215a3130() {
   return (neuron0x215a1610()*1.91446);
}

double uuNNEB::synapse0x215a3170() {
   return (neuron0x215a1950()*-0.220708);
}

double uuNNEB::synapse0x215a31b0() {
   return (neuron0x215a1c90()*-0.809622);
}

double uuNNEB::synapse0x215a31f0() {
   return (neuron0x215a1fd0()*1.13641);
}

double uuNNEB::synapse0x215a3f80() {
   return (neuron0x215a0910()*2.03498);
}

double uuNNEB::synapse0x215a3fc0() {
   return (neuron0x215a0c50()*1.50096);
}

double uuNNEB::synapse0x215a4000() {
   return (neuron0x215a0f90()*-0.0779476);
}

double uuNNEB::synapse0x215a4040() {
   return (neuron0x215a12d0()*-1.07668);
}

double uuNNEB::synapse0x215a4080() {
   return (neuron0x215a1610()*-1.85923);
}

double uuNNEB::synapse0x215a40c0() {
   return (neuron0x215a1950()*-1.04043);
}

double uuNNEB::synapse0x215a4100() {
   return (neuron0x215a1c90()*0.36375);
}

double uuNNEB::synapse0x215a4140() {
   return (neuron0x215a1fd0()*-0.367085);
}

double uuNNEB::synapse0x215a44c0() {
   return (neuron0x215a0910()*-0.727066);
}

double uuNNEB::synapse0x215a4500() {
   return (neuron0x215a0c50()*-1.0188);
}

double uuNNEB::synapse0x215a4540() {
   return (neuron0x215a0f90()*0.468547);
}

double uuNNEB::synapse0x215a4580() {
   return (neuron0x215a12d0()*-0.405341);
}

double uuNNEB::synapse0x215a45c0() {
   return (neuron0x215a1610()*1.12725);
}

double uuNNEB::synapse0x215a4600() {
   return (neuron0x215a1950()*-0.0173553);
}

double uuNNEB::synapse0x215a4640() {
   return (neuron0x215a1c90()*-1.04423);
}

double uuNNEB::synapse0x215a4680() {
   return (neuron0x215a1fd0()*0.532221);
}

double uuNNEB::synapse0x215a4a00() {
   return (neuron0x215a0910()*2.09286);
}

double uuNNEB::synapse0x215a4a40() {
   return (neuron0x215a0c50()*-0.466192);
}

double uuNNEB::synapse0x215a4a80() {
   return (neuron0x215a0f90()*0.37012);
}

double uuNNEB::synapse0x215a4ac0() {
   return (neuron0x215a12d0()*-0.436637);
}

double uuNNEB::synapse0x215a4b00() {
   return (neuron0x215a1610()*0.0712444);
}

double uuNNEB::synapse0x215a4b40() {
   return (neuron0x215a1950()*-0.309679);
}

double uuNNEB::synapse0x215a4b80() {
   return (neuron0x215a1c90()*-1.06154);
}

double uuNNEB::synapse0x215a4bc0() {
   return (neuron0x215a1fd0()*0.0765883);
}

double uuNNEB::synapse0x20cedf30() {
   return (neuron0x215a0910()*0.668865);
}

double uuNNEB::synapse0x20cedf70() {
   return (neuron0x215a0c50()*-0.60273);
}

double uuNNEB::synapse0x20e198e0() {
   return (neuron0x215a0f90()*0.283323);
}

double uuNNEB::synapse0x20e19920() {
   return (neuron0x215a12d0()*0.0257678);
}

double uuNNEB::synapse0x20e19960() {
   return (neuron0x215a1610()*0.519152);
}

double uuNNEB::synapse0x20e199a0() {
   return (neuron0x215a1950()*-0.260405);
}

double uuNNEB::synapse0x20e199e0() {
   return (neuron0x215a1c90()*-0.0699433);
}

double uuNNEB::synapse0x20e19a20() {
   return (neuron0x215a1fd0()*0.405117);
}

double uuNNEB::synapse0x215a5710() {
   return (neuron0x215a0910()*-2.56026);
}

double uuNNEB::synapse0x215a5750() {
   return (neuron0x215a0c50()*-0.0943651);
}

double uuNNEB::synapse0x215a5790() {
   return (neuron0x215a0f90()*-0.228386);
}

double uuNNEB::synapse0x215a57d0() {
   return (neuron0x215a12d0()*-0.199914);
}

double uuNNEB::synapse0x215a5810() {
   return (neuron0x215a1610()*0.472089);
}

double uuNNEB::synapse0x215a5850() {
   return (neuron0x215a1950()*0.374635);
}

double uuNNEB::synapse0x215a5890() {
   return (neuron0x215a1c90()*2.80952);
}

double uuNNEB::synapse0x215a58d0() {
   return (neuron0x215a1fd0()*0.0236235);
}

double uuNNEB::synapse0x215a5c50() {
   return (neuron0x215a2440()*1.80182);
}

double uuNNEB::synapse0x215a5c90() {
   return (neuron0x215a2870()*0.452891);
}

double uuNNEB::synapse0x215a5cd0() {
   return (neuron0x215a2db0()*-1.64631);
}

double uuNNEB::synapse0x215a5d10() {
   return (neuron0x215a3380()*-1.19492);
}

double uuNNEB::synapse0x215a5d50() {
   return (neuron0x215a38c0()*1.45913);
}

double uuNNEB::synapse0x215a5d90() {
   return (neuron0x215a3c40()*0.370735);
}

double uuNNEB::synapse0x215a5dd0() {
   return (neuron0x215a4180()*-2.74557);
}

double uuNNEB::synapse0x215a5e10() {
   return (neuron0x215a46c0()*1.27757);
}

double uuNNEB::synapse0x215a5e50() {
   return (neuron0x215a4c00()*0.113696);
}

double uuNNEB::synapse0x215a5e90() {
   return (neuron0x215a5460()*1.2331);
}

double uuNNEB::synapse0x215a6210() {
   return (neuron0x215a2440()*3.08306);
}

double uuNNEB::synapse0x215a6250() {
   return (neuron0x215a2870()*-2.61474);
}

double uuNNEB::synapse0x215a6290() {
   return (neuron0x215a2db0()*-1.11434);
}

double uuNNEB::synapse0x215a62d0() {
   return (neuron0x215a3380()*1.99314);
}

double uuNNEB::synapse0x215a6310() {
   return (neuron0x215a38c0()*0.197835);
}

double uuNNEB::synapse0x215a6350() {
   return (neuron0x215a3c40()*-0.142996);
}

double uuNNEB::synapse0x215a6390() {
   return (neuron0x215a4180()*-0.816864);
}

double uuNNEB::synapse0x215a63d0() {
   return (neuron0x215a46c0()*2.032);
}

double uuNNEB::synapse0x215a6410() {
   return (neuron0x215a4c00()*0.89448);
}

double uuNNEB::synapse0x215a6450() {
   return (neuron0x215a5460()*1.29876);
}

double uuNNEB::synapse0x215a67d0() {
   return (neuron0x215a2440()*0.414836);
}

double uuNNEB::synapse0x215a6810() {
   return (neuron0x215a2870()*0.191323);
}

double uuNNEB::synapse0x215a6850() {
   return (neuron0x215a2db0()*-3.98714);
}

double uuNNEB::synapse0x215a6890() {
   return (neuron0x215a3380()*0.447498);
}

double uuNNEB::synapse0x215a68d0() {
   return (neuron0x215a38c0()*2.68837);
}

double uuNNEB::synapse0x215a6910() {
   return (neuron0x215a3c40()*-1.63414);
}

double uuNNEB::synapse0x215a6950() {
   return (neuron0x215a4180()*-1.59613);
}

double uuNNEB::synapse0x215a6990() {
   return (neuron0x215a46c0()*1.4858);
}

double uuNNEB::synapse0x215a69d0() {
   return (neuron0x215a4c00()*-0.59188);
}

double uuNNEB::synapse0x215a6a10() {
   return (neuron0x215a5460()*0.696164);
}

double uuNNEB::synapse0x215a6d90() {
   return (neuron0x215a2440()*-0.544148);
}

double uuNNEB::synapse0x215a6dd0() {
   return (neuron0x215a2870()*-0.835737);
}

double uuNNEB::synapse0x215a6e10() {
   return (neuron0x215a2db0()*-1.91574);
}

double uuNNEB::synapse0x215a6e50() {
   return (neuron0x215a3380()*-0.57913);
}

double uuNNEB::synapse0x215a6e90() {
   return (neuron0x215a38c0()*1.35325);
}

double uuNNEB::synapse0x215a6ed0() {
   return (neuron0x215a3c40()*0.215738);
}

double uuNNEB::synapse0x215a6f10() {
   return (neuron0x215a4180()*1.04704);
}

double uuNNEB::synapse0x215a6f50() {
   return (neuron0x215a46c0()*-0.289321);
}

double uuNNEB::synapse0x215a6f90() {
   return (neuron0x215a4c00()*-0.204792);
}

double uuNNEB::synapse0x215a6fd0() {
   return (neuron0x215a5460()*0.0657331);
}

double uuNNEB::synapse0x215a7350() {
   return (neuron0x215a2440()*2.17066);
}

double uuNNEB::synapse0x215a7390() {
   return (neuron0x215a2870()*0.508575);
}

double uuNNEB::synapse0x215a73d0() {
   return (neuron0x215a2db0()*0.374617);
}

double uuNNEB::synapse0x215a7410() {
   return (neuron0x215a3380()*2.54452);
}

double uuNNEB::synapse0x215a7450() {
   return (neuron0x215a38c0()*1.03871);
}

double uuNNEB::synapse0x215a7490() {
   return (neuron0x215a3c40()*1.66025);
}

double uuNNEB::synapse0x215a74d0() {
   return (neuron0x215a4180()*-0.749529);
}

double uuNNEB::synapse0x215a7510() {
   return (neuron0x215a46c0()*0.516512);
}

double uuNNEB::synapse0x215a7550() {
   return (neuron0x215a4c00()*0.492925);
}

double uuNNEB::synapse0x215a5050() {
   return (neuron0x215a5460()*1.1131);
}

double uuNNEB::synapse0x215a53d0() {
   return (neuron0x215a5910()*3.22095);
}

double uuNNEB::synapse0x215a5410() {
   return (neuron0x215a5ed0()*3.08946);
}

double uuNNEB::synapse0x215a2310() {
   return (neuron0x215a6490()*3.36558);
}

double uuNNEB::synapse0x215a2350() {
   return (neuron0x215a6a50()*-4.11024);
}

double uuNNEB::synapse0x215a2390() {
   return (neuron0x215a7010()*2.09935);
}

