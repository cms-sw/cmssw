#include "ruNNEB.h"
#include <cmath>

double ruNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.15492)/1.68616;
   input2 = (in2 - 1.14956)/1.69316;
   input3 = (in3 - 1.89563)/1.49913;
   input4 = (in4 - 1.90666)/1.49442;
   input5 = (in5 - 0.318454)/1.50742;
   input6 = (in6 - 0.305354)/1.51455;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0xc536fd0();
     default:
         return 0.;
   }
}

double ruNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.15492)/1.68616;
   input2 = (input[2] - 1.14956)/1.69316;
   input3 = (input[3] - 1.89563)/1.49913;
   input4 = (input[4] - 1.90666)/1.49442;
   input5 = (input[5] - 0.318454)/1.50742;
   input6 = (input[6] - 0.305354)/1.51455;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0xc536fd0();
     default:
         return 0.;
   }
}

double ruNNEB::neuron0xc532850() {
   return input0;
}

double ruNNEB::neuron0xc532b90() {
   return input1;
}

double ruNNEB::neuron0xc532ed0() {
   return input2;
}

double ruNNEB::neuron0xc533210() {
   return input3;
}

double ruNNEB::neuron0xc533550() {
   return input4;
}

double ruNNEB::neuron0xc533890() {
   return input5;
}

double ruNNEB::neuron0xc533bd0() {
   return input6;
}

double ruNNEB::neuron0xc533f10() {
   return input7;
}

double ruNNEB::input0xc534380() {
   double input = 1.45945;
   input += synapse0xc493250();
   input += synapse0xc53b450();
   input += synapse0xc534630();
   input += synapse0xc534670();
   input += synapse0xc5346b0();
   input += synapse0xc5346f0();
   input += synapse0xc534730();
   input += synapse0xc534770();
   return input;
}

double ruNNEB::neuron0xc534380() {
   double input = input0xc534380();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc5347b0() {
   double input = 2.06722;
   input += synapse0xc534af0();
   input += synapse0xc534b30();
   input += synapse0xc534b70();
   input += synapse0xc534bb0();
   input += synapse0xc534bf0();
   input += synapse0xc534c30();
   input += synapse0xc534c70();
   input += synapse0xc534cb0();
   return input;
}

double ruNNEB::neuron0xc5347b0() {
   double input = input0xc5347b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc534cf0() {
   double input = -0.660232;
   input += synapse0xc535030();
   input += synapse0xbd91f10();
   input += synapse0xbd91f50();
   input += synapse0xc535180();
   input += synapse0xc5351c0();
   input += synapse0xc535200();
   input += synapse0xc535240();
   input += synapse0xc535280();
   return input;
}

double ruNNEB::neuron0xc534cf0() {
   double input = input0xc534cf0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc5352c0() {
   double input = -0.124235;
   input += synapse0xc535600();
   input += synapse0xc535640();
   input += synapse0xc535680();
   input += synapse0xc5356c0();
   input += synapse0xc535700();
   input += synapse0xc535740();
   input += synapse0xc535780();
   input += synapse0xc5357c0();
   return input;
}

double ruNNEB::neuron0xc5352c0() {
   double input = input0xc5352c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc535800() {
   double input = 1.07315;
   input += synapse0xc535b40();
   input += synapse0xc532780();
   input += synapse0xc53b490();
   input += synapse0xbd90a50();
   input += synapse0xc535070();
   input += synapse0xc5350b0();
   input += synapse0xc5350f0();
   input += synapse0xc535130();
   return input;
}

double ruNNEB::neuron0xc535800() {
   double input = input0xc535800();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc535b80() {
   double input = -0.607245;
   input += synapse0xc535ec0();
   input += synapse0xc535f00();
   input += synapse0xc535f40();
   input += synapse0xc535f80();
   input += synapse0xc535fc0();
   input += synapse0xc536000();
   input += synapse0xc536040();
   input += synapse0xc536080();
   return input;
}

double ruNNEB::neuron0xc535b80() {
   double input = input0xc535b80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc5360c0() {
   double input = 0.389523;
   input += synapse0xc536400();
   input += synapse0xc536440();
   input += synapse0xc536480();
   input += synapse0xc5364c0();
   input += synapse0xc536500();
   input += synapse0xc536540();
   input += synapse0xc536580();
   input += synapse0xc5365c0();
   return input;
}

double ruNNEB::neuron0xc5360c0() {
   double input = input0xc5360c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc536600() {
   double input = -2.57324;
   input += synapse0xc536940();
   input += synapse0xc536980();
   input += synapse0xc5369c0();
   input += synapse0xc536a00();
   input += synapse0xc536a40();
   input += synapse0xc536a80();
   input += synapse0xc536ac0();
   input += synapse0xc536b00();
   return input;
}

double ruNNEB::neuron0xc536600() {
   double input = input0xc536600();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc536b40() {
   double input = -1.05451;
   input += synapse0xbc7fe70();
   input += synapse0xbc7feb0();
   input += synapse0xbdab820();
   input += synapse0xbdab860();
   input += synapse0xbdab8a0();
   input += synapse0xbdab8e0();
   input += synapse0xbdab920();
   input += synapse0xbdab960();
   return input;
}

double ruNNEB::neuron0xc536b40() {
   double input = input0xc536b40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc5373a0() {
   double input = 0.758507;
   input += synapse0xc537650();
   input += synapse0xc537690();
   input += synapse0xc5376d0();
   input += synapse0xc537710();
   input += synapse0xc537750();
   input += synapse0xc537790();
   input += synapse0xc5377d0();
   input += synapse0xc537810();
   return input;
}

double ruNNEB::neuron0xc5373a0() {
   double input = input0xc5373a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc537850() {
   double input = -0.2328;
   input += synapse0xc537b90();
   input += synapse0xc537bd0();
   input += synapse0xc537c10();
   input += synapse0xc537c50();
   input += synapse0xc537c90();
   input += synapse0xc537cd0();
   input += synapse0xc537d10();
   input += synapse0xc537d50();
   input += synapse0xc537d90();
   input += synapse0xc537dd0();
   return input;
}

double ruNNEB::neuron0xc537850() {
   double input = input0xc537850();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc537e10() {
   double input = 0.0236358;
   input += synapse0xc538150();
   input += synapse0xc538190();
   input += synapse0xc5381d0();
   input += synapse0xc538210();
   input += synapse0xc538250();
   input += synapse0xc538290();
   input += synapse0xc5382d0();
   input += synapse0xc538310();
   input += synapse0xc538350();
   input += synapse0xc538390();
   return input;
}

double ruNNEB::neuron0xc537e10() {
   double input = input0xc537e10();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc5383d0() {
   double input = 0.155914;
   input += synapse0xc538710();
   input += synapse0xc538750();
   input += synapse0xc538790();
   input += synapse0xc5387d0();
   input += synapse0xc538810();
   input += synapse0xc538850();
   input += synapse0xc538890();
   input += synapse0xc5388d0();
   input += synapse0xc538910();
   input += synapse0xc538950();
   return input;
}

double ruNNEB::neuron0xc5383d0() {
   double input = input0xc5383d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc538990() {
   double input = -0.295368;
   input += synapse0xc538cd0();
   input += synapse0xc538d10();
   input += synapse0xc538d50();
   input += synapse0xc538d90();
   input += synapse0xc538dd0();
   input += synapse0xc538e10();
   input += synapse0xc538e50();
   input += synapse0xc538e90();
   input += synapse0xc538ed0();
   input += synapse0xc538f10();
   return input;
}

double ruNNEB::neuron0xc538990() {
   double input = input0xc538990();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc538f50() {
   double input = 0.132164;
   input += synapse0xc539290();
   input += synapse0xc5392d0();
   input += synapse0xc539310();
   input += synapse0xc539350();
   input += synapse0xc539390();
   input += synapse0xc5393d0();
   input += synapse0xc539410();
   input += synapse0xc539450();
   input += synapse0xc539490();
   input += synapse0xc536f90();
   return input;
}

double ruNNEB::neuron0xc538f50() {
   double input = input0xc538f50();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEB::input0xc536fd0() {
   double input = -2.06837;
   input += synapse0xc537310();
   input += synapse0xc537350();
   input += synapse0xc534250();
   input += synapse0xc534290();
   input += synapse0xc5342d0();
   return input;
}

double ruNNEB::neuron0xc536fd0() {
   double input = input0xc536fd0();
   return (input * 1)+0;
}

double ruNNEB::synapse0xc493250() {
   return (neuron0xc532850()*-0.0701012);
}

double ruNNEB::synapse0xc53b450() {
   return (neuron0xc532b90()*-0.66992);
}

double ruNNEB::synapse0xc534630() {
   return (neuron0xc532ed0()*-0.640978);
}

double ruNNEB::synapse0xc534670() {
   return (neuron0xc533210()*1.0476);
}

double ruNNEB::synapse0xc5346b0() {
   return (neuron0xc533550()*-0.211549);
}

double ruNNEB::synapse0xc5346f0() {
   return (neuron0xc533890()*0.604938);
}

double ruNNEB::synapse0xc534730() {
   return (neuron0xc533bd0()*-0.48798);
}

double ruNNEB::synapse0xc534770() {
   return (neuron0xc533f10()*0.379182);
}

double ruNNEB::synapse0xc534af0() {
   return (neuron0xc532850()*-1.1129);
}

double ruNNEB::synapse0xc534b30() {
   return (neuron0xc532b90()*-0.682829);
}

double ruNNEB::synapse0xc534b70() {
   return (neuron0xc532ed0()*-0.341504);
}

double ruNNEB::synapse0xc534bb0() {
   return (neuron0xc533210()*0.49885);
}

double ruNNEB::synapse0xc534bf0() {
   return (neuron0xc533550()*0.812563);
}

double ruNNEB::synapse0xc534c30() {
   return (neuron0xc533890()*0.346436);
}

double ruNNEB::synapse0xc534c70() {
   return (neuron0xc533bd0()*0.112762);
}

double ruNNEB::synapse0xc534cb0() {
   return (neuron0xc533f10()*-0.00576687);
}

double ruNNEB::synapse0xc535030() {
   return (neuron0xc532850()*0.905371);
}

double ruNNEB::synapse0xbd91f10() {
   return (neuron0xc532b90()*-0.081892);
}

double ruNNEB::synapse0xbd91f50() {
   return (neuron0xc532ed0()*-1.43722);
}

double ruNNEB::synapse0xc535180() {
   return (neuron0xc533210()*0.822035);
}

double ruNNEB::synapse0xc5351c0() {
   return (neuron0xc533550()*0.0800306);
}

double ruNNEB::synapse0xc535200() {
   return (neuron0xc533890()*-0.25427);
}

double ruNNEB::synapse0xc535240() {
   return (neuron0xc533bd0()*-0.147545);
}

double ruNNEB::synapse0xc535280() {
   return (neuron0xc533f10()*-0.713625);
}

double ruNNEB::synapse0xc535600() {
   return (neuron0xc532850()*-0.0268293);
}

double ruNNEB::synapse0xc535640() {
   return (neuron0xc532b90()*0.136042);
}

double ruNNEB::synapse0xc535680() {
   return (neuron0xc532ed0()*-0.152346);
}

double ruNNEB::synapse0xc5356c0() {
   return (neuron0xc533210()*-0.0287884);
}

double ruNNEB::synapse0xc535700() {
   return (neuron0xc533550()*0.120797);
}

double ruNNEB::synapse0xc535740() {
   return (neuron0xc533890()*-0.63162);
}

double ruNNEB::synapse0xc535780() {
   return (neuron0xc533bd0()*1.03652);
}

double ruNNEB::synapse0xc5357c0() {
   return (neuron0xc533f10()*0.725806);
}

double ruNNEB::synapse0xc535b40() {
   return (neuron0xc532850()*0.259529);
}

double ruNNEB::synapse0xc532780() {
   return (neuron0xc532b90()*-0.096735);
}

double ruNNEB::synapse0xc53b490() {
   return (neuron0xc532ed0()*-1.03728);
}

double ruNNEB::synapse0xbd90a50() {
   return (neuron0xc533210()*-0.130331);
}

double ruNNEB::synapse0xc535070() {
   return (neuron0xc533550()*-0.665413);
}

double ruNNEB::synapse0xc5350b0() {
   return (neuron0xc533890()*-0.0465597);
}

double ruNNEB::synapse0xc5350f0() {
   return (neuron0xc533bd0()*0.125835);
}

double ruNNEB::synapse0xc535130() {
   return (neuron0xc533f10()*0.903114);
}

double ruNNEB::synapse0xc535ec0() {
   return (neuron0xc532850()*-0.493811);
}

double ruNNEB::synapse0xc535f00() {
   return (neuron0xc532b90()*0.104356);
}

double ruNNEB::synapse0xc535f40() {
   return (neuron0xc532ed0()*-0.57549);
}

double ruNNEB::synapse0xc535f80() {
   return (neuron0xc533210()*-0.0153415);
}

double ruNNEB::synapse0xc535fc0() {
   return (neuron0xc533550()*0.0543057);
}

double ruNNEB::synapse0xc536000() {
   return (neuron0xc533890()*-0.549977);
}

double ruNNEB::synapse0xc536040() {
   return (neuron0xc533bd0()*-0.461156);
}

double ruNNEB::synapse0xc536080() {
   return (neuron0xc533f10()*0.0203922);
}

double ruNNEB::synapse0xc536400() {
   return (neuron0xc532850()*0.234716);
}

double ruNNEB::synapse0xc536440() {
   return (neuron0xc532b90()*0.651407);
}

double ruNNEB::synapse0xc536480() {
   return (neuron0xc532ed0()*-0.214669);
}

double ruNNEB::synapse0xc5364c0() {
   return (neuron0xc533210()*-0.160465);
}

double ruNNEB::synapse0xc536500() {
   return (neuron0xc533550()*0.0914841);
}

double ruNNEB::synapse0xc536540() {
   return (neuron0xc533890()*-0.74246);
}

double ruNNEB::synapse0xc536580() {
   return (neuron0xc533bd0()*0.583625);
}

double ruNNEB::synapse0xc5365c0() {
   return (neuron0xc533f10()*0.198533);
}

double ruNNEB::synapse0xc536940() {
   return (neuron0xc532850()*-1.40039);
}

double ruNNEB::synapse0xc536980() {
   return (neuron0xc532b90()*0.724146);
}

double ruNNEB::synapse0xc5369c0() {
   return (neuron0xc532ed0()*-0.179277);
}

double ruNNEB::synapse0xc536a00() {
   return (neuron0xc533210()*1.00082);
}

double ruNNEB::synapse0xc536a40() {
   return (neuron0xc533550()*-0.468151);
}

double ruNNEB::synapse0xc536a80() {
   return (neuron0xc533890()*0.551447);
}

double ruNNEB::synapse0xc536ac0() {
   return (neuron0xc533bd0()*-0.236782);
}

double ruNNEB::synapse0xc536b00() {
   return (neuron0xc533f10()*0.378075);
}

double ruNNEB::synapse0xbc7fe70() {
   return (neuron0xc532850()*0.35642);
}

double ruNNEB::synapse0xbc7feb0() {
   return (neuron0xc532b90()*-1.68175);
}

double ruNNEB::synapse0xbdab820() {
   return (neuron0xc532ed0()*0.779485);
}

double ruNNEB::synapse0xbdab860() {
   return (neuron0xc533210()*0.371443);
}

double ruNNEB::synapse0xbdab8a0() {
   return (neuron0xc533550()*0.162279);
}

double ruNNEB::synapse0xbdab8e0() {
   return (neuron0xc533890()*0.24014);
}

double ruNNEB::synapse0xbdab920() {
   return (neuron0xc533bd0()*-0.283817);
}

double ruNNEB::synapse0xbdab960() {
   return (neuron0xc533f10()*-0.482751);
}

double ruNNEB::synapse0xc537650() {
   return (neuron0xc532850()*-0.762919);
}

double ruNNEB::synapse0xc537690() {
   return (neuron0xc532b90()*0.18839);
}

double ruNNEB::synapse0xc5376d0() {
   return (neuron0xc532ed0()*-0.829236);
}

double ruNNEB::synapse0xc537710() {
   return (neuron0xc533210()*0.463294);
}

double ruNNEB::synapse0xc537750() {
   return (neuron0xc533550()*-0.169371);
}

double ruNNEB::synapse0xc537790() {
   return (neuron0xc533890()*0.171274);
}

double ruNNEB::synapse0xc5377d0() {
   return (neuron0xc533bd0()*-0.195367);
}

double ruNNEB::synapse0xc537810() {
   return (neuron0xc533f10()*0.276946);
}

double ruNNEB::synapse0xc537b90() {
   return (neuron0xc534380()*0.89148);
}

double ruNNEB::synapse0xc537bd0() {
   return (neuron0xc5347b0()*-1.36244);
}

double ruNNEB::synapse0xc537c10() {
   return (neuron0xc534cf0()*1.21674);
}

double ruNNEB::synapse0xc537c50() {
   return (neuron0xc5352c0()*0.396925);
}

double ruNNEB::synapse0xc537c90() {
   return (neuron0xc535800()*-1.70521);
}

double ruNNEB::synapse0xc537cd0() {
   return (neuron0xc535b80()*-0.232645);
}

double ruNNEB::synapse0xc537d10() {
   return (neuron0xc5360c0()*0.674072);
}

double ruNNEB::synapse0xc537d50() {
   return (neuron0xc536600()*1.33276);
}

double ruNNEB::synapse0xc537d90() {
   return (neuron0xc536b40()*-1.47118);
}

double ruNNEB::synapse0xc537dd0() {
   return (neuron0xc5373a0()*-1.54233);
}

double ruNNEB::synapse0xc538150() {
   return (neuron0xc534380()*1.01373);
}

double ruNNEB::synapse0xc538190() {
   return (neuron0xc5347b0()*-0.704411);
}

double ruNNEB::synapse0xc5381d0() {
   return (neuron0xc534cf0()*0.465801);
}

double ruNNEB::synapse0xc538210() {
   return (neuron0xc5352c0()*-0.566637);
}

double ruNNEB::synapse0xc538250() {
   return (neuron0xc535800()*-0.699478);
}

double ruNNEB::synapse0xc538290() {
   return (neuron0xc535b80()*-0.39378);
}

double ruNNEB::synapse0xc5382d0() {
   return (neuron0xc5360c0()*0.895162);
}

double ruNNEB::synapse0xc538310() {
   return (neuron0xc536600()*0.890616);
}

double ruNNEB::synapse0xc538350() {
   return (neuron0xc536b40()*-0.417674);
}

double ruNNEB::synapse0xc538390() {
   return (neuron0xc5373a0()*-0.425274);
}

double ruNNEB::synapse0xc538710() {
   return (neuron0xc534380()*0.916152);
}

double ruNNEB::synapse0xc538750() {
   return (neuron0xc5347b0()*-0.55573);
}

double ruNNEB::synapse0xc538790() {
   return (neuron0xc534cf0()*0.523426);
}

double ruNNEB::synapse0xc5387d0() {
   return (neuron0xc5352c0()*-0.803919);
}

double ruNNEB::synapse0xc538810() {
   return (neuron0xc535800()*-0.730321);
}

double ruNNEB::synapse0xc538850() {
   return (neuron0xc535b80()*-0.496462);
}

double ruNNEB::synapse0xc538890() {
   return (neuron0xc5360c0()*0.818055);
}

double ruNNEB::synapse0xc5388d0() {
   return (neuron0xc536600()*1.25811);
}

double ruNNEB::synapse0xc538910() {
   return (neuron0xc536b40()*-0.496195);
}

double ruNNEB::synapse0xc538950() {
   return (neuron0xc5373a0()*-0.769757);
}

double ruNNEB::synapse0xc538cd0() {
   return (neuron0xc534380()*-1.67636);
}

double ruNNEB::synapse0xc538d10() {
   return (neuron0xc5347b0()*-0.0178184);
}

double ruNNEB::synapse0xc538d50() {
   return (neuron0xc534cf0()*0.0371507);
}

double ruNNEB::synapse0xc538d90() {
   return (neuron0xc5352c0()*0.104829);
}

double ruNNEB::synapse0xc538dd0() {
   return (neuron0xc535800()*1.10022);
}

double ruNNEB::synapse0xc538e10() {
   return (neuron0xc535b80()*0.300674);
}

double ruNNEB::synapse0xc538e50() {
   return (neuron0xc5360c0()*-1.27645);
}

double ruNNEB::synapse0xc538e90() {
   return (neuron0xc536600()*-0.661545);
}

double ruNNEB::synapse0xc538ed0() {
   return (neuron0xc536b40()*0.600524);
}

double ruNNEB::synapse0xc538f10() {
   return (neuron0xc5373a0()*1.10324);
}

double ruNNEB::synapse0xc539290() {
   return (neuron0xc534380()*0.342316);
}

double ruNNEB::synapse0xc5392d0() {
   return (neuron0xc5347b0()*-0.856519);
}

double ruNNEB::synapse0xc539310() {
   return (neuron0xc534cf0()*0.525865);
}

double ruNNEB::synapse0xc539350() {
   return (neuron0xc5352c0()*0.121581);
}

double ruNNEB::synapse0xc539390() {
   return (neuron0xc535800()*-1.23787);
}

double ruNNEB::synapse0xc5393d0() {
   return (neuron0xc535b80()*-0.166805);
}

double ruNNEB::synapse0xc539410() {
   return (neuron0xc5360c0()*-0.190686);
}

double ruNNEB::synapse0xc539450() {
   return (neuron0xc536600()*0.687436);
}

double ruNNEB::synapse0xc539490() {
   return (neuron0xc536b40()*-0.671269);
}

double ruNNEB::synapse0xc536f90() {
   return (neuron0xc5373a0()*-1.17779);
}

double ruNNEB::synapse0xc537310() {
   return (neuron0xc537850()*4.14555);
}

double ruNNEB::synapse0xc537350() {
   return (neuron0xc537e10()*3.50214);
}

double ruNNEB::synapse0xc534250() {
   return (neuron0xc5383d0()*4.58479);
}

double ruNNEB::synapse0xc534290() {
   return (neuron0xc538990()*-3.61183);
}

double ruNNEB::synapse0xc5342d0() {
   return (neuron0xc538f50()*0.901646);
}

