#include "ddNNEB.h"
#include <cmath>

double ddNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.15492)/1.68616;
   input2 = (in2 - 1.14956)/1.69316;
   input3 = (in3 - 1.89563)/1.49913;
   input4 = (in4 - 0.3065)/1.46726;
   input5 = (in5 - 0.318454)/1.50742;
   input6 = (in6 - 0.305354)/1.51455;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x4a4af10();
     default:
         return 0.;
   }
}

double ddNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.15492)/1.68616;
   input2 = (input[2] - 1.14956)/1.69316;
   input3 = (input[3] - 1.89563)/1.49913;
   input4 = (input[4] - 0.3065)/1.46726;
   input5 = (input[5] - 0.318454)/1.50742;
   input6 = (input[6] - 0.305354)/1.51455;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x4a4af10();
     default:
         return 0.;
   }
}

double ddNNEB::neuron0x4a46790() {
   return input0;
}

double ddNNEB::neuron0x4a46ad0() {
   return input1;
}

double ddNNEB::neuron0x4a46e10() {
   return input2;
}

double ddNNEB::neuron0x4a47150() {
   return input3;
}

double ddNNEB::neuron0x4a47490() {
   return input4;
}

double ddNNEB::neuron0x4a477d0() {
   return input5;
}

double ddNNEB::neuron0x4a47b10() {
   return input6;
}

double ddNNEB::neuron0x4a47e50() {
   return input7;
}

double ddNNEB::input0x4a482c0() {
   double input = -2.63524;
   input += synapse0x49a7190();
   input += synapse0x4a4f390();
   input += synapse0x4a48570();
   input += synapse0x4a485b0();
   input += synapse0x4a485f0();
   input += synapse0x4a48630();
   input += synapse0x4a48670();
   input += synapse0x4a486b0();
   return input;
}

double ddNNEB::neuron0x4a482c0() {
   double input = input0x4a482c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a486f0() {
   double input = 1.91074;
   input += synapse0x4a48a30();
   input += synapse0x4a48a70();
   input += synapse0x4a48ab0();
   input += synapse0x4a48af0();
   input += synapse0x4a48b30();
   input += synapse0x4a48b70();
   input += synapse0x4a48bb0();
   input += synapse0x4a48bf0();
   return input;
}

double ddNNEB::neuron0x4a486f0() {
   double input = input0x4a486f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a48c30() {
   double input = -0.128303;
   input += synapse0x4a48f70();
   input += synapse0x42a5e50();
   input += synapse0x42a5e90();
   input += synapse0x4a490c0();
   input += synapse0x4a49100();
   input += synapse0x4a49140();
   input += synapse0x4a49180();
   input += synapse0x4a491c0();
   return input;
}

double ddNNEB::neuron0x4a48c30() {
   double input = input0x4a48c30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a49200() {
   double input = -0.0143879;
   input += synapse0x4a49540();
   input += synapse0x4a49580();
   input += synapse0x4a495c0();
   input += synapse0x4a49600();
   input += synapse0x4a49640();
   input += synapse0x4a49680();
   input += synapse0x4a496c0();
   input += synapse0x4a49700();
   return input;
}

double ddNNEB::neuron0x4a49200() {
   double input = input0x4a49200();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a49740() {
   double input = 1.50607;
   input += synapse0x4a49a80();
   input += synapse0x4a466c0();
   input += synapse0x4a4f3d0();
   input += synapse0x42a4990();
   input += synapse0x4a48fb0();
   input += synapse0x4a48ff0();
   input += synapse0x4a49030();
   input += synapse0x4a49070();
   return input;
}

double ddNNEB::neuron0x4a49740() {
   double input = input0x4a49740();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a49ac0() {
   double input = 1.02996;
   input += synapse0x4a49e00();
   input += synapse0x4a49e40();
   input += synapse0x4a49e80();
   input += synapse0x4a49ec0();
   input += synapse0x4a49f00();
   input += synapse0x4a49f40();
   input += synapse0x4a49f80();
   input += synapse0x4a49fc0();
   return input;
}

double ddNNEB::neuron0x4a49ac0() {
   double input = input0x4a49ac0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4a000() {
   double input = 0.911485;
   input += synapse0x4a4a340();
   input += synapse0x4a4a380();
   input += synapse0x4a4a3c0();
   input += synapse0x4a4a400();
   input += synapse0x4a4a440();
   input += synapse0x4a4a480();
   input += synapse0x4a4a4c0();
   input += synapse0x4a4a500();
   return input;
}

double ddNNEB::neuron0x4a4a000() {
   double input = input0x4a4a000();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4a540() {
   double input = 1.01528;
   input += synapse0x4a4a880();
   input += synapse0x4a4a8c0();
   input += synapse0x4a4a900();
   input += synapse0x4a4a940();
   input += synapse0x4a4a980();
   input += synapse0x4a4a9c0();
   input += synapse0x4a4aa00();
   input += synapse0x4a4aa40();
   return input;
}

double ddNNEB::neuron0x4a4a540() {
   double input = input0x4a4a540();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4aa80() {
   double input = -1.78246;
   input += synapse0x4193db0();
   input += synapse0x4193df0();
   input += synapse0x42bf760();
   input += synapse0x42bf7a0();
   input += synapse0x42bf7e0();
   input += synapse0x42bf820();
   input += synapse0x42bf860();
   input += synapse0x42bf8a0();
   return input;
}

double ddNNEB::neuron0x4a4aa80() {
   double input = input0x4a4aa80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4b2e0() {
   double input = -0.164083;
   input += synapse0x4a4b590();
   input += synapse0x4a4b5d0();
   input += synapse0x4a4b610();
   input += synapse0x4a4b650();
   input += synapse0x4a4b690();
   input += synapse0x4a4b6d0();
   input += synapse0x4a4b710();
   input += synapse0x4a4b750();
   return input;
}

double ddNNEB::neuron0x4a4b2e0() {
   double input = input0x4a4b2e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4b790() {
   double input = -1.84156;
   input += synapse0x4a4bad0();
   input += synapse0x4a4bb10();
   input += synapse0x4a4bb50();
   input += synapse0x4a4bb90();
   input += synapse0x4a4bbd0();
   input += synapse0x4a4bc10();
   input += synapse0x4a4bc50();
   input += synapse0x4a4bc90();
   input += synapse0x4a4bcd0();
   input += synapse0x4a4bd10();
   return input;
}

double ddNNEB::neuron0x4a4b790() {
   double input = input0x4a4b790();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4bd50() {
   double input = -2.55771;
   input += synapse0x4a4c090();
   input += synapse0x4a4c0d0();
   input += synapse0x4a4c110();
   input += synapse0x4a4c150();
   input += synapse0x4a4c190();
   input += synapse0x4a4c1d0();
   input += synapse0x4a4c210();
   input += synapse0x4a4c250();
   input += synapse0x4a4c290();
   input += synapse0x4a4c2d0();
   return input;
}

double ddNNEB::neuron0x4a4bd50() {
   double input = input0x4a4bd50();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4c310() {
   double input = -3.54477;
   input += synapse0x4a4c650();
   input += synapse0x4a4c690();
   input += synapse0x4a4c6d0();
   input += synapse0x4a4c710();
   input += synapse0x4a4c750();
   input += synapse0x4a4c790();
   input += synapse0x4a4c7d0();
   input += synapse0x4a4c810();
   input += synapse0x4a4c850();
   input += synapse0x4a4c890();
   return input;
}

double ddNNEB::neuron0x4a4c310() {
   double input = input0x4a4c310();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4c8d0() {
   double input = -1.55382;
   input += synapse0x4a4cc10();
   input += synapse0x4a4cc50();
   input += synapse0x4a4cc90();
   input += synapse0x4a4ccd0();
   input += synapse0x4a4cd10();
   input += synapse0x4a4cd50();
   input += synapse0x4a4cd90();
   input += synapse0x4a4cdd0();
   input += synapse0x4a4ce10();
   input += synapse0x4a4ce50();
   return input;
}

double ddNNEB::neuron0x4a4c8d0() {
   double input = input0x4a4c8d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4ce90() {
   double input = 1.9974;
   input += synapse0x4a4d1d0();
   input += synapse0x4a4d210();
   input += synapse0x4a4d250();
   input += synapse0x4a4d290();
   input += synapse0x4a4d2d0();
   input += synapse0x4a4d310();
   input += synapse0x4a4d350();
   input += synapse0x4a4d390();
   input += synapse0x4a4d3d0();
   input += synapse0x4a4aed0();
   return input;
}

double ddNNEB::neuron0x4a4ce90() {
   double input = input0x4a4ce90();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEB::input0x4a4af10() {
   double input = -1.41785;
   input += synapse0x4a4b250();
   input += synapse0x4a4b290();
   input += synapse0x4a48190();
   input += synapse0x4a481d0();
   input += synapse0x4a48210();
   return input;
}

double ddNNEB::neuron0x4a4af10() {
   double input = input0x4a4af10();
   return (input * 1)+0;
}

double ddNNEB::synapse0x49a7190() {
   return (neuron0x4a46790()*0.542666);
}

double ddNNEB::synapse0x4a4f390() {
   return (neuron0x4a46ad0()*-0.41876);
}

double ddNNEB::synapse0x4a48570() {
   return (neuron0x4a46e10()*0.255798);
}

double ddNNEB::synapse0x4a485b0() {
   return (neuron0x4a47150()*-0.242769);
}

double ddNNEB::synapse0x4a485f0() {
   return (neuron0x4a47490()*0.13839);
}

double ddNNEB::synapse0x4a48630() {
   return (neuron0x4a477d0()*0.565941);
}

double ddNNEB::synapse0x4a48670() {
   return (neuron0x4a47b10()*-0.018932);
}

double ddNNEB::synapse0x4a486b0() {
   return (neuron0x4a47e50()*0.323705);
}

double ddNNEB::synapse0x4a48a30() {
   return (neuron0x4a46790()*0.656695);
}

double ddNNEB::synapse0x4a48a70() {
   return (neuron0x4a46ad0()*1.32388);
}

double ddNNEB::synapse0x4a48ab0() {
   return (neuron0x4a46e10()*-2.49611);
}

double ddNNEB::synapse0x4a48af0() {
   return (neuron0x4a47150()*-0.0410726);
}

double ddNNEB::synapse0x4a48b30() {
   return (neuron0x4a47490()*0.551725);
}

double ddNNEB::synapse0x4a48b70() {
   return (neuron0x4a477d0()*-1.76628);
}

double ddNNEB::synapse0x4a48bb0() {
   return (neuron0x4a47b10()*-0.287117);
}

double ddNNEB::synapse0x4a48bf0() {
   return (neuron0x4a47e50()*1.74831);
}

double ddNNEB::synapse0x4a48f70() {
   return (neuron0x4a46790()*3.05652);
}

double ddNNEB::synapse0x42a5e50() {
   return (neuron0x4a46ad0()*-0.10501);
}

double ddNNEB::synapse0x42a5e90() {
   return (neuron0x4a46e10()*0.267233);
}

double ddNNEB::synapse0x4a490c0() {
   return (neuron0x4a47150()*0.26929);
}

double ddNNEB::synapse0x4a49100() {
   return (neuron0x4a47490()*-0.780574);
}

double ddNNEB::synapse0x4a49140() {
   return (neuron0x4a477d0()*-1.55704);
}

double ddNNEB::synapse0x4a49180() {
   return (neuron0x4a47b10()*-0.300177);
}

double ddNNEB::synapse0x4a491c0() {
   return (neuron0x4a47e50()*-1.07422);
}

double ddNNEB::synapse0x4a49540() {
   return (neuron0x4a46790()*0.496764);
}

double ddNNEB::synapse0x4a49580() {
   return (neuron0x4a46ad0()*0.597019);
}

double ddNNEB::synapse0x4a495c0() {
   return (neuron0x4a46e10()*0.673857);
}

double ddNNEB::synapse0x4a49600() {
   return (neuron0x4a47150()*0.185765);
}

double ddNNEB::synapse0x4a49640() {
   return (neuron0x4a47490()*0.220123);
}

double ddNNEB::synapse0x4a49680() {
   return (neuron0x4a477d0()*1.37324);
}

double ddNNEB::synapse0x4a496c0() {
   return (neuron0x4a47b10()*0.598104);
}

double ddNNEB::synapse0x4a49700() {
   return (neuron0x4a47e50()*1.28393);
}

double ddNNEB::synapse0x4a49a80() {
   return (neuron0x4a46790()*1.69539);
}

double ddNNEB::synapse0x4a466c0() {
   return (neuron0x4a46ad0()*-0.403427);
}

double ddNNEB::synapse0x4a4f3d0() {
   return (neuron0x4a46e10()*0.565482);
}

double ddNNEB::synapse0x42a4990() {
   return (neuron0x4a47150()*-0.928146);
}

double ddNNEB::synapse0x4a48fb0() {
   return (neuron0x4a47490()*-0.0268804);
}

double ddNNEB::synapse0x4a48ff0() {
   return (neuron0x4a477d0()*0.0993066);
}

double ddNNEB::synapse0x4a49030() {
   return (neuron0x4a47b10()*-0.292285);
}

double ddNNEB::synapse0x4a49070() {
   return (neuron0x4a47e50()*-0.42177);
}

double ddNNEB::synapse0x4a49e00() {
   return (neuron0x4a46790()*-0.0855238);
}

double ddNNEB::synapse0x4a49e40() {
   return (neuron0x4a46ad0()*-1.21593);
}

double ddNNEB::synapse0x4a49e80() {
   return (neuron0x4a46e10()*-0.375884);
}

double ddNNEB::synapse0x4a49ec0() {
   return (neuron0x4a47150()*-0.598447);
}

double ddNNEB::synapse0x4a49f00() {
   return (neuron0x4a47490()*-0.0568257);
}

double ddNNEB::synapse0x4a49f40() {
   return (neuron0x4a477d0()*1.57924);
}

double ddNNEB::synapse0x4a49f80() {
   return (neuron0x4a47b10()*0.147867);
}

double ddNNEB::synapse0x4a49fc0() {
   return (neuron0x4a47e50()*1.38729);
}

double ddNNEB::synapse0x4a4a340() {
   return (neuron0x4a46790()*0.485434);
}

double ddNNEB::synapse0x4a4a380() {
   return (neuron0x4a46ad0()*-0.835655);
}

double ddNNEB::synapse0x4a4a3c0() {
   return (neuron0x4a46e10()*0.0218874);
}

double ddNNEB::synapse0x4a4a400() {
   return (neuron0x4a47150()*0.848471);
}

double ddNNEB::synapse0x4a4a440() {
   return (neuron0x4a47490()*-0.520038);
}

double ddNNEB::synapse0x4a4a480() {
   return (neuron0x4a477d0()*1.05246);
}

double ddNNEB::synapse0x4a4a4c0() {
   return (neuron0x4a47b10()*-0.149156);
}

double ddNNEB::synapse0x4a4a500() {
   return (neuron0x4a47e50()*-0.583154);
}

double ddNNEB::synapse0x4a4a880() {
   return (neuron0x4a46790()*-0.916327);
}

double ddNNEB::synapse0x4a4a8c0() {
   return (neuron0x4a46ad0()*0.702795);
}

double ddNNEB::synapse0x4a4a900() {
   return (neuron0x4a46e10()*0.919346);
}

double ddNNEB::synapse0x4a4a940() {
   return (neuron0x4a47150()*-0.0825796);
}

double ddNNEB::synapse0x4a4a980() {
   return (neuron0x4a47490()*-0.252398);
}

double ddNNEB::synapse0x4a4a9c0() {
   return (neuron0x4a477d0()*1.93309);
}

double ddNNEB::synapse0x4a4aa00() {
   return (neuron0x4a47b10()*0.402714);
}

double ddNNEB::synapse0x4a4aa40() {
   return (neuron0x4a47e50()*-0.705064);
}

double ddNNEB::synapse0x4193db0() {
   return (neuron0x4a46790()*0.279067);
}

double ddNNEB::synapse0x4193df0() {
   return (neuron0x4a46ad0()*2.10666);
}

double ddNNEB::synapse0x42bf760() {
   return (neuron0x4a46e10()*-1.93947);
}

double ddNNEB::synapse0x42bf7a0() {
   return (neuron0x4a47150()*0.275969);
}

double ddNNEB::synapse0x42bf7e0() {
   return (neuron0x4a47490()*-0.106968);
}

double ddNNEB::synapse0x42bf820() {
   return (neuron0x4a477d0()*-0.688045);
}

double ddNNEB::synapse0x42bf860() {
   return (neuron0x4a47b10()*-0.668713);
}

double ddNNEB::synapse0x42bf8a0() {
   return (neuron0x4a47e50()*1.05971);
}

double ddNNEB::synapse0x4a4b590() {
   return (neuron0x4a46790()*-1.40774);
}

double ddNNEB::synapse0x4a4b5d0() {
   return (neuron0x4a46ad0()*0.313147);
}

double ddNNEB::synapse0x4a4b610() {
   return (neuron0x4a46e10()*-0.734327);
}

double ddNNEB::synapse0x4a4b650() {
   return (neuron0x4a47150()*0.152364);
}

double ddNNEB::synapse0x4a4b690() {
   return (neuron0x4a47490()*0.201345);
}

double ddNNEB::synapse0x4a4b6d0() {
   return (neuron0x4a477d0()*-0.450168);
}

double ddNNEB::synapse0x4a4b710() {
   return (neuron0x4a47b10()*-0.0113884);
}

double ddNNEB::synapse0x4a4b750() {
   return (neuron0x4a47e50()*2.59905);
}

double ddNNEB::synapse0x4a4bad0() {
   return (neuron0x4a482c0()*0.516919);
}

double ddNNEB::synapse0x4a4bb10() {
   return (neuron0x4a486f0()*0.119394);
}

double ddNNEB::synapse0x4a4bb50() {
   return (neuron0x4a48c30()*-1.4767);
}

double ddNNEB::synapse0x4a4bb90() {
   return (neuron0x4a49200()*-0.0500033);
}

double ddNNEB::synapse0x4a4bbd0() {
   return (neuron0x4a49740()*0.298463);
}

double ddNNEB::synapse0x4a4bc10() {
   return (neuron0x4a49ac0()*2.45755);
}

double ddNNEB::synapse0x4a4bc50() {
   return (neuron0x4a4a000()*0.654612);
}

double ddNNEB::synapse0x4a4bc90() {
   return (neuron0x4a4a540()*0.57596);
}

double ddNNEB::synapse0x4a4bcd0() {
   return (neuron0x4a4aa80()*-1.65405);
}

double ddNNEB::synapse0x4a4bd10() {
   return (neuron0x4a4b2e0()*2.42557);
}

double ddNNEB::synapse0x4a4c090() {
   return (neuron0x4a482c0()*0.502582);
}

double ddNNEB::synapse0x4a4c0d0() {
   return (neuron0x4a486f0()*2.8062);
}

double ddNNEB::synapse0x4a4c110() {
   return (neuron0x4a48c30()*-5.0152);
}

double ddNNEB::synapse0x4a4c150() {
   return (neuron0x4a49200()*-0.479722);
}

double ddNNEB::synapse0x4a4c190() {
   return (neuron0x4a49740()*0.762427);
}

double ddNNEB::synapse0x4a4c1d0() {
   return (neuron0x4a49ac0()*0.465059);
}

double ddNNEB::synapse0x4a4c210() {
   return (neuron0x4a4a000()*1.25011);
}

double ddNNEB::synapse0x4a4c250() {
   return (neuron0x4a4a540()*0.848167);
}

double ddNNEB::synapse0x4a4c290() {
   return (neuron0x4a4aa80()*-3.38675);
}

double ddNNEB::synapse0x4a4c2d0() {
   return (neuron0x4a4b2e0()*0.890426);
}

double ddNNEB::synapse0x4a4c650() {
   return (neuron0x4a482c0()*2.11989);
}

double ddNNEB::synapse0x4a4c690() {
   return (neuron0x4a486f0()*-0.554945);
}

double ddNNEB::synapse0x4a4c6d0() {
   return (neuron0x4a48c30()*-0.633086);
}

double ddNNEB::synapse0x4a4c710() {
   return (neuron0x4a49200()*0.574811);
}

double ddNNEB::synapse0x4a4c750() {
   return (neuron0x4a49740()*0.354028);
}

double ddNNEB::synapse0x4a4c790() {
   return (neuron0x4a49ac0()*-0.156698);
}

double ddNNEB::synapse0x4a4c7d0() {
   return (neuron0x4a4a000()*2.36578);
}

double ddNNEB::synapse0x4a4c810() {
   return (neuron0x4a4a540()*0.196586);
}

double ddNNEB::synapse0x4a4c850() {
   return (neuron0x4a4aa80()*1.56175);
}

double ddNNEB::synapse0x4a4c890() {
   return (neuron0x4a4b2e0()*-0.0753345);
}

double ddNNEB::synapse0x4a4cc10() {
   return (neuron0x4a482c0()*1.1924);
}

double ddNNEB::synapse0x4a4cc50() {
   return (neuron0x4a486f0()*0.35832);
}

double ddNNEB::synapse0x4a4cc90() {
   return (neuron0x4a48c30()*-1.20353);
}

double ddNNEB::synapse0x4a4ccd0() {
   return (neuron0x4a49200()*-1.16455);
}

double ddNNEB::synapse0x4a4cd10() {
   return (neuron0x4a49740()*1.49197);
}

double ddNNEB::synapse0x4a4cd50() {
   return (neuron0x4a49ac0()*0.73065);
}

double ddNNEB::synapse0x4a4cd90() {
   return (neuron0x4a4a000()*0.00900645);
}

double ddNNEB::synapse0x4a4cdd0() {
   return (neuron0x4a4a540()*0.682976);
}

double ddNNEB::synapse0x4a4ce10() {
   return (neuron0x4a4aa80()*0.888089);
}

double ddNNEB::synapse0x4a4ce50() {
   return (neuron0x4a4b2e0()*2.13105);
}

double ddNNEB::synapse0x4a4d1d0() {
   return (neuron0x4a482c0()*-0.612113);
}

double ddNNEB::synapse0x4a4d210() {
   return (neuron0x4a486f0()*-0.686453);
}

double ddNNEB::synapse0x4a4d250() {
   return (neuron0x4a48c30()*-2.45879);
}

double ddNNEB::synapse0x4a4d290() {
   return (neuron0x4a49200()*-0.257105);
}

double ddNNEB::synapse0x4a4d2d0() {
   return (neuron0x4a49740()*-0.999997);
}

double ddNNEB::synapse0x4a4d310() {
   return (neuron0x4a49ac0()*1.33032);
}

double ddNNEB::synapse0x4a4d350() {
   return (neuron0x4a4a000()*-0.300316);
}

double ddNNEB::synapse0x4a4d390() {
   return (neuron0x4a4a540()*-0.902281);
}

double ddNNEB::synapse0x4a4d3d0() {
   return (neuron0x4a4aa80()*0.0236984);
}

double ddNNEB::synapse0x4a4aed0() {
   return (neuron0x4a4b2e0()*-0.749823);
}

double ddNNEB::synapse0x4a4b250() {
   return (neuron0x4a4b790()*2.1077);
}

double ddNNEB::synapse0x4a4b290() {
   return (neuron0x4a4bd50()*3.37063);
}

double ddNNEB::synapse0x4a48190() {
   return (neuron0x4a4c310()*4.32082);
}

double ddNNEB::synapse0x4a481d0() {
   return (neuron0x4a4c8d0()*2.21584);
}

double ddNNEB::synapse0x4a48210() {
   return (neuron0x4a4ce90()*-5.43053);
}

