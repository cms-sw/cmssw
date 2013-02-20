#include "ccNNEB.h"
#include <cmath>

double ccNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 1.15492)/1.68616;
   input1 = (in1 - 1.14956)/1.69316;
   input2 = (in2 - 1.89563)/1.49913;
   input3 = (in3 - 1.90666)/1.49442;
   input4 = (in4 - 0.3065)/1.46726;
   input5 = (in5 - 0.318454)/1.50742;
   input6 = (in6 - 0.305354)/1.51455;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x605af50();
     default:
         return 0.;
   }
}

double ccNNEB::Value(int index, double* input) {
   input0 = (input[0] - 1.15492)/1.68616;
   input1 = (input[1] - 1.14956)/1.69316;
   input2 = (input[2] - 1.89563)/1.49913;
   input3 = (input[3] - 1.90666)/1.49442;
   input4 = (input[4] - 0.3065)/1.46726;
   input5 = (input[5] - 0.318454)/1.50742;
   input6 = (input[6] - 0.305354)/1.51455;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x605af50();
     default:
         return 0.;
   }
}

double ccNNEB::neuron0x60567d0() {
   return input0;
}

double ccNNEB::neuron0x6056b10() {
   return input1;
}

double ccNNEB::neuron0x6056e50() {
   return input2;
}

double ccNNEB::neuron0x6057190() {
   return input3;
}

double ccNNEB::neuron0x60574d0() {
   return input4;
}

double ccNNEB::neuron0x6057810() {
   return input5;
}

double ccNNEB::neuron0x6057b50() {
   return input6;
}

double ccNNEB::neuron0x6057e90() {
   return input7;
}

double ccNNEB::input0x6058300() {
   double input = -0.539159;
   input += synapse0x5fb71d0();
   input += synapse0x605f3d0();
   input += synapse0x60585b0();
   input += synapse0x60585f0();
   input += synapse0x6058630();
   input += synapse0x6058670();
   input += synapse0x60586b0();
   input += synapse0x60586f0();
   return input;
}

double ccNNEB::neuron0x6058300() {
   double input = input0x6058300();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x6058730() {
   double input = 1.51054;
   input += synapse0x6058a70();
   input += synapse0x6058ab0();
   input += synapse0x6058af0();
   input += synapse0x6058b30();
   input += synapse0x6058b70();
   input += synapse0x6058bb0();
   input += synapse0x6058bf0();
   input += synapse0x6058c30();
   return input;
}

double ccNNEB::neuron0x6058730() {
   double input = input0x6058730();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x6058c70() {
   double input = -0.0379268;
   input += synapse0x6058fb0();
   input += synapse0x58b5e90();
   input += synapse0x58b5ed0();
   input += synapse0x6059100();
   input += synapse0x6059140();
   input += synapse0x6059180();
   input += synapse0x60591c0();
   input += synapse0x6059200();
   return input;
}

double ccNNEB::neuron0x6058c70() {
   double input = input0x6058c70();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x6059240() {
   double input = 0.463642;
   input += synapse0x6059580();
   input += synapse0x60595c0();
   input += synapse0x6059600();
   input += synapse0x6059640();
   input += synapse0x6059680();
   input += synapse0x60596c0();
   input += synapse0x6059700();
   input += synapse0x6059740();
   return input;
}

double ccNNEB::neuron0x6059240() {
   double input = input0x6059240();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x6059780() {
   double input = -1.54839;
   input += synapse0x6059ac0();
   input += synapse0x6056700();
   input += synapse0x605f410();
   input += synapse0x58b49d0();
   input += synapse0x6058ff0();
   input += synapse0x6059030();
   input += synapse0x6059070();
   input += synapse0x60590b0();
   return input;
}

double ccNNEB::neuron0x6059780() {
   double input = input0x6059780();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x6059b00() {
   double input = -1.25221;
   input += synapse0x6059e40();
   input += synapse0x6059e80();
   input += synapse0x6059ec0();
   input += synapse0x6059f00();
   input += synapse0x6059f40();
   input += synapse0x6059f80();
   input += synapse0x6059fc0();
   input += synapse0x605a000();
   return input;
}

double ccNNEB::neuron0x6059b00() {
   double input = input0x6059b00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605a040() {
   double input = -0.0674997;
   input += synapse0x605a380();
   input += synapse0x605a3c0();
   input += synapse0x605a400();
   input += synapse0x605a440();
   input += synapse0x605a480();
   input += synapse0x605a4c0();
   input += synapse0x605a500();
   input += synapse0x605a540();
   return input;
}

double ccNNEB::neuron0x605a040() {
   double input = input0x605a040();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605a580() {
   double input = -2.0379;
   input += synapse0x605a8c0();
   input += synapse0x605a900();
   input += synapse0x605a940();
   input += synapse0x605a980();
   input += synapse0x605a9c0();
   input += synapse0x605aa00();
   input += synapse0x605aa40();
   input += synapse0x605aa80();
   return input;
}

double ccNNEB::neuron0x605a580() {
   double input = input0x605a580();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605aac0() {
   double input = 2.45266;
   input += synapse0x57a3df0();
   input += synapse0x57a3e30();
   input += synapse0x58cf7a0();
   input += synapse0x58cf7e0();
   input += synapse0x58cf820();
   input += synapse0x58cf860();
   input += synapse0x58cf8a0();
   input += synapse0x58cf8e0();
   return input;
}

double ccNNEB::neuron0x605aac0() {
   double input = input0x605aac0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605b320() {
   double input = 0.575118;
   input += synapse0x605b5d0();
   input += synapse0x605b610();
   input += synapse0x605b650();
   input += synapse0x605b690();
   input += synapse0x605b6d0();
   input += synapse0x605b710();
   input += synapse0x605b750();
   input += synapse0x605b790();
   return input;
}

double ccNNEB::neuron0x605b320() {
   double input = input0x605b320();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605b7d0() {
   double input = -0.834104;
   input += synapse0x605bb10();
   input += synapse0x605bb50();
   input += synapse0x605bb90();
   input += synapse0x605bbd0();
   input += synapse0x605bc10();
   input += synapse0x605bc50();
   input += synapse0x605bc90();
   input += synapse0x605bcd0();
   input += synapse0x605bd10();
   input += synapse0x605bd50();
   return input;
}

double ccNNEB::neuron0x605b7d0() {
   double input = input0x605b7d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605bd90() {
   double input = -0.69539;
   input += synapse0x605c0d0();
   input += synapse0x605c110();
   input += synapse0x605c150();
   input += synapse0x605c190();
   input += synapse0x605c1d0();
   input += synapse0x605c210();
   input += synapse0x605c250();
   input += synapse0x605c290();
   input += synapse0x605c2d0();
   input += synapse0x605c310();
   return input;
}

double ccNNEB::neuron0x605bd90() {
   double input = input0x605bd90();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605c350() {
   double input = -0.879929;
   input += synapse0x605c690();
   input += synapse0x605c6d0();
   input += synapse0x605c710();
   input += synapse0x605c750();
   input += synapse0x605c790();
   input += synapse0x605c7d0();
   input += synapse0x605c810();
   input += synapse0x605c850();
   input += synapse0x605c890();
   input += synapse0x605c8d0();
   return input;
}

double ccNNEB::neuron0x605c350() {
   double input = input0x605c350();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605c910() {
   double input = 0.56029;
   input += synapse0x605cc50();
   input += synapse0x605cc90();
   input += synapse0x605ccd0();
   input += synapse0x605cd10();
   input += synapse0x605cd50();
   input += synapse0x605cd90();
   input += synapse0x605cdd0();
   input += synapse0x605ce10();
   input += synapse0x605ce50();
   input += synapse0x605ce90();
   return input;
}

double ccNNEB::neuron0x605c910() {
   double input = input0x605c910();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605ced0() {
   double input = 0.386744;
   input += synapse0x605d210();
   input += synapse0x605d250();
   input += synapse0x605d290();
   input += synapse0x605d2d0();
   input += synapse0x605d310();
   input += synapse0x605d350();
   input += synapse0x605d390();
   input += synapse0x605d3d0();
   input += synapse0x605d410();
   input += synapse0x605af10();
   return input;
}

double ccNNEB::neuron0x605ced0() {
   double input = input0x605ced0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEB::input0x605af50() {
   double input = 0.868956;
   input += synapse0x605b290();
   input += synapse0x605b2d0();
   input += synapse0x60581d0();
   input += synapse0x6058210();
   input += synapse0x6058250();
   return input;
}

double ccNNEB::neuron0x605af50() {
   double input = input0x605af50();
   return (input * 1)+0;
}

double ccNNEB::synapse0x5fb71d0() {
   return (neuron0x60567d0()*-0.0699207);
}

double ccNNEB::synapse0x605f3d0() {
   return (neuron0x6056b10()*1.41084);
}

double ccNNEB::synapse0x60585b0() {
   return (neuron0x6056e50()*0.0620302);
}

double ccNNEB::synapse0x60585f0() {
   return (neuron0x6057190()*0.0458559);
}

double ccNNEB::synapse0x6058630() {
   return (neuron0x60574d0()*0.535032);
}

double ccNNEB::synapse0x6058670() {
   return (neuron0x6057810()*0.522631);
}

double ccNNEB::synapse0x60586b0() {
   return (neuron0x6057b50()*1.0977);
}

double ccNNEB::synapse0x60586f0() {
   return (neuron0x6057e90()*0.623202);
}

double ccNNEB::synapse0x6058a70() {
   return (neuron0x60567d0()*1.58823);
}

double ccNNEB::synapse0x6058ab0() {
   return (neuron0x6056b10()*2.37636);
}

double ccNNEB::synapse0x6058af0() {
   return (neuron0x6056e50()*-0.234837);
}

double ccNNEB::synapse0x6058b30() {
   return (neuron0x6057190()*0.0246192);
}

double ccNNEB::synapse0x6058b70() {
   return (neuron0x60574d0()*-1.50138);
}

double ccNNEB::synapse0x6058bb0() {
   return (neuron0x6057810()*0.0862727);
}

double ccNNEB::synapse0x6058bf0() {
   return (neuron0x6057b50()*0.0195872);
}

double ccNNEB::synapse0x6058c30() {
   return (neuron0x6057e90()*-1.47387);
}

double ccNNEB::synapse0x6058fb0() {
   return (neuron0x60567d0()*0.795657);
}

double ccNNEB::synapse0x58b5e90() {
   return (neuron0x6056b10()*0.276677);
}

double ccNNEB::synapse0x58b5ed0() {
   return (neuron0x6056e50()*-0.0981622);
}

double ccNNEB::synapse0x6059100() {
   return (neuron0x6057190()*0.192442);
}

double ccNNEB::synapse0x6059140() {
   return (neuron0x60574d0()*0.222694);
}

double ccNNEB::synapse0x6059180() {
   return (neuron0x6057810()*1.06613);
}

double ccNNEB::synapse0x60591c0() {
   return (neuron0x6057b50()*0.514147);
}

double ccNNEB::synapse0x6059200() {
   return (neuron0x6057e90()*0.145554);
}

double ccNNEB::synapse0x6059580() {
   return (neuron0x60567d0()*0.627626);
}

double ccNNEB::synapse0x60595c0() {
   return (neuron0x6056b10()*0.437069);
}

double ccNNEB::synapse0x6059600() {
   return (neuron0x6056e50()*0.264892);
}

double ccNNEB::synapse0x6059640() {
   return (neuron0x6057190()*0.190372);
}

double ccNNEB::synapse0x6059680() {
   return (neuron0x60574d0()*-1.01744);
}

double ccNNEB::synapse0x60596c0() {
   return (neuron0x6057810()*-0.611537);
}

double ccNNEB::synapse0x6059700() {
   return (neuron0x6057b50()*-0.0684924);
}

double ccNNEB::synapse0x6059740() {
   return (neuron0x6057e90()*-0.441747);
}

double ccNNEB::synapse0x6059ac0() {
   return (neuron0x60567d0()*-0.936029);
}

double ccNNEB::synapse0x6056700() {
   return (neuron0x6056b10()*0.281874);
}

double ccNNEB::synapse0x605f410() {
   return (neuron0x6056e50()*-0.559508);
}

double ccNNEB::synapse0x58b49d0() {
   return (neuron0x6057190()*0.333256);
}

double ccNNEB::synapse0x6058ff0() {
   return (neuron0x60574d0()*0.898024);
}

double ccNNEB::synapse0x6059030() {
   return (neuron0x6057810()*0.700511);
}

double ccNNEB::synapse0x6059070() {
   return (neuron0x6057b50()*-1.07369);
}

double ccNNEB::synapse0x60590b0() {
   return (neuron0x6057e90()*-0.0370767);
}

double ccNNEB::synapse0x6059e40() {
   return (neuron0x60567d0()*0.464066);
}

double ccNNEB::synapse0x6059e80() {
   return (neuron0x6056b10()*-0.470511);
}

double ccNNEB::synapse0x6059ec0() {
   return (neuron0x6056e50()*0.165462);
}

double ccNNEB::synapse0x6059f00() {
   return (neuron0x6057190()*0.0436253);
}

double ccNNEB::synapse0x6059f40() {
   return (neuron0x60574d0()*-0.565435);
}

double ccNNEB::synapse0x6059f80() {
   return (neuron0x6057810()*0.386292);
}

double ccNNEB::synapse0x6059fc0() {
   return (neuron0x6057b50()*0.468514);
}

double ccNNEB::synapse0x605a000() {
   return (neuron0x6057e90()*-1.09823);
}

double ccNNEB::synapse0x605a380() {
   return (neuron0x60567d0()*-0.830445);
}

double ccNNEB::synapse0x605a3c0() {
   return (neuron0x6056b10()*-0.856368);
}

double ccNNEB::synapse0x605a400() {
   return (neuron0x6056e50()*-0.324607);
}

double ccNNEB::synapse0x605a440() {
   return (neuron0x6057190()*-0.479792);
}

double ccNNEB::synapse0x605a480() {
   return (neuron0x60574d0()*-0.310053);
}

double ccNNEB::synapse0x605a4c0() {
   return (neuron0x6057810()*-0.211249);
}

double ccNNEB::synapse0x605a500() {
   return (neuron0x6057b50()*-0.663044);
}

double ccNNEB::synapse0x605a540() {
   return (neuron0x6057e90()*-0.891015);
}

double ccNNEB::synapse0x605a8c0() {
   return (neuron0x60567d0()*0.380276);
}

double ccNNEB::synapse0x605a900() {
   return (neuron0x6056b10()*0.496164);
}

double ccNNEB::synapse0x605a940() {
   return (neuron0x6056e50()*-0.396836);
}

double ccNNEB::synapse0x605a980() {
   return (neuron0x6057190()*-0.67967);
}

double ccNNEB::synapse0x605a9c0() {
   return (neuron0x60574d0()*0.668008);
}

double ccNNEB::synapse0x605aa00() {
   return (neuron0x6057810()*-0.598866);
}

double ccNNEB::synapse0x605aa40() {
   return (neuron0x6057b50()*-0.646267);
}

double ccNNEB::synapse0x605aa80() {
   return (neuron0x6057e90()*0.397705);
}

double ccNNEB::synapse0x57a3df0() {
   return (neuron0x60567d0()*1.48193);
}

double ccNNEB::synapse0x57a3e30() {
   return (neuron0x6056b10()*0.140198);
}

double ccNNEB::synapse0x58cf7a0() {
   return (neuron0x6056e50()*-0.126392);
}

double ccNNEB::synapse0x58cf7e0() {
   return (neuron0x6057190()*-0.202857);
}

double ccNNEB::synapse0x58cf820() {
   return (neuron0x60574d0()*0.457037);
}

double ccNNEB::synapse0x58cf860() {
   return (neuron0x6057810()*-0.0149029);
}

double ccNNEB::synapse0x58cf8a0() {
   return (neuron0x6057b50()*-0.200188);
}

double ccNNEB::synapse0x58cf8e0() {
   return (neuron0x6057e90()*-0.131662);
}

double ccNNEB::synapse0x605b5d0() {
   return (neuron0x60567d0()*1.24678);
}

double ccNNEB::synapse0x605b610() {
   return (neuron0x6056b10()*0.100963);
}

double ccNNEB::synapse0x605b650() {
   return (neuron0x6056e50()*0.252164);
}

double ccNNEB::synapse0x605b690() {
   return (neuron0x6057190()*0.470284);
}

double ccNNEB::synapse0x605b6d0() {
   return (neuron0x60574d0()*1.31056);
}

double ccNNEB::synapse0x605b710() {
   return (neuron0x6057810()*1.2156);
}

double ccNNEB::synapse0x605b750() {
   return (neuron0x6057b50()*0.579176);
}

double ccNNEB::synapse0x605b790() {
   return (neuron0x6057e90()*0.94709);
}

double ccNNEB::synapse0x605bb10() {
   return (neuron0x6058300()*-0.207301);
}

double ccNNEB::synapse0x605bb50() {
   return (neuron0x6058730()*0.395181);
}

double ccNNEB::synapse0x605bb90() {
   return (neuron0x6058c70()*0.0484047);
}

double ccNNEB::synapse0x605bbd0() {
   return (neuron0x6059240()*-1.1885);
}

double ccNNEB::synapse0x605bc10() {
   return (neuron0x6059780()*-0.0594185);
}

double ccNNEB::synapse0x605bc50() {
   return (neuron0x6059b00()*-1.59126);
}

double ccNNEB::synapse0x605bc90() {
   return (neuron0x605a040()*-0.136816);
}

double ccNNEB::synapse0x605bcd0() {
   return (neuron0x605a580()*-0.647761);
}

double ccNNEB::synapse0x605bd10() {
   return (neuron0x605aac0()*0.974329);
}

double ccNNEB::synapse0x605bd50() {
   return (neuron0x605b320()*0.237124);
}

double ccNNEB::synapse0x605c0d0() {
   return (neuron0x6058300()*-0.306592);
}

double ccNNEB::synapse0x605c110() {
   return (neuron0x6058730()*1.53717);
}

double ccNNEB::synapse0x605c150() {
   return (neuron0x6058c70()*-0.417984);
}

double ccNNEB::synapse0x605c190() {
   return (neuron0x6059240()*-2.71232);
}

double ccNNEB::synapse0x605c1d0() {
   return (neuron0x6059780()*-1.17791);
}

double ccNNEB::synapse0x605c210() {
   return (neuron0x6059b00()*-1.59499);
}

double ccNNEB::synapse0x605c250() {
   return (neuron0x605a040()*-0.572163);
}

double ccNNEB::synapse0x605c290() {
   return (neuron0x605a580()*-1.13002);
}

double ccNNEB::synapse0x605c2d0() {
   return (neuron0x605aac0()*1.7962);
}

double ccNNEB::synapse0x605c310() {
   return (neuron0x605b320()*0.39981);
}

double ccNNEB::synapse0x605c690() {
   return (neuron0x6058300()*-0.0395236);
}

double ccNNEB::synapse0x605c6d0() {
   return (neuron0x6058730()*0.94403);
}

double ccNNEB::synapse0x605c710() {
   return (neuron0x6058c70()*-0.448583);
}

double ccNNEB::synapse0x605c750() {
   return (neuron0x6059240()*-2.71095);
}

double ccNNEB::synapse0x605c790() {
   return (neuron0x6059780()*-1.14823);
}

double ccNNEB::synapse0x605c7d0() {
   return (neuron0x6059b00()*-1.33223);
}

double ccNNEB::synapse0x605c810() {
   return (neuron0x605a040()*-0.671872);
}

double ccNNEB::synapse0x605c850() {
   return (neuron0x605a580()*-1.26958);
}

double ccNNEB::synapse0x605c890() {
   return (neuron0x605aac0()*2.30656);
}

double ccNNEB::synapse0x605c8d0() {
   return (neuron0x605b320()*-0.383468);
}

double ccNNEB::synapse0x605cc50() {
   return (neuron0x6058300()*0.146573);
}

double ccNNEB::synapse0x605cc90() {
   return (neuron0x6058730()*1.58013);
}

double ccNNEB::synapse0x605ccd0() {
   return (neuron0x6058c70()*-0.619888);
}

double ccNNEB::synapse0x605cd10() {
   return (neuron0x6059240()*0.990389);
}

double ccNNEB::synapse0x605cd50() {
   return (neuron0x6059780()*-0.785703);
}

double ccNNEB::synapse0x605cd90() {
   return (neuron0x6059b00()*-1.15902);
}

double ccNNEB::synapse0x605cdd0() {
   return (neuron0x605a040()*-0.218559);
}

double ccNNEB::synapse0x605ce10() {
   return (neuron0x605a580()*-0.805096);
}

double ccNNEB::synapse0x605ce50() {
   return (neuron0x605aac0()*1.00191);
}

double ccNNEB::synapse0x605ce90() {
   return (neuron0x605b320()*-0.73646);
}

double ccNNEB::synapse0x605d210() {
   return (neuron0x6058300()*1.10806);
}

double ccNNEB::synapse0x605d250() {
   return (neuron0x6058730()*-0.320801);
}

double ccNNEB::synapse0x605d290() {
   return (neuron0x6058c70()*0.0299629);
}

double ccNNEB::synapse0x605d2d0() {
   return (neuron0x6059240()*0.160331);
}

double ccNNEB::synapse0x605d310() {
   return (neuron0x6059780()*0.333674);
}

double ccNNEB::synapse0x605d350() {
   return (neuron0x6059b00()*0.6104);
}

double ccNNEB::synapse0x605d390() {
   return (neuron0x605a040()*-0.690923);
}

double ccNNEB::synapse0x605d3d0() {
   return (neuron0x605a580()*0.800533);
}

double ccNNEB::synapse0x605d410() {
   return (neuron0x605aac0()*-0.335794);
}

double ccNNEB::synapse0x605af10() {
   return (neuron0x605b320()*-0.00422044);
}

double ccNNEB::synapse0x605b290() {
   return (neuron0x605b7d0()*1.76016);
}

double ccNNEB::synapse0x605b2d0() {
   return (neuron0x605bd90()*3.82695);
}

double ccNNEB::synapse0x60581d0() {
   return (neuron0x605c350()*3.97341);
}

double ccNNEB::synapse0x6058210() {
   return (neuron0x605c910()*1.94878);
}

double ccNNEB::synapse0x6058250() {
   return (neuron0x605ced0()*-1.83486);
}

