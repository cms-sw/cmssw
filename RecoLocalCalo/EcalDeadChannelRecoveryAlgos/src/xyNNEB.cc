#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/xyNNEB.h"
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


double ldNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.15492)/1.68616;
   input2 = (in2 - 1.14956)/1.69316;
   input3 = (in3 - 1.89563)/1.49913;
   input4 = (in4 - 1.90666)/1.49442;
   input5 = (in5 - 0.3065)/1.46726;
   input6 = (in6 - 0.318454)/1.50742;
   input7 = (in7 - 0.305354)/1.51455;
   switch(index) {
     case 0:
         return neuron0x1827a010();
     default:
         return 0.;
   }
}

double ldNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.15492)/1.68616;
   input2 = (input[2] - 1.14956)/1.69316;
   input3 = (input[3] - 1.89563)/1.49913;
   input4 = (input[4] - 1.90666)/1.49442;
   input5 = (input[5] - 0.3065)/1.46726;
   input6 = (input[6] - 0.318454)/1.50742;
   input7 = (input[7] - 0.305354)/1.51455;
   switch(index) {
     case 0:
         return neuron0x1827a010();
     default:
         return 0.;
   }
}

double ldNNEB::neuron0x18275890() {
   return input0;
}

double ldNNEB::neuron0x18275bd0() {
   return input1;
}

double ldNNEB::neuron0x18275f10() {
   return input2;
}

double ldNNEB::neuron0x18276250() {
   return input3;
}

double ldNNEB::neuron0x18276590() {
   return input4;
}

double ldNNEB::neuron0x182768d0() {
   return input5;
}

double ldNNEB::neuron0x18276c10() {
   return input6;
}

double ldNNEB::neuron0x18276f50() {
   return input7;
}

double ldNNEB::input0x182773c0() {
   double input = -0.900178;
   input += synapse0x181d6290();
   input += synapse0x1827e490();
   input += synapse0x18277670();
   input += synapse0x182776b0();
   input += synapse0x182776f0();
   input += synapse0x18277730();
   input += synapse0x18277770();
   input += synapse0x182777b0();
   return input;
}

double ldNNEB::neuron0x182773c0() {
   double input = input0x182773c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x182777f0() {
   double input = 2.16276;
   input += synapse0x18277b30();
   input += synapse0x18277b70();
   input += synapse0x18277bb0();
   input += synapse0x18277bf0();
   input += synapse0x18277c30();
   input += synapse0x18277c70();
   input += synapse0x18277cb0();
   input += synapse0x18277cf0();
   return input;
}

double ldNNEB::neuron0x182777f0() {
   double input = input0x182777f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x18277d30() {
   double input = 0.152496;
   input += synapse0x18278070();
   input += synapse0x17ad4f50();
   input += synapse0x17ad4f90();
   input += synapse0x182781c0();
   input += synapse0x18278200();
   input += synapse0x18278240();
   input += synapse0x18278280();
   input += synapse0x182782c0();
   return input;
}

double ldNNEB::neuron0x18277d30() {
   double input = input0x18277d30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x18278300() {
   double input = -0.320593;
   input += synapse0x18278640();
   input += synapse0x18278680();
   input += synapse0x182786c0();
   input += synapse0x18278700();
   input += synapse0x18278740();
   input += synapse0x18278780();
   input += synapse0x182787c0();
   input += synapse0x18278800();
   return input;
}

double ldNNEB::neuron0x18278300() {
   double input = input0x18278300();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x18278840() {
   double input = -0.58249;
   input += synapse0x18278b80();
   input += synapse0x182757c0();
   input += synapse0x1827e4d0();
   input += synapse0x17ad3a90();
   input += synapse0x182780b0();
   input += synapse0x182780f0();
   input += synapse0x18278130();
   input += synapse0x18278170();
   return input;
}

double ldNNEB::neuron0x18278840() {
   double input = input0x18278840();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x18278bc0() {
   double input = 0.173697;
   input += synapse0x18278f00();
   input += synapse0x18278f40();
   input += synapse0x18278f80();
   input += synapse0x18278fc0();
   input += synapse0x18279000();
   input += synapse0x18279040();
   input += synapse0x18279080();
   input += synapse0x182790c0();
   return input;
}

double ldNNEB::neuron0x18278bc0() {
   double input = input0x18278bc0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x18279100() {
   double input = -1.94555;
   input += synapse0x18279440();
   input += synapse0x18279480();
   input += synapse0x182794c0();
   input += synapse0x18279500();
   input += synapse0x18279540();
   input += synapse0x18279580();
   input += synapse0x182795c0();
   input += synapse0x18279600();
   return input;
}

double ldNNEB::neuron0x18279100() {
   double input = input0x18279100();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x18279640() {
   double input = 0.545632;
   input += synapse0x18279980();
   input += synapse0x182799c0();
   input += synapse0x18279a00();
   input += synapse0x18279a40();
   input += synapse0x18279a80();
   input += synapse0x18279ac0();
   input += synapse0x18279b00();
   input += synapse0x18279b40();
   return input;
}

double ldNNEB::neuron0x18279640() {
   double input = input0x18279640();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x18279b80() {
   double input = -0.563904;
   input += synapse0x179c2eb0();
   input += synapse0x179c2ef0();
   input += synapse0x17aee860();
   input += synapse0x17aee8a0();
   input += synapse0x17aee8e0();
   input += synapse0x17aee920();
   input += synapse0x17aee960();
   input += synapse0x17aee9a0();
   return input;
}

double ldNNEB::neuron0x18279b80() {
   double input = input0x18279b80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x1827a3e0() {
   double input = 2.57453;
   input += synapse0x1827a690();
   input += synapse0x1827a6d0();
   input += synapse0x1827a710();
   input += synapse0x1827a750();
   input += synapse0x1827a790();
   input += synapse0x1827a7d0();
   input += synapse0x1827a810();
   input += synapse0x1827a850();
   return input;
}

double ldNNEB::neuron0x1827a3e0() {
   double input = input0x1827a3e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x1827a890() {
   double input = 1.03566;
   input += synapse0x1827abd0();
   input += synapse0x1827ac10();
   input += synapse0x1827ac50();
   input += synapse0x1827ac90();
   input += synapse0x1827acd0();
   input += synapse0x1827ad10();
   input += synapse0x1827ad50();
   input += synapse0x1827ad90();
   input += synapse0x1827add0();
   input += synapse0x1827ae10();
   return input;
}

double ldNNEB::neuron0x1827a890() {
   double input = input0x1827a890();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x1827ae50() {
   double input = -0.093052;
   input += synapse0x1827b190();
   input += synapse0x1827b1d0();
   input += synapse0x1827b210();
   input += synapse0x1827b250();
   input += synapse0x1827b290();
   input += synapse0x1827b2d0();
   input += synapse0x1827b310();
   input += synapse0x1827b350();
   input += synapse0x1827b390();
   input += synapse0x1827b3d0();
   return input;
}

double ldNNEB::neuron0x1827ae50() {
   double input = input0x1827ae50();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x1827b410() {
   double input = 0.160093;
   input += synapse0x1827b750();
   input += synapse0x1827b790();
   input += synapse0x1827b7d0();
   input += synapse0x1827b810();
   input += synapse0x1827b850();
   input += synapse0x1827b890();
   input += synapse0x1827b8d0();
   input += synapse0x1827b910();
   input += synapse0x1827b950();
   input += synapse0x1827b990();
   return input;
}

double ldNNEB::neuron0x1827b410() {
   double input = input0x1827b410();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x1827b9d0() {
   double input = -0.193108;
   input += synapse0x1827bd10();
   input += synapse0x1827bd50();
   input += synapse0x1827bd90();
   input += synapse0x1827bdd0();
   input += synapse0x1827be10();
   input += synapse0x1827be50();
   input += synapse0x1827be90();
   input += synapse0x1827bed0();
   input += synapse0x1827bf10();
   input += synapse0x1827bf50();
   return input;
}

double ldNNEB::neuron0x1827b9d0() {
   double input = input0x1827b9d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x1827bf90() {
   double input = -0.780892;
   input += synapse0x1827c2d0();
   input += synapse0x1827c310();
   input += synapse0x1827c350();
   input += synapse0x1827c390();
   input += synapse0x1827c3d0();
   input += synapse0x1827c410();
   input += synapse0x1827c450();
   input += synapse0x1827c490();
   input += synapse0x1827c4d0();
   input += synapse0x18279fd0();
   return input;
}

double ldNNEB::neuron0x1827bf90() {
   double input = input0x1827bf90();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEB::input0x1827a010() {
   double input = 3.54463;
   input += synapse0x1827a350();
   input += synapse0x1827a390();
   input += synapse0x18277290();
   input += synapse0x182772d0();
   input += synapse0x18277310();
   return input;
}

double ldNNEB::neuron0x1827a010() {
   double input = input0x1827a010();
   return (input * 1)+0;
}

double ldNNEB::synapse0x181d6290() {
   return (neuron0x18275890()*0.614766);
}

double ldNNEB::synapse0x1827e490() {
   return (neuron0x18275bd0()*0.413269);
}

double ldNNEB::synapse0x18277670() {
   return (neuron0x18275f10()*-1.73024);
}

double ldNNEB::synapse0x182776b0() {
   return (neuron0x18276250()*0.00373702);
}

double ldNNEB::synapse0x182776f0() {
   return (neuron0x18276590()*0.106991);
}

double ldNNEB::synapse0x18277730() {
   return (neuron0x182768d0()*-0.449699);
}

double ldNNEB::synapse0x18277770() {
   return (neuron0x18276c10()*-0.00781349);
}

double ldNNEB::synapse0x182777b0() {
   return (neuron0x18276f50()*0.339282);
}

double ldNNEB::synapse0x18277b30() {
   return (neuron0x18275890()*0.0803725);
}

double ldNNEB::synapse0x18277b70() {
   return (neuron0x18275bd0()*-0.332201);
}

double ldNNEB::synapse0x18277bb0() {
   return (neuron0x18275f10()*-0.0784683);
}

double ldNNEB::synapse0x18277bf0() {
   return (neuron0x18276250()*-0.0932347);
}

double ldNNEB::synapse0x18277c30() {
   return (neuron0x18276590()*0.990996);
}

double ldNNEB::synapse0x18277c70() {
   return (neuron0x182768d0()*0.456115);
}

double ldNNEB::synapse0x18277cb0() {
   return (neuron0x18276c10()*-0.287584);
}

double ldNNEB::synapse0x18277cf0() {
   return (neuron0x18276f50()*-0.0594785);
}

double ldNNEB::synapse0x18278070() {
   return (neuron0x18275890()*1.15983);
}

double ldNNEB::synapse0x17ad4f50() {
   return (neuron0x18275bd0()*0.710525);
}

double ldNNEB::synapse0x17ad4f90() {
   return (neuron0x18275f10()*-0.994761);
}

double ldNNEB::synapse0x182781c0() {
   return (neuron0x18276250()*0.271231);
}

double ldNNEB::synapse0x18278200() {
   return (neuron0x18276590()*-0.140435);
}

double ldNNEB::synapse0x18278240() {
   return (neuron0x182768d0()*-0.688977);
}

double ldNNEB::synapse0x18278280() {
   return (neuron0x18276c10()*0.106436);
}

double ldNNEB::synapse0x182782c0() {
   return (neuron0x18276f50()*0.390168);
}

double ldNNEB::synapse0x18278640() {
   return (neuron0x18275890()*0.257245);
}

double ldNNEB::synapse0x18278680() {
   return (neuron0x18275bd0()*0.258381);
}

double ldNNEB::synapse0x182786c0() {
   return (neuron0x18275f10()*0.33651);
}

double ldNNEB::synapse0x18278700() {
   return (neuron0x18276250()*1.93473);
}

double ldNNEB::synapse0x18278740() {
   return (neuron0x18276590()*0.00886707);
}

double ldNNEB::synapse0x18278780() {
   return (neuron0x182768d0()*-0.527713);
}

double ldNNEB::synapse0x182787c0() {
   return (neuron0x18276c10()*0.369871);
}

double ldNNEB::synapse0x18278800() {
   return (neuron0x18276f50()*0.494573);
}

double ldNNEB::synapse0x18278b80() {
   return (neuron0x18275890()*-0.888076);
}

double ldNNEB::synapse0x182757c0() {
   return (neuron0x18275bd0()*-0.592806);
}

double ldNNEB::synapse0x1827e4d0() {
   return (neuron0x18275f10()*-1.22797);
}

double ldNNEB::synapse0x17ad3a90() {
   return (neuron0x18276250()*-0.220221);
}

double ldNNEB::synapse0x182780b0() {
   return (neuron0x18276590()*-0.481517);
}

double ldNNEB::synapse0x182780f0() {
   return (neuron0x182768d0()*-0.528706);
}

double ldNNEB::synapse0x18278130() {
   return (neuron0x18276c10()*-0.902304);
}

double ldNNEB::synapse0x18278170() {
   return (neuron0x18276f50()*-1.31047);
}

double ldNNEB::synapse0x18278f00() {
   return (neuron0x18275890()*1.18818);
}

double ldNNEB::synapse0x18278f40() {
   return (neuron0x18275bd0()*0.549287);
}

double ldNNEB::synapse0x18278f80() {
   return (neuron0x18275f10()*0.812188);
}

double ldNNEB::synapse0x18278fc0() {
   return (neuron0x18276250()*-0.692576);
}

double ldNNEB::synapse0x18279000() {
   return (neuron0x18276590()*0.0828682);
}

double ldNNEB::synapse0x18279040() {
   return (neuron0x182768d0()*0.301323);
}

double ldNNEB::synapse0x18279080() {
   return (neuron0x18276c10()*0.259526);
}

double ldNNEB::synapse0x182790c0() {
   return (neuron0x18276f50()*0.337679);
}

double ldNNEB::synapse0x18279440() {
   return (neuron0x18275890()*-1.17139);
}

double ldNNEB::synapse0x18279480() {
   return (neuron0x18275bd0()*0.410552);
}

double ldNNEB::synapse0x182794c0() {
   return (neuron0x18275f10()*0.588709);
}

double ldNNEB::synapse0x18279500() {
   return (neuron0x18276250()*0.0275081);
}

double ldNNEB::synapse0x18279540() {
   return (neuron0x18276590()*0.547308);
}

double ldNNEB::synapse0x18279580() {
   return (neuron0x182768d0()*-0.0396708);
}

double ldNNEB::synapse0x182795c0() {
   return (neuron0x18276c10()*0.0315642);
}

double ldNNEB::synapse0x18279600() {
   return (neuron0x18276f50()*0.400394);
}

double ldNNEB::synapse0x18279980() {
   return (neuron0x18275890()*-0.606966);
}

double ldNNEB::synapse0x182799c0() {
   return (neuron0x18275bd0()*1.26538);
}

double ldNNEB::synapse0x18279a00() {
   return (neuron0x18275f10()*-0.145802);
}

double ldNNEB::synapse0x18279a40() {
   return (neuron0x18276250()*0.203945);
}

double ldNNEB::synapse0x18279a80() {
   return (neuron0x18276590()*-0.376903);
}

double ldNNEB::synapse0x18279ac0() {
   return (neuron0x182768d0()*0.521134);
}

double ldNNEB::synapse0x18279b00() {
   return (neuron0x18276c10()*-0.0731846);
}

double ldNNEB::synapse0x18279b40() {
   return (neuron0x18276f50()*0.179746);
}

double ldNNEB::synapse0x179c2eb0() {
   return (neuron0x18275890()*-0.221377);
}

double ldNNEB::synapse0x179c2ef0() {
   return (neuron0x18275bd0()*0.42709);
}

double ldNNEB::synapse0x17aee860() {
   return (neuron0x18275f10()*-0.35336);
}

double ldNNEB::synapse0x17aee8a0() {
   return (neuron0x18276250()*-0.718037);
}

double ldNNEB::synapse0x17aee8e0() {
   return (neuron0x18276590()*-0.127489);
}

double ldNNEB::synapse0x17aee920() {
   return (neuron0x182768d0()*-0.662367);
}

double ldNNEB::synapse0x17aee960() {
   return (neuron0x18276c10()*0.365699);
}

double ldNNEB::synapse0x17aee9a0() {
   return (neuron0x18276f50()*-0.936714);
}

double ldNNEB::synapse0x1827a690() {
   return (neuron0x18275890()*-1.4265);
}

double ldNNEB::synapse0x1827a6d0() {
   return (neuron0x18275bd0()*-0.460301);
}

double ldNNEB::synapse0x1827a710() {
   return (neuron0x18275f10()*-0.054207);
}

double ldNNEB::synapse0x1827a750() {
   return (neuron0x18276250()*-0.122564);
}

double ldNNEB::synapse0x1827a790() {
   return (neuron0x18276590()*-0.0448578);
}

double ldNNEB::synapse0x1827a7d0() {
   return (neuron0x182768d0()*0.636673);
}

double ldNNEB::synapse0x1827a810() {
   return (neuron0x18276c10()*0.200266);
}

double ldNNEB::synapse0x1827a850() {
   return (neuron0x18276f50()*0.284904);
}

double ldNNEB::synapse0x1827abd0() {
   return (neuron0x182773c0()*-0.96263);
}

double ldNNEB::synapse0x1827ac10() {
   return (neuron0x182777f0()*5.42215);
}

double ldNNEB::synapse0x1827ac50() {
   return (neuron0x18277d30()*-1.06695);
}

double ldNNEB::synapse0x1827ac90() {
   return (neuron0x18278300()*-0.724369);
}

double ldNNEB::synapse0x1827acd0() {
   return (neuron0x18278840()*0.0415235);
}

double ldNNEB::synapse0x1827ad10() {
   return (neuron0x18278bc0()*-1.11119);
}

double ldNNEB::synapse0x1827ad50() {
   return (neuron0x18279100()*3.29088);
}

double ldNNEB::synapse0x1827ad90() {
   return (neuron0x18279640()*-3.47163);
}

double ldNNEB::synapse0x1827add0() {
   return (neuron0x18279b80()*1.96684);
}

double ldNNEB::synapse0x1827ae10() {
   return (neuron0x1827a3e0()*-5.05257);
}

double ldNNEB::synapse0x1827b190() {
   return (neuron0x182773c0()*0.169895);
}

double ldNNEB::synapse0x1827b1d0() {
   return (neuron0x182777f0()*-1.03492);
}

double ldNNEB::synapse0x1827b210() {
   return (neuron0x18277d30()*1.0047);
}

double ldNNEB::synapse0x1827b250() {
   return (neuron0x18278300()*0.541598);
}

double ldNNEB::synapse0x1827b290() {
   return (neuron0x18278840()*-0.540631);
}

double ldNNEB::synapse0x1827b2d0() {
   return (neuron0x18278bc0()*0.368238);
}

double ldNNEB::synapse0x1827b310() {
   return (neuron0x18279100()*-0.540273);
}

double ldNNEB::synapse0x1827b350() {
   return (neuron0x18279640()*1.34685);
}

double ldNNEB::synapse0x1827b390() {
   return (neuron0x18279b80()*-0.429885);
}

double ldNNEB::synapse0x1827b3d0() {
   return (neuron0x1827a3e0()*1.91775);
}

double ldNNEB::synapse0x1827b750() {
   return (neuron0x182773c0()*1.12693);
}

double ldNNEB::synapse0x1827b790() {
   return (neuron0x182777f0()*-2.70427);
}

double ldNNEB::synapse0x1827b7d0() {
   return (neuron0x18277d30()*1.87916);
}

double ldNNEB::synapse0x1827b810() {
   return (neuron0x18278300()*0.12951);
}

double ldNNEB::synapse0x1827b850() {
   return (neuron0x18278840()*0.0521041);
}

double ldNNEB::synapse0x1827b890() {
   return (neuron0x18278bc0()*-0.480644);
}

double ldNNEB::synapse0x1827b8d0() {
   return (neuron0x18279100()*-2.42394);
}

double ldNNEB::synapse0x1827b910() {
   return (neuron0x18279640()*1.85898);
}

double ldNNEB::synapse0x1827b950() {
   return (neuron0x18279b80()*-0.281259);
}

double ldNNEB::synapse0x1827b990() {
   return (neuron0x1827a3e0()*3.24442);
}

double ldNNEB::synapse0x1827bd10() {
   return (neuron0x182773c0()*-0.661399);
}

double ldNNEB::synapse0x1827bd50() {
   return (neuron0x182777f0()*0.69702);
}

double ldNNEB::synapse0x1827bd90() {
   return (neuron0x18277d30()*1.11072);
}

double ldNNEB::synapse0x1827bdd0() {
   return (neuron0x18278300()*0.36713);
}

double ldNNEB::synapse0x1827be10() {
   return (neuron0x18278840()*-0.165389);
}

double ldNNEB::synapse0x1827be50() {
   return (neuron0x18278bc0()*0.351639);
}

double ldNNEB::synapse0x1827be90() {
   return (neuron0x18279100()*0.817737);
}

double ldNNEB::synapse0x1827bed0() {
   return (neuron0x18279640()*-0.109831);
}

double ldNNEB::synapse0x1827bf10() {
   return (neuron0x18279b80()*-0.545354);
}

double ldNNEB::synapse0x1827bf50() {
   return (neuron0x1827a3e0()*-0.635203);
}

double ldNNEB::synapse0x1827c2d0() {
   return (neuron0x182773c0()*1.57996);
}

double ldNNEB::synapse0x1827c310() {
   return (neuron0x182777f0()*-1.28325);
}

double ldNNEB::synapse0x1827c350() {
   return (neuron0x18277d30()*-1.39606);
}

double ldNNEB::synapse0x1827c390() {
   return (neuron0x18278300()*-0.0116294);
}

double ldNNEB::synapse0x1827c3d0() {
   return (neuron0x18278840()*-0.373634);
}

double ldNNEB::synapse0x1827c410() {
   return (neuron0x18278bc0()*0.015702);
}

double ldNNEB::synapse0x1827c450() {
   return (neuron0x18279100()*-1.86785);
}

double ldNNEB::synapse0x1827c490() {
   return (neuron0x18279640()*0.902913);
}

double ldNNEB::synapse0x1827c4d0() {
   return (neuron0x18279b80()*-0.959895);
}

double ldNNEB::synapse0x18279fd0() {
   return (neuron0x1827a3e0()*2.46191);
}

double ldNNEB::synapse0x1827a350() {
   return (neuron0x1827a890()*-0.0487259);
}

double ldNNEB::synapse0x1827a390() {
   return (neuron0x1827ae50()*0.655329);
}

double ldNNEB::synapse0x18277290() {
   return (neuron0x1827b410()*-4.96624);
}

double ldNNEB::synapse0x182772d0() {
   return (neuron0x1827b9d0()*4.81781);
}

double ldNNEB::synapse0x18277310() {
   return (neuron0x1827bf90()*-4.50754);
}



double llNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.15492)/1.68616;
   input2 = (in2 - 1.89563)/1.49913;
   input3 = (in3 - 1.90666)/1.49442;
   input4 = (in4 - 0.3065)/1.46726;
   input5 = (in5 - 0.318454)/1.50742;
   input6 = (in6 - 0.305354)/1.51455;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x2f8ff90();
     default:
         return 0.;
   }
}

double llNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.15492)/1.68616;
   input2 = (input[2] - 1.89563)/1.49913;
   input3 = (input[3] - 1.90666)/1.49442;
   input4 = (input[4] - 0.3065)/1.46726;
   input5 = (input[5] - 0.318454)/1.50742;
   input6 = (input[6] - 0.305354)/1.51455;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x2f8ff90();
     default:
         return 0.;
   }
}

double llNNEB::neuron0x2f8b810() {
   return input0;
}

double llNNEB::neuron0x2f8bb50() {
   return input1;
}

double llNNEB::neuron0x2f8be90() {
   return input2;
}

double llNNEB::neuron0x2f8c1d0() {
   return input3;
}

double llNNEB::neuron0x2f8c510() {
   return input4;
}

double llNNEB::neuron0x2f8c850() {
   return input5;
}

double llNNEB::neuron0x2f8cb90() {
   return input6;
}

double llNNEB::neuron0x2f8ced0() {
   return input7;
}

double llNNEB::input0x2f8d340() {
   double input = -0.0926358;
   input += synapse0x2eec210();
   input += synapse0x2f94410();
   input += synapse0x2f8d5f0();
   input += synapse0x2f8d630();
   input += synapse0x2f8d670();
   input += synapse0x2f8d6b0();
   input += synapse0x2f8d6f0();
   input += synapse0x2f8d730();
   return input;
}

double llNNEB::neuron0x2f8d340() {
   double input = input0x2f8d340();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8d770() {
   double input = -0.565862;
   input += synapse0x2f8dab0();
   input += synapse0x2f8daf0();
   input += synapse0x2f8db30();
   input += synapse0x2f8db70();
   input += synapse0x2f8dbb0();
   input += synapse0x2f8dbf0();
   input += synapse0x2f8dc30();
   input += synapse0x2f8dc70();
   return input;
}

double llNNEB::neuron0x2f8d770() {
   double input = input0x2f8d770();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8dcb0() {
   double input = 0.115237;
   input += synapse0x2f8dff0();
   input += synapse0x27eaed0();
   input += synapse0x27eaf10();
   input += synapse0x2f8e140();
   input += synapse0x2f8e180();
   input += synapse0x2f8e1c0();
   input += synapse0x2f8e200();
   input += synapse0x2f8e240();
   return input;
}

double llNNEB::neuron0x2f8dcb0() {
   double input = input0x2f8dcb0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8e280() {
   double input = 0.890509;
   input += synapse0x2f8e5c0();
   input += synapse0x2f8e600();
   input += synapse0x2f8e640();
   input += synapse0x2f8e680();
   input += synapse0x2f8e6c0();
   input += synapse0x2f8e700();
   input += synapse0x2f8e740();
   input += synapse0x2f8e780();
   return input;
}

double llNNEB::neuron0x2f8e280() {
   double input = input0x2f8e280();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8e7c0() {
   double input = 0.775207;
   input += synapse0x2f8eb00();
   input += synapse0x2f8b740();
   input += synapse0x2f94450();
   input += synapse0x27e9a10();
   input += synapse0x2f8e030();
   input += synapse0x2f8e070();
   input += synapse0x2f8e0b0();
   input += synapse0x2f8e0f0();
   return input;
}

double llNNEB::neuron0x2f8e7c0() {
   double input = input0x2f8e7c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8eb40() {
   double input = 0.230515;
   input += synapse0x2f8ee80();
   input += synapse0x2f8eec0();
   input += synapse0x2f8ef00();
   input += synapse0x2f8ef40();
   input += synapse0x2f8ef80();
   input += synapse0x2f8efc0();
   input += synapse0x2f8f000();
   input += synapse0x2f8f040();
   return input;
}

double llNNEB::neuron0x2f8eb40() {
   double input = input0x2f8eb40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8f080() {
   double input = 0.862616;
   input += synapse0x2f8f3c0();
   input += synapse0x2f8f400();
   input += synapse0x2f8f440();
   input += synapse0x2f8f480();
   input += synapse0x2f8f4c0();
   input += synapse0x2f8f500();
   input += synapse0x2f8f540();
   input += synapse0x2f8f580();
   return input;
}

double llNNEB::neuron0x2f8f080() {
   double input = input0x2f8f080();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8f5c0() {
   double input = -0.332968;
   input += synapse0x2f8f900();
   input += synapse0x2f8f940();
   input += synapse0x2f8f980();
   input += synapse0x2f8f9c0();
   input += synapse0x2f8fa00();
   input += synapse0x2f8fa40();
   input += synapse0x2f8fa80();
   input += synapse0x2f8fac0();
   return input;
}

double llNNEB::neuron0x2f8f5c0() {
   double input = input0x2f8f5c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8fb00() {
   double input = 0.082267;
   input += synapse0x26d8e30();
   input += synapse0x26d8e70();
   input += synapse0x28047e0();
   input += synapse0x2804820();
   input += synapse0x2804860();
   input += synapse0x28048a0();
   input += synapse0x28048e0();
   input += synapse0x2804920();
   return input;
}

double llNNEB::neuron0x2f8fb00() {
   double input = input0x2f8fb00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f90360() {
   double input = -0.304636;
   input += synapse0x2f90610();
   input += synapse0x2f90650();
   input += synapse0x2f90690();
   input += synapse0x2f906d0();
   input += synapse0x2f90710();
   input += synapse0x2f90750();
   input += synapse0x2f90790();
   input += synapse0x2f907d0();
   return input;
}

double llNNEB::neuron0x2f90360() {
   double input = input0x2f90360();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f90810() {
   double input = -0.0916165;
   input += synapse0x2f90b50();
   input += synapse0x2f90b90();
   input += synapse0x2f90bd0();
   input += synapse0x2f90c10();
   input += synapse0x2f90c50();
   input += synapse0x2f90c90();
   input += synapse0x2f90cd0();
   input += synapse0x2f90d10();
   input += synapse0x2f90d50();
   input += synapse0x2f90d90();
   return input;
}

double llNNEB::neuron0x2f90810() {
   double input = input0x2f90810();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f90dd0() {
   double input = -0.443607;
   input += synapse0x2f91110();
   input += synapse0x2f91150();
   input += synapse0x2f91190();
   input += synapse0x2f911d0();
   input += synapse0x2f91210();
   input += synapse0x2f91250();
   input += synapse0x2f91290();
   input += synapse0x2f912d0();
   input += synapse0x2f91310();
   input += synapse0x2f91350();
   return input;
}

double llNNEB::neuron0x2f90dd0() {
   double input = input0x2f90dd0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f91390() {
   double input = -0.00521054;
   input += synapse0x2f916d0();
   input += synapse0x2f91710();
   input += synapse0x2f91750();
   input += synapse0x2f91790();
   input += synapse0x2f917d0();
   input += synapse0x2f91810();
   input += synapse0x2f91850();
   input += synapse0x2f91890();
   input += synapse0x2f918d0();
   input += synapse0x2f91910();
   return input;
}

double llNNEB::neuron0x2f91390() {
   double input = input0x2f91390();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f91950() {
   double input = -0.446475;
   input += synapse0x2f91c90();
   input += synapse0x2f91cd0();
   input += synapse0x2f91d10();
   input += synapse0x2f91d50();
   input += synapse0x2f91d90();
   input += synapse0x2f91dd0();
   input += synapse0x2f91e10();
   input += synapse0x2f91e50();
   input += synapse0x2f91e90();
   input += synapse0x2f91ed0();
   return input;
}

double llNNEB::neuron0x2f91950() {
   double input = input0x2f91950();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f91f10() {
   double input = 2.12019;
   input += synapse0x2f92250();
   input += synapse0x2f92290();
   input += synapse0x2f922d0();
   input += synapse0x2f92310();
   input += synapse0x2f92350();
   input += synapse0x2f92390();
   input += synapse0x2f923d0();
   input += synapse0x2f92410();
   input += synapse0x2f92450();
   input += synapse0x2f8ff50();
   return input;
}

double llNNEB::neuron0x2f91f10() {
   double input = input0x2f91f10();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEB::input0x2f8ff90() {
   double input = 2.9809;
   input += synapse0x2f902d0();
   input += synapse0x2f90310();
   input += synapse0x2f8d210();
   input += synapse0x2f8d250();
   input += synapse0x2f8d290();
   return input;
}

double llNNEB::neuron0x2f8ff90() {
   double input = input0x2f8ff90();
   return (input * 1)+0;
}

double llNNEB::synapse0x2eec210() {
   return (neuron0x2f8b810()*1.18802);
}

double llNNEB::synapse0x2f94410() {
   return (neuron0x2f8bb50()*0.697943);
}

double llNNEB::synapse0x2f8d5f0() {
   return (neuron0x2f8be90()*1.29077);
}

double llNNEB::synapse0x2f8d630() {
   return (neuron0x2f8c1d0()*0.974011);
}

double llNNEB::synapse0x2f8d670() {
   return (neuron0x2f8c510()*0.531372);
}

double llNNEB::synapse0x2f8d6b0() {
   return (neuron0x2f8c850()*0.81473);
}

double llNNEB::synapse0x2f8d6f0() {
   return (neuron0x2f8cb90()*2.01243);
}

double llNNEB::synapse0x2f8d730() {
   return (neuron0x2f8ced0()*2.38963);
}

double llNNEB::synapse0x2f8dab0() {
   return (neuron0x2f8b810()*-0.925515);
}

double llNNEB::synapse0x2f8daf0() {
   return (neuron0x2f8bb50()*0.478754);
}

double llNNEB::synapse0x2f8db30() {
   return (neuron0x2f8be90()*-0.187566);
}

double llNNEB::synapse0x2f8db70() {
   return (neuron0x2f8c1d0()*0.794948);
}

double llNNEB::synapse0x2f8dbb0() {
   return (neuron0x2f8c510()*0.174207);
}

double llNNEB::synapse0x2f8dbf0() {
   return (neuron0x2f8c850()*0.30047);
}

double llNNEB::synapse0x2f8dc30() {
   return (neuron0x2f8cb90()*1.05822);
}

double llNNEB::synapse0x2f8dc70() {
   return (neuron0x2f8ced0()*0.192867);
}

double llNNEB::synapse0x2f8dff0() {
   return (neuron0x2f8b810()*2.4499);
}

double llNNEB::synapse0x27eaed0() {
   return (neuron0x2f8bb50()*0.382343);
}

double llNNEB::synapse0x27eaf10() {
   return (neuron0x2f8be90()*1.60773);
}

double llNNEB::synapse0x2f8e140() {
   return (neuron0x2f8c1d0()*1.90051);
}

double llNNEB::synapse0x2f8e180() {
   return (neuron0x2f8c510()*1.52519);
}

double llNNEB::synapse0x2f8e1c0() {
   return (neuron0x2f8c850()*1.13771);
}

double llNNEB::synapse0x2f8e200() {
   return (neuron0x2f8cb90()*3.64255);
}

double llNNEB::synapse0x2f8e240() {
   return (neuron0x2f8ced0()*3.49626);
}

double llNNEB::synapse0x2f8e5c0() {
   return (neuron0x2f8b810()*1.23427);
}

double llNNEB::synapse0x2f8e600() {
   return (neuron0x2f8bb50()*-0.0519786);
}

double llNNEB::synapse0x2f8e640() {
   return (neuron0x2f8be90()*0.211897);
}

double llNNEB::synapse0x2f8e680() {
   return (neuron0x2f8c1d0()*0.200567);
}

double llNNEB::synapse0x2f8e6c0() {
   return (neuron0x2f8c510()*-0.23557);
}

double llNNEB::synapse0x2f8e700() {
   return (neuron0x2f8c850()*-0.27122);
}

double llNNEB::synapse0x2f8e740() {
   return (neuron0x2f8cb90()*-0.403843);
}

double llNNEB::synapse0x2f8e780() {
   return (neuron0x2f8ced0()*-0.44644);
}

double llNNEB::synapse0x2f8eb00() {
   return (neuron0x2f8b810()*1.95955);
}

double llNNEB::synapse0x2f8b740() {
   return (neuron0x2f8bb50()*0.227203);
}

double llNNEB::synapse0x2f94450() {
   return (neuron0x2f8be90()*1.08097);
}

double llNNEB::synapse0x27e9a10() {
   return (neuron0x2f8c1d0()*1.31399);
}

double llNNEB::synapse0x2f8e030() {
   return (neuron0x2f8c510()*0.807464);
}

double llNNEB::synapse0x2f8e070() {
   return (neuron0x2f8c850()*0.5526);
}

double llNNEB::synapse0x2f8e0b0() {
   return (neuron0x2f8cb90()*1.54524);
}

double llNNEB::synapse0x2f8e0f0() {
   return (neuron0x2f8ced0()*1.25477);
}

double llNNEB::synapse0x2f8ee80() {
   return (neuron0x2f8b810()*1.00016);
}

double llNNEB::synapse0x2f8eec0() {
   return (neuron0x2f8bb50()*-0.29655);
}

double llNNEB::synapse0x2f8ef00() {
   return (neuron0x2f8be90()*-1.64098);
}

double llNNEB::synapse0x2f8ef40() {
   return (neuron0x2f8c1d0()*0.598756);
}

double llNNEB::synapse0x2f8ef80() {
   return (neuron0x2f8c510()*-0.255078);
}

double llNNEB::synapse0x2f8efc0() {
   return (neuron0x2f8c850()*-0.350156);
}

double llNNEB::synapse0x2f8f000() {
   return (neuron0x2f8cb90()*0.872753);
}

double llNNEB::synapse0x2f8f040() {
   return (neuron0x2f8ced0()*0.2515);
}

double llNNEB::synapse0x2f8f3c0() {
   return (neuron0x2f8b810()*2.0049);
}

double llNNEB::synapse0x2f8f400() {
   return (neuron0x2f8bb50()*-0.538494);
}

double llNNEB::synapse0x2f8f440() {
   return (neuron0x2f8be90()*0.457742);
}

double llNNEB::synapse0x2f8f480() {
   return (neuron0x2f8c1d0()*0.681461);
}

double llNNEB::synapse0x2f8f4c0() {
   return (neuron0x2f8c510()*-0.64671);
}

double llNNEB::synapse0x2f8f500() {
   return (neuron0x2f8c850()*-0.655839);
}

double llNNEB::synapse0x2f8f540() {
   return (neuron0x2f8cb90()*-0.9445);
}

double llNNEB::synapse0x2f8f580() {
   return (neuron0x2f8ced0()*-1.16369);
}

double llNNEB::synapse0x2f8f900() {
   return (neuron0x2f8b810()*-0.307722);
}

double llNNEB::synapse0x2f8f940() {
   return (neuron0x2f8bb50()*0.00796732);
}

double llNNEB::synapse0x2f8f980() {
   return (neuron0x2f8be90()*-0.182972);
}

double llNNEB::synapse0x2f8f9c0() {
   return (neuron0x2f8c1d0()*-0.619658);
}

double llNNEB::synapse0x2f8fa00() {
   return (neuron0x2f8c510()*-0.597408);
}

double llNNEB::synapse0x2f8fa40() {
   return (neuron0x2f8c850()*-0.0893563);
}

double llNNEB::synapse0x2f8fa80() {
   return (neuron0x2f8cb90()*-1.37406);
}

double llNNEB::synapse0x2f8fac0() {
   return (neuron0x2f8ced0()*-1.96252);
}

double llNNEB::synapse0x26d8e30() {
   return (neuron0x2f8b810()*-2.49165);
}

double llNNEB::synapse0x26d8e70() {
   return (neuron0x2f8bb50()*-1.00368);
}

double llNNEB::synapse0x28047e0() {
   return (neuron0x2f8be90()*-1.71502);
}

double llNNEB::synapse0x2804820() {
   return (neuron0x2f8c1d0()*-1.38569);
}

double llNNEB::synapse0x2804860() {
   return (neuron0x2f8c510()*-0.972498);
}

double llNNEB::synapse0x28048a0() {
   return (neuron0x2f8c850()*-0.694178);
}

double llNNEB::synapse0x28048e0() {
   return (neuron0x2f8cb90()*-3.13402);
}

double llNNEB::synapse0x2804920() {
   return (neuron0x2f8ced0()*-4.05429);
}

double llNNEB::synapse0x2f90610() {
   return (neuron0x2f8b810()*0.952673);
}

double llNNEB::synapse0x2f90650() {
   return (neuron0x2f8bb50()*0.573276);
}

double llNNEB::synapse0x2f90690() {
   return (neuron0x2f8be90()*0.0479708);
}

double llNNEB::synapse0x2f906d0() {
   return (neuron0x2f8c1d0()*0.290278);
}

double llNNEB::synapse0x2f90710() {
   return (neuron0x2f8c510()*-0.130862);
}

double llNNEB::synapse0x2f90750() {
   return (neuron0x2f8c850()*-0.196606);
}

double llNNEB::synapse0x2f90790() {
   return (neuron0x2f8cb90()*-0.94221);
}

double llNNEB::synapse0x2f907d0() {
   return (neuron0x2f8ced0()*-1.11217);
}

double llNNEB::synapse0x2f90b50() {
   return (neuron0x2f8d340()*2.23731);
}

double llNNEB::synapse0x2f90b90() {
   return (neuron0x2f8d770()*-4.23388);
}

double llNNEB::synapse0x2f90bd0() {
   return (neuron0x2f8dcb0()*0.512447);
}

double llNNEB::synapse0x2f90c10() {
   return (neuron0x2f8e280()*1.72921);
}

double llNNEB::synapse0x2f90c50() {
   return (neuron0x2f8e7c0()*2.99858);
}

double llNNEB::synapse0x2f90c90() {
   return (neuron0x2f8eb40()*3.37404);
}

double llNNEB::synapse0x2f90cd0() {
   return (neuron0x2f8f080()*-6.86377);
}

double llNNEB::synapse0x2f90d10() {
   return (neuron0x2f8f5c0()*0.238987);
}

double llNNEB::synapse0x2f90d50() {
   return (neuron0x2f8fb00()*3.94311);
}

double llNNEB::synapse0x2f90d90() {
   return (neuron0x2f90360()*-3.95507);
}

double llNNEB::synapse0x2f91110() {
   return (neuron0x2f8d340()*0.029347);
}

double llNNEB::synapse0x2f91150() {
   return (neuron0x2f8d770()*1.85203);
}

double llNNEB::synapse0x2f91190() {
   return (neuron0x2f8dcb0()*-0.249941);
}

double llNNEB::synapse0x2f911d0() {
   return (neuron0x2f8e280()*1.14122);
}

double llNNEB::synapse0x2f91210() {
   return (neuron0x2f8e7c0()*-0.0270421);
}

double llNNEB::synapse0x2f91250() {
   return (neuron0x2f8eb40()*-0.581733);
}

double llNNEB::synapse0x2f91290() {
   return (neuron0x2f8f080()*-0.693613);
}

double llNNEB::synapse0x2f912d0() {
   return (neuron0x2f8f5c0()*-0.351173);
}

double llNNEB::synapse0x2f91310() {
   return (neuron0x2f8fb00()*-0.4098);
}

double llNNEB::synapse0x2f91350() {
   return (neuron0x2f90360()*0.0447752);
}

double llNNEB::synapse0x2f916d0() {
   return (neuron0x2f8d340()*1.47129);
}

double llNNEB::synapse0x2f91710() {
   return (neuron0x2f8d770()*-3.90508);
}

double llNNEB::synapse0x2f91750() {
   return (neuron0x2f8dcb0()*0.767496);
}

double llNNEB::synapse0x2f91790() {
   return (neuron0x2f8e280()*1.01144);
}

double llNNEB::synapse0x2f917d0() {
   return (neuron0x2f8e7c0()*1.74916);
}

double llNNEB::synapse0x2f91810() {
   return (neuron0x2f8eb40()*2.63845);
}

double llNNEB::synapse0x2f91850() {
   return (neuron0x2f8f080()*-6.46134);
}

double llNNEB::synapse0x2f91890() {
   return (neuron0x2f8f5c0()*0.771541);
}

double llNNEB::synapse0x2f918d0() {
   return (neuron0x2f8fb00()*3.16939);
}

double llNNEB::synapse0x2f91910() {
   return (neuron0x2f90360()*-5.83595);
}

double llNNEB::synapse0x2f91c90() {
   return (neuron0x2f8d340()*1.43372);
}

double llNNEB::synapse0x2f91cd0() {
   return (neuron0x2f8d770()*-1.89272);
}

double llNNEB::synapse0x2f91d10() {
   return (neuron0x2f8dcb0()*0.428573);
}

double llNNEB::synapse0x2f91d50() {
   return (neuron0x2f8e280()*1.57341);
}

double llNNEB::synapse0x2f91d90() {
   return (neuron0x2f8e7c0()*1.82221);
}

double llNNEB::synapse0x2f91dd0() {
   return (neuron0x2f8eb40()*1.58248);
}

double llNNEB::synapse0x2f91e10() {
   return (neuron0x2f8f080()*-3.14869);
}

double llNNEB::synapse0x2f91e50() {
   return (neuron0x2f8f5c0()*-0.0304965);
}

double llNNEB::synapse0x2f91e90() {
   return (neuron0x2f8fb00()*1.75882);
}

double llNNEB::synapse0x2f91ed0() {
   return (neuron0x2f90360()*-2.44213);
}

double llNNEB::synapse0x2f92250() {
   return (neuron0x2f8d340()*0.707172);
}

double llNNEB::synapse0x2f92290() {
   return (neuron0x2f8d770()*0.393115);
}

double llNNEB::synapse0x2f922d0() {
   return (neuron0x2f8dcb0()*-0.0246247);
}

double llNNEB::synapse0x2f92310() {
   return (neuron0x2f8e280()*-5.06829);
}

double llNNEB::synapse0x2f92350() {
   return (neuron0x2f8e7c0()*-0.0752881);
}

double llNNEB::synapse0x2f92390() {
   return (neuron0x2f8eb40()*-0.515807);
}

double llNNEB::synapse0x2f923d0() {
   return (neuron0x2f8f080()*1.03079);
}

double llNNEB::synapse0x2f92410() {
   return (neuron0x2f8f5c0()*-0.35588);
}

double llNNEB::synapse0x2f92450() {
   return (neuron0x2f8fb00()*0.37701);
}

double llNNEB::synapse0x2f8ff50() {
   return (neuron0x2f90360()*1.99131);
}

double llNNEB::synapse0x2f902d0() {
   return (neuron0x2f90810()*-0.700239);
}

double llNNEB::synapse0x2f90310() {
   return (neuron0x2f90dd0()*4.15526);
}

double llNNEB::synapse0x2f8d210() {
   return (neuron0x2f91390()*2.92611);
}

double llNNEB::synapse0x2f8d250() {
   return (neuron0x2f91950()*0.738438);
}

double llNNEB::synapse0x2f8d290() {
   return (neuron0x2f91f10()*-7.41024);
}

double luNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.15492)/1.68616;
   input2 = (in2 - 1.14956)/1.69316;
   input3 = (in3 - 1.89563)/1.49913;
   input4 = (in4 - 1.90666)/1.49442;
   input5 = (in5 - 0.3065)/1.46726;
   input6 = (in6 - 0.318454)/1.50742;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x10fdc050();
     default:
         return 0.;
   }
}

double luNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.15492)/1.68616;
   input2 = (input[2] - 1.14956)/1.69316;
   input3 = (input[3] - 1.89563)/1.49913;
   input4 = (input[4] - 1.90666)/1.49442;
   input5 = (input[5] - 0.3065)/1.46726;
   input6 = (input[6] - 0.318454)/1.50742;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x10fdc050();
     default:
         return 0.;
   }
}

double luNNEB::neuron0x10fd78d0() {
   return input0;
}

double luNNEB::neuron0x10fd7c10() {
   return input1;
}

double luNNEB::neuron0x10fd7f50() {
   return input2;
}

double luNNEB::neuron0x10fd8290() {
   return input3;
}

double luNNEB::neuron0x10fd85d0() {
   return input4;
}

double luNNEB::neuron0x10fd8910() {
   return input5;
}

double luNNEB::neuron0x10fd8c50() {
   return input6;
}

double luNNEB::neuron0x10fd8f90() {
   return input7;
}

double luNNEB::input0x10fd9400() {
   double input = -0.482263;
   input += synapse0x10f382d0();
   input += synapse0x10fe04d0();
   input += synapse0x10fd96b0();
   input += synapse0x10fd96f0();
   input += synapse0x10fd9730();
   input += synapse0x10fd9770();
   input += synapse0x10fd97b0();
   input += synapse0x10fd97f0();
   return input;
}

double luNNEB::neuron0x10fd9400() {
   double input = input0x10fd9400();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fd9830() {
   double input = 0.47095;
   input += synapse0x10fd9b70();
   input += synapse0x10fd9bb0();
   input += synapse0x10fd9bf0();
   input += synapse0x10fd9c30();
   input += synapse0x10fd9c70();
   input += synapse0x10fd9cb0();
   input += synapse0x10fd9cf0();
   input += synapse0x10fd9d30();
   return input;
}

double luNNEB::neuron0x10fd9830() {
   double input = input0x10fd9830();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fd9d70() {
   double input = -0.263786;
   input += synapse0x10fda0b0();
   input += synapse0x10836f90();
   input += synapse0x10836fd0();
   input += synapse0x10fda200();
   input += synapse0x10fda240();
   input += synapse0x10fda280();
   input += synapse0x10fda2c0();
   input += synapse0x10fda300();
   return input;
}

double luNNEB::neuron0x10fd9d70() {
   double input = input0x10fd9d70();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fda340() {
   double input = -2.45341;
   input += synapse0x10fda680();
   input += synapse0x10fda6c0();
   input += synapse0x10fda700();
   input += synapse0x10fda740();
   input += synapse0x10fda780();
   input += synapse0x10fda7c0();
   input += synapse0x10fda800();
   input += synapse0x10fda840();
   return input;
}

double luNNEB::neuron0x10fda340() {
   double input = input0x10fda340();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fda880() {
   double input = -0.186997;
   input += synapse0x10fdabc0();
   input += synapse0x10fd7800();
   input += synapse0x10fe0510();
   input += synapse0x10835ad0();
   input += synapse0x10fda0f0();
   input += synapse0x10fda130();
   input += synapse0x10fda170();
   input += synapse0x10fda1b0();
   return input;
}

double luNNEB::neuron0x10fda880() {
   double input = input0x10fda880();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdac00() {
   double input = 1.98839;
   input += synapse0x10fdaf40();
   input += synapse0x10fdaf80();
   input += synapse0x10fdafc0();
   input += synapse0x10fdb000();
   input += synapse0x10fdb040();
   input += synapse0x10fdb080();
   input += synapse0x10fdb0c0();
   input += synapse0x10fdb100();
   return input;
}

double luNNEB::neuron0x10fdac00() {
   double input = input0x10fdac00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdb140() {
   double input = -1.00701;
   input += synapse0x10fdb480();
   input += synapse0x10fdb4c0();
   input += synapse0x10fdb500();
   input += synapse0x10fdb540();
   input += synapse0x10fdb580();
   input += synapse0x10fdb5c0();
   input += synapse0x10fdb600();
   input += synapse0x10fdb640();
   return input;
}

double luNNEB::neuron0x10fdb140() {
   double input = input0x10fdb140();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdb680() {
   double input = -0.560147;
   input += synapse0x10fdb9c0();
   input += synapse0x10fdba00();
   input += synapse0x10fdba40();
   input += synapse0x10fdba80();
   input += synapse0x10fdbac0();
   input += synapse0x10fdbb00();
   input += synapse0x10fdbb40();
   input += synapse0x10fdbb80();
   return input;
}

double luNNEB::neuron0x10fdb680() {
   double input = input0x10fdb680();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdbbc0() {
   double input = -0.721986;
   input += synapse0x10724ef0();
   input += synapse0x10724f30();
   input += synapse0x108508a0();
   input += synapse0x108508e0();
   input += synapse0x10850920();
   input += synapse0x10850960();
   input += synapse0x108509a0();
   input += synapse0x108509e0();
   return input;
}

double luNNEB::neuron0x10fdbbc0() {
   double input = input0x10fdbbc0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdc420() {
   double input = -0.736728;
   input += synapse0x10fdc6d0();
   input += synapse0x10fdc710();
   input += synapse0x10fdc750();
   input += synapse0x10fdc790();
   input += synapse0x10fdc7d0();
   input += synapse0x10fdc810();
   input += synapse0x10fdc850();
   input += synapse0x10fdc890();
   return input;
}

double luNNEB::neuron0x10fdc420() {
   double input = input0x10fdc420();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdc8d0() {
   double input = 0.481603;
   input += synapse0x10fdcc10();
   input += synapse0x10fdcc50();
   input += synapse0x10fdcc90();
   input += synapse0x10fdccd0();
   input += synapse0x10fdcd10();
   input += synapse0x10fdcd50();
   input += synapse0x10fdcd90();
   input += synapse0x10fdcdd0();
   input += synapse0x10fdce10();
   input += synapse0x10fdce50();
   return input;
}

double luNNEB::neuron0x10fdc8d0() {
   double input = input0x10fdc8d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdce90() {
   double input = 0.390999;
   input += synapse0x10fdd1d0();
   input += synapse0x10fdd210();
   input += synapse0x10fdd250();
   input += synapse0x10fdd290();
   input += synapse0x10fdd2d0();
   input += synapse0x10fdd310();
   input += synapse0x10fdd350();
   input += synapse0x10fdd390();
   input += synapse0x10fdd3d0();
   input += synapse0x10fdd410();
   return input;
}

double luNNEB::neuron0x10fdce90() {
   double input = input0x10fdce90();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdd450() {
   double input = -0.566348;
   input += synapse0x10fdd790();
   input += synapse0x10fdd7d0();
   input += synapse0x10fdd810();
   input += synapse0x10fdd850();
   input += synapse0x10fdd890();
   input += synapse0x10fdd8d0();
   input += synapse0x10fdd910();
   input += synapse0x10fdd950();
   input += synapse0x10fdd990();
   input += synapse0x10fdd9d0();
   return input;
}

double luNNEB::neuron0x10fdd450() {
   double input = input0x10fdd450();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdda10() {
   double input = 0.577951;
   input += synapse0x10fddd50();
   input += synapse0x10fddd90();
   input += synapse0x10fdddd0();
   input += synapse0x10fdde10();
   input += synapse0x10fdde50();
   input += synapse0x10fdde90();
   input += synapse0x10fdded0();
   input += synapse0x10fddf10();
   input += synapse0x10fddf50();
   input += synapse0x10fddf90();
   return input;
}

double luNNEB::neuron0x10fdda10() {
   double input = input0x10fdda10();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fddfd0() {
   double input = -0.372479;
   input += synapse0x10fde310();
   input += synapse0x10fde350();
   input += synapse0x10fde390();
   input += synapse0x10fde3d0();
   input += synapse0x10fde410();
   input += synapse0x10fde450();
   input += synapse0x10fde490();
   input += synapse0x10fde4d0();
   input += synapse0x10fde510();
   input += synapse0x10fdc010();
   return input;
}

double luNNEB::neuron0x10fddfd0() {
   double input = input0x10fddfd0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEB::input0x10fdc050() {
   double input = 4.72312;
   input += synapse0x10fdc390();
   input += synapse0x10fdc3d0();
   input += synapse0x10fd92d0();
   input += synapse0x10fd9310();
   input += synapse0x10fd9350();
   return input;
}

double luNNEB::neuron0x10fdc050() {
   double input = input0x10fdc050();
   return (input * 1)+0;
}

double luNNEB::synapse0x10f382d0() {
   return (neuron0x10fd78d0()*0.162147);
}

double luNNEB::synapse0x10fe04d0() {
   return (neuron0x10fd7c10()*0.461803);
}

double luNNEB::synapse0x10fd96b0() {
   return (neuron0x10fd7f50()*-1.37936);
}

double luNNEB::synapse0x10fd96f0() {
   return (neuron0x10fd8290()*-0.202747);
}

double luNNEB::synapse0x10fd9730() {
   return (neuron0x10fd85d0()*-0.0631962);
}

double luNNEB::synapse0x10fd9770() {
   return (neuron0x10fd8910()*-0.0559553);
}

double luNNEB::synapse0x10fd97b0() {
   return (neuron0x10fd8c50()*-0.328558);
}

double luNNEB::synapse0x10fd97f0() {
   return (neuron0x10fd8f90()*0.20233);
}

double luNNEB::synapse0x10fd9b70() {
   return (neuron0x10fd78d0()*2.65087);
}

double luNNEB::synapse0x10fd9bb0() {
   return (neuron0x10fd7c10()*-1.28916);
}

double luNNEB::synapse0x10fd9bf0() {
   return (neuron0x10fd7f50()*0.0722316);
}

double luNNEB::synapse0x10fd9c30() {
   return (neuron0x10fd8290()*0.40294);
}

double luNNEB::synapse0x10fd9c70() {
   return (neuron0x10fd85d0()*0.403065);
}

double luNNEB::synapse0x10fd9cb0() {
   return (neuron0x10fd8910()*0.325961);
}

double luNNEB::synapse0x10fd9cf0() {
   return (neuron0x10fd8c50()*-1.24393);
}

double luNNEB::synapse0x10fd9d30() {
   return (neuron0x10fd8f90()*0.0787346);
}

double luNNEB::synapse0x10fda0b0() {
   return (neuron0x10fd78d0()*1.33716);
}

double luNNEB::synapse0x10836f90() {
   return (neuron0x10fd7c10()*1.12612);
}

double luNNEB::synapse0x10836fd0() {
   return (neuron0x10fd7f50()*2.05235);
}

double luNNEB::synapse0x10fda200() {
   return (neuron0x10fd8290()*1.23352);
}

double luNNEB::synapse0x10fda240() {
   return (neuron0x10fd85d0()*0.750904);
}

double luNNEB::synapse0x10fda280() {
   return (neuron0x10fd8910()*1.041);
}

double luNNEB::synapse0x10fda2c0() {
   return (neuron0x10fd8c50()*0.647542);
}

double luNNEB::synapse0x10fda300() {
   return (neuron0x10fd8f90()*1.34407);
}

double luNNEB::synapse0x10fda680() {
   return (neuron0x10fd78d0()*-0.265954);
}

double luNNEB::synapse0x10fda6c0() {
   return (neuron0x10fd7c10()*0.289035);
}

double luNNEB::synapse0x10fda700() {
   return (neuron0x10fd7f50()*1.06486);
}

double luNNEB::synapse0x10fda740() {
   return (neuron0x10fd8290()*-1.19126);
}

double luNNEB::synapse0x10fda780() {
   return (neuron0x10fd85d0()*-0.03881);
}

double luNNEB::synapse0x10fda7c0() {
   return (neuron0x10fd8910()*0.233711);
}

double luNNEB::synapse0x10fda800() {
   return (neuron0x10fd8c50()*-0.349635);
}

double luNNEB::synapse0x10fda840() {
   return (neuron0x10fd8f90()*-0.0842293);
}

double luNNEB::synapse0x10fdabc0() {
   return (neuron0x10fd78d0()*-1.35044);
}

double luNNEB::synapse0x10fd7800() {
   return (neuron0x10fd7c10()*-0.879086);
}

double luNNEB::synapse0x10fe0510() {
   return (neuron0x10fd7f50()*-1.33893);
}

double luNNEB::synapse0x10835ad0() {
   return (neuron0x10fd8290()*-1.55637);
}

double luNNEB::synapse0x10fda0f0() {
   return (neuron0x10fd85d0()*0.0523866);
}

double luNNEB::synapse0x10fda130() {
   return (neuron0x10fd8910()*-0.793344);
}

double luNNEB::synapse0x10fda170() {
   return (neuron0x10fd8c50()*-0.73362);
}

double luNNEB::synapse0x10fda1b0() {
   return (neuron0x10fd8f90()*-0.531828);
}

double luNNEB::synapse0x10fdaf40() {
   return (neuron0x10fd78d0()*1.17694);
}

double luNNEB::synapse0x10fdaf80() {
   return (neuron0x10fd7c10()*0.323671);
}

double luNNEB::synapse0x10fdafc0() {
   return (neuron0x10fd7f50()*-0.781744);
}

double luNNEB::synapse0x10fdb000() {
   return (neuron0x10fd8290()*-0.713545);
}

double luNNEB::synapse0x10fdb040() {
   return (neuron0x10fd85d0()*0.464629);
}

double luNNEB::synapse0x10fdb080() {
   return (neuron0x10fd8910()*0.0100201);
}

double luNNEB::synapse0x10fdb0c0() {
   return (neuron0x10fd8c50()*0.0764805);
}

double luNNEB::synapse0x10fdb100() {
   return (neuron0x10fd8f90()*-0.451934);
}

double luNNEB::synapse0x10fdb480() {
   return (neuron0x10fd78d0()*0.893002);
}

double luNNEB::synapse0x10fdb4c0() {
   return (neuron0x10fd7c10()*0.512438);
}

double luNNEB::synapse0x10fdb500() {
   return (neuron0x10fd7f50()*0.199848);
}

double luNNEB::synapse0x10fdb540() {
   return (neuron0x10fd8290()*-0.0643325);
}

double luNNEB::synapse0x10fdb580() {
   return (neuron0x10fd85d0()*0.0574431);
}

double luNNEB::synapse0x10fdb5c0() {
   return (neuron0x10fd8910()*0.00360423);
}

double luNNEB::synapse0x10fdb600() {
   return (neuron0x10fd8c50()*-0.758948);
}

double luNNEB::synapse0x10fdb640() {
   return (neuron0x10fd8f90()*-0.0799728);
}

double luNNEB::synapse0x10fdb9c0() {
   return (neuron0x10fd78d0()*-3.16954);
}

double luNNEB::synapse0x10fdba00() {
   return (neuron0x10fd7c10()*-1.53585);
}

double luNNEB::synapse0x10fdba40() {
   return (neuron0x10fd7f50()*-4.1974);
}

double luNNEB::synapse0x10fdba80() {
   return (neuron0x10fd8290()*-3.48157);
}

double luNNEB::synapse0x10fdbac0() {
   return (neuron0x10fd85d0()*-1.96283);
}

double luNNEB::synapse0x10fdbb00() {
   return (neuron0x10fd8910()*-2.29231);
}

double luNNEB::synapse0x10fdbb40() {
   return (neuron0x10fd8c50()*-1.23163);
}

double luNNEB::synapse0x10fdbb80() {
   return (neuron0x10fd8f90()*-2.98389);
}

double luNNEB::synapse0x10724ef0() {
   return (neuron0x10fd78d0()*-1.35246);
}

double luNNEB::synapse0x10724f30() {
   return (neuron0x10fd7c10()*0.840469);
}

double luNNEB::synapse0x108508a0() {
   return (neuron0x10fd7f50()*-1.29467);
}

double luNNEB::synapse0x108508e0() {
   return (neuron0x10fd8290()*-0.735558);
}

double luNNEB::synapse0x10850920() {
   return (neuron0x10fd85d0()*0.0368447);
}

double luNNEB::synapse0x10850960() {
   return (neuron0x10fd8910()*0.407403);
}

double luNNEB::synapse0x108509a0() {
   return (neuron0x10fd8c50()*-0.414209);
}

double luNNEB::synapse0x108509e0() {
   return (neuron0x10fd8f90()*-0.606965);
}

double luNNEB::synapse0x10fdc6d0() {
   return (neuron0x10fd78d0()*0.46383);
}

double luNNEB::synapse0x10fdc710() {
   return (neuron0x10fd7c10()*1.2554);
}

double luNNEB::synapse0x10fdc750() {
   return (neuron0x10fd7f50()*0.178097);
}

double luNNEB::synapse0x10fdc790() {
   return (neuron0x10fd8290()*0.49485);
}

double luNNEB::synapse0x10fdc7d0() {
   return (neuron0x10fd85d0()*0.191809);
}

double luNNEB::synapse0x10fdc810() {
   return (neuron0x10fd8910()*-0.0236312);
}

double luNNEB::synapse0x10fdc850() {
   return (neuron0x10fd8c50()*-0.0430808);
}

double luNNEB::synapse0x10fdc890() {
   return (neuron0x10fd8f90()*1.10224);
}

double luNNEB::synapse0x10fdcc10() {
   return (neuron0x10fd9400()*-3.072);
}

double luNNEB::synapse0x10fdcc50() {
   return (neuron0x10fd9830()*1.9901);
}

double luNNEB::synapse0x10fdcc90() {
   return (neuron0x10fd9d70()*-0.337279);
}

double luNNEB::synapse0x10fdccd0() {
   return (neuron0x10fda340()*-1.69344);
}

double luNNEB::synapse0x10fdcd10() {
   return (neuron0x10fda880()*1.27032);
}

double luNNEB::synapse0x10fdcd50() {
   return (neuron0x10fdac00()*-1.50811);
}

double luNNEB::synapse0x10fdcd90() {
   return (neuron0x10fdb140()*4.59316);
}

double luNNEB::synapse0x10fdcdd0() {
   return (neuron0x10fdb680()*-0.19978);
}

double luNNEB::synapse0x10fdce10() {
   return (neuron0x10fdbbc0()*2.61239);
}

double luNNEB::synapse0x10fdce50() {
   return (neuron0x10fdc420()*4.132);
}

double luNNEB::synapse0x10fdd1d0() {
   return (neuron0x10fd9400()*-1.53604);
}

double luNNEB::synapse0x10fdd210() {
   return (neuron0x10fd9830()*0.946982);
}

double luNNEB::synapse0x10fdd250() {
   return (neuron0x10fd9d70()*-0.291547);
}

double luNNEB::synapse0x10fdd290() {
   return (neuron0x10fda340()*-1.67342);
}

double luNNEB::synapse0x10fdd2d0() {
   return (neuron0x10fda880()*-0.206261);
}

double luNNEB::synapse0x10fdd310() {
   return (neuron0x10fdac00()*-0.889833);
}

double luNNEB::synapse0x10fdd350() {
   return (neuron0x10fdb140()*1.5124);
}

double luNNEB::synapse0x10fdd390() {
   return (neuron0x10fdb680()*-0.112377);
}

double luNNEB::synapse0x10fdd3d0() {
   return (neuron0x10fdbbc0()*-0.15154);
}

double luNNEB::synapse0x10fdd410() {
   return (neuron0x10fdc420()*-0.219815);
}

double luNNEB::synapse0x10fdd790() {
   return (neuron0x10fd9400()*0.953703);
}

double luNNEB::synapse0x10fdd7d0() {
   return (neuron0x10fd9830()*-0.0869858);
}

double luNNEB::synapse0x10fdd810() {
   return (neuron0x10fd9d70()*-0.203974);
}

double luNNEB::synapse0x10fdd850() {
   return (neuron0x10fda340()*0.593883);
}

double luNNEB::synapse0x10fdd890() {
   return (neuron0x10fda880()*-0.368326);
}

double luNNEB::synapse0x10fdd8d0() {
   return (neuron0x10fdac00()*0.86465);
}

double luNNEB::synapse0x10fdd910() {
   return (neuron0x10fdb140()*-1.45982);
}

double luNNEB::synapse0x10fdd950() {
   return (neuron0x10fdb680()*0.0559001);
}

double luNNEB::synapse0x10fdd990() {
   return (neuron0x10fdbbc0()*-1.13586);
}

double luNNEB::synapse0x10fdd9d0() {
   return (neuron0x10fdc420()*-1.47209);
}

double luNNEB::synapse0x10fddd50() {
   return (neuron0x10fd9400()*-4.16288);
}

double luNNEB::synapse0x10fddd90() {
   return (neuron0x10fd9830()*0.670889);
}

double luNNEB::synapse0x10fdddd0() {
   return (neuron0x10fd9d70()*-1.31031);
}

double luNNEB::synapse0x10fdde10() {
   return (neuron0x10fda340()*-0.339257);
}

double luNNEB::synapse0x10fdde50() {
   return (neuron0x10fda880()*2.24316);
}

double luNNEB::synapse0x10fdde90() {
   return (neuron0x10fdac00()*-1.00384);
}

double luNNEB::synapse0x10fdded0() {
   return (neuron0x10fdb140()*3.56398);
}

double luNNEB::synapse0x10fddf10() {
   return (neuron0x10fdb680()*0.835351);
}

double luNNEB::synapse0x10fddf50() {
   return (neuron0x10fdbbc0()*3.77029);
}

double luNNEB::synapse0x10fddf90() {
   return (neuron0x10fdc420()*3.42921);
}

double luNNEB::synapse0x10fde310() {
   return (neuron0x10fd9400()*3.06253);
}

double luNNEB::synapse0x10fde350() {
   return (neuron0x10fd9830()*1.96367);
}

double luNNEB::synapse0x10fde390() {
   return (neuron0x10fd9d70()*-0.293902);
}

double luNNEB::synapse0x10fde3d0() {
   return (neuron0x10fda340()*1.84213);
}

double luNNEB::synapse0x10fde410() {
   return (neuron0x10fda880()*0.259737);
}

double luNNEB::synapse0x10fde450() {
   return (neuron0x10fdac00()*2.22645);
}

double luNNEB::synapse0x10fde490() {
   return (neuron0x10fdb140()*-2.77222);
}

double luNNEB::synapse0x10fde4d0() {
   return (neuron0x10fdb680()*-0.297916);
}

double luNNEB::synapse0x10fde510() {
   return (neuron0x10fdbbc0()*-0.219784);
}

double luNNEB::synapse0x10fdc010() {
   return (neuron0x10fdc420()*-0.502024);
}

double luNNEB::synapse0x10fdc390() {
   return (neuron0x10fdc8d0()*2.88626);
}

double luNNEB::synapse0x10fdc3d0() {
   return (neuron0x10fdce90()*3.92546);
}

double luNNEB::synapse0x10fd92d0() {
   return (neuron0x10fdd450()*-0.55782);
}

double luNNEB::synapse0x10fd9310() {
   return (neuron0x10fdda10()*-0.59229);
}

double luNNEB::synapse0x10fd9350() {
   return (neuron0x10fddfd0()*-8.44197);
}

double rdNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.15492)/1.68616;
   input2 = (in2 - 1.14956)/1.69316;
   input3 = (in3 - 1.89563)/1.49913;
   input4 = (in4 - 1.90666)/1.49442;
   input5 = (in5 - 0.3065)/1.46726;
   input6 = (in6 - 0.305354)/1.51455;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x14ac00d0();
     default:
         return 0.;
   }
}

double rdNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.15492)/1.68616;
   input2 = (input[2] - 1.14956)/1.69316;
   input3 = (input[3] - 1.89563)/1.49913;
   input4 = (input[4] - 1.90666)/1.49442;
   input5 = (input[5] - 0.3065)/1.46726;
   input6 = (input[6] - 0.305354)/1.51455;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x14ac00d0();
     default:
         return 0.;
   }
}

double rdNNEB::neuron0x14abb950() {
   return input0;
}

double rdNNEB::neuron0x14abbc90() {
   return input1;
}

double rdNNEB::neuron0x14abbfd0() {
   return input2;
}

double rdNNEB::neuron0x14abc310() {
   return input3;
}

double rdNNEB::neuron0x14abc650() {
   return input4;
}

double rdNNEB::neuron0x14abc990() {
   return input5;
}

double rdNNEB::neuron0x14abccd0() {
   return input6;
}

double rdNNEB::neuron0x14abd010() {
   return input7;
}

double rdNNEB::input0x14abd480() {
   double input = 2.42798;
   input += synapse0x14a1c350();
   input += synapse0x14ac4550();
   input += synapse0x14abd730();
   input += synapse0x14abd770();
   input += synapse0x14abd7b0();
   input += synapse0x14abd7f0();
   input += synapse0x14abd830();
   input += synapse0x14abd870();
   return input;
}

double rdNNEB::neuron0x14abd480() {
   double input = input0x14abd480();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abd8b0() {
   double input = -0.411164;
   input += synapse0x14abdbf0();
   input += synapse0x14abdc30();
   input += synapse0x14abdc70();
   input += synapse0x14abdcb0();
   input += synapse0x14abdcf0();
   input += synapse0x14abdd30();
   input += synapse0x14abdd70();
   input += synapse0x14abddb0();
   return input;
}

double rdNNEB::neuron0x14abd8b0() {
   double input = input0x14abd8b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abddf0() {
   double input = 1.80772;
   input += synapse0x14abe130();
   input += synapse0x1431b010();
   input += synapse0x1431b050();
   input += synapse0x14abe280();
   input += synapse0x14abe2c0();
   input += synapse0x14abe300();
   input += synapse0x14abe340();
   input += synapse0x14abe380();
   return input;
}

double rdNNEB::neuron0x14abddf0() {
   double input = input0x14abddf0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abe3c0() {
   double input = 0.642085;
   input += synapse0x14abe700();
   input += synapse0x14abe740();
   input += synapse0x14abe780();
   input += synapse0x14abe7c0();
   input += synapse0x14abe800();
   input += synapse0x14abe840();
   input += synapse0x14abe880();
   input += synapse0x14abe8c0();
   return input;
}

double rdNNEB::neuron0x14abe3c0() {
   double input = input0x14abe3c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abe900() {
   double input = 1.10385;
   input += synapse0x14abec40();
   input += synapse0x14abb880();
   input += synapse0x14ac4590();
   input += synapse0x14319b50();
   input += synapse0x14abe170();
   input += synapse0x14abe1b0();
   input += synapse0x14abe1f0();
   input += synapse0x14abe230();
   return input;
}

double rdNNEB::neuron0x14abe900() {
   double input = input0x14abe900();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abec80() {
   double input = 1.58604;
   input += synapse0x14abefc0();
   input += synapse0x14abf000();
   input += synapse0x14abf040();
   input += synapse0x14abf080();
   input += synapse0x14abf0c0();
   input += synapse0x14abf100();
   input += synapse0x14abf140();
   input += synapse0x14abf180();
   return input;
}

double rdNNEB::neuron0x14abec80() {
   double input = input0x14abec80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abf1c0() {
   double input = -0.249532;
   input += synapse0x14abf500();
   input += synapse0x14abf540();
   input += synapse0x14abf580();
   input += synapse0x14abf5c0();
   input += synapse0x14abf600();
   input += synapse0x14abf640();
   input += synapse0x14abf680();
   input += synapse0x14abf6c0();
   return input;
}

double rdNNEB::neuron0x14abf1c0() {
   double input = input0x14abf1c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abf700() {
   double input = 0.976559;
   input += synapse0x14abfa40();
   input += synapse0x14abfa80();
   input += synapse0x14abfac0();
   input += synapse0x14abfb00();
   input += synapse0x14abfb40();
   input += synapse0x14abfb80();
   input += synapse0x14abfbc0();
   input += synapse0x14abfc00();
   return input;
}

double rdNNEB::neuron0x14abf700() {
   double input = input0x14abf700();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abfc40() {
   double input = -1.43635;
   input += synapse0x14208f70();
   input += synapse0x14208fb0();
   input += synapse0x14334920();
   input += synapse0x14334960();
   input += synapse0x143349a0();
   input += synapse0x143349e0();
   input += synapse0x14334a20();
   input += synapse0x14334a60();
   return input;
}

double rdNNEB::neuron0x14abfc40() {
   double input = input0x14abfc40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac04a0() {
   double input = -0.127806;
   input += synapse0x14ac0750();
   input += synapse0x14ac0790();
   input += synapse0x14ac07d0();
   input += synapse0x14ac0810();
   input += synapse0x14ac0850();
   input += synapse0x14ac0890();
   input += synapse0x14ac08d0();
   input += synapse0x14ac0910();
   return input;
}

double rdNNEB::neuron0x14ac04a0() {
   double input = input0x14ac04a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac0950() {
   double input = 0.455212;
   input += synapse0x14ac0c90();
   input += synapse0x14ac0cd0();
   input += synapse0x14ac0d10();
   input += synapse0x14ac0d50();
   input += synapse0x14ac0d90();
   input += synapse0x14ac0dd0();
   input += synapse0x14ac0e10();
   input += synapse0x14ac0e50();
   input += synapse0x14ac0e90();
   input += synapse0x14ac0ed0();
   return input;
}

double rdNNEB::neuron0x14ac0950() {
   double input = input0x14ac0950();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac0f10() {
   double input = -0.695286;
   input += synapse0x14ac1250();
   input += synapse0x14ac1290();
   input += synapse0x14ac12d0();
   input += synapse0x14ac1310();
   input += synapse0x14ac1350();
   input += synapse0x14ac1390();
   input += synapse0x14ac13d0();
   input += synapse0x14ac1410();
   input += synapse0x14ac1450();
   input += synapse0x14ac1490();
   return input;
}

double rdNNEB::neuron0x14ac0f10() {
   double input = input0x14ac0f10();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac14d0() {
   double input = -0.580753;
   input += synapse0x14ac1810();
   input += synapse0x14ac1850();
   input += synapse0x14ac1890();
   input += synapse0x14ac18d0();
   input += synapse0x14ac1910();
   input += synapse0x14ac1950();
   input += synapse0x14ac1990();
   input += synapse0x14ac19d0();
   input += synapse0x14ac1a10();
   input += synapse0x14ac1a50();
   return input;
}

double rdNNEB::neuron0x14ac14d0() {
   double input = input0x14ac14d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac1a90() {
   double input = 0.49098;
   input += synapse0x14ac1dd0();
   input += synapse0x14ac1e10();
   input += synapse0x14ac1e50();
   input += synapse0x14ac1e90();
   input += synapse0x14ac1ed0();
   input += synapse0x14ac1f10();
   input += synapse0x14ac1f50();
   input += synapse0x14ac1f90();
   input += synapse0x14ac1fd0();
   input += synapse0x14ac2010();
   return input;
}

double rdNNEB::neuron0x14ac1a90() {
   double input = input0x14ac1a90();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac2050() {
   double input = -0.0961925;
   input += synapse0x14ac2390();
   input += synapse0x14ac23d0();
   input += synapse0x14ac2410();
   input += synapse0x14ac2450();
   input += synapse0x14ac2490();
   input += synapse0x14ac24d0();
   input += synapse0x14ac2510();
   input += synapse0x14ac2550();
   input += synapse0x14ac2590();
   input += synapse0x14ac0090();
   return input;
}

double rdNNEB::neuron0x14ac2050() {
   double input = input0x14ac2050();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac00d0() {
   double input = 1.63819;
   input += synapse0x14ac0410();
   input += synapse0x14ac0450();
   input += synapse0x14abd350();
   input += synapse0x14abd390();
   input += synapse0x14abd3d0();
   return input;
}

double rdNNEB::neuron0x14ac00d0() {
   double input = input0x14ac00d0();
   return (input * 1)+0;
}

double rdNNEB::synapse0x14a1c350() {
   return (neuron0x14abb950()*-0.216406);
}

double rdNNEB::synapse0x14ac4550() {
   return (neuron0x14abbc90()*-0.420975);
}

double rdNNEB::synapse0x14abd730() {
   return (neuron0x14abbfd0()*-0.56903);
}

double rdNNEB::synapse0x14abd770() {
   return (neuron0x14abc310()*0.170109);
}

double rdNNEB::synapse0x14abd7b0() {
   return (neuron0x14abc650()*1.72569);
}

double rdNNEB::synapse0x14abd7f0() {
   return (neuron0x14abc990()*-0.141621);
}

double rdNNEB::synapse0x14abd830() {
   return (neuron0x14abccd0()*0.258394);
}

double rdNNEB::synapse0x14abd870() {
   return (neuron0x14abd010()*-0.0122614);
}

double rdNNEB::synapse0x14abdbf0() {
   return (neuron0x14abb950()*1.14533);
}

double rdNNEB::synapse0x14abdc30() {
   return (neuron0x14abbc90()*-0.415792);
}

double rdNNEB::synapse0x14abdc70() {
   return (neuron0x14abbfd0()*-0.661272);
}

double rdNNEB::synapse0x14abdcb0() {
   return (neuron0x14abc310()*0.0745773);
}

double rdNNEB::synapse0x14abdcf0() {
   return (neuron0x14abc650()*-0.0324746);
}

double rdNNEB::synapse0x14abdd30() {
   return (neuron0x14abc990()*0.0664283);
}

double rdNNEB::synapse0x14abdd70() {
   return (neuron0x14abccd0()*-1.67815);
}

double rdNNEB::synapse0x14abddb0() {
   return (neuron0x14abd010()*0.343732);
}

double rdNNEB::synapse0x14abe130() {
   return (neuron0x14abb950()*1.70759);
}

double rdNNEB::synapse0x1431b010() {
   return (neuron0x14abbc90()*-0.829353);
}

double rdNNEB::synapse0x1431b050() {
   return (neuron0x14abbfd0()*0.728579);
}

double rdNNEB::synapse0x14abe280() {
   return (neuron0x14abc310()*0.608541);
}

double rdNNEB::synapse0x14abe2c0() {
   return (neuron0x14abc650()*-0.991141);
}

double rdNNEB::synapse0x14abe300() {
   return (neuron0x14abc990()*-0.397236);
}

double rdNNEB::synapse0x14abe340() {
   return (neuron0x14abccd0()*-0.505365);
}

double rdNNEB::synapse0x14abe380() {
   return (neuron0x14abd010()*0.0697281);
}

double rdNNEB::synapse0x14abe700() {
   return (neuron0x14abb950()*-0.606239);
}

double rdNNEB::synapse0x14abe740() {
   return (neuron0x14abbc90()*1.85788);
}

double rdNNEB::synapse0x14abe780() {
   return (neuron0x14abbfd0()*-0.0541781);
}

double rdNNEB::synapse0x14abe7c0() {
   return (neuron0x14abc310()*-0.156056);
}

double rdNNEB::synapse0x14abe800() {
   return (neuron0x14abc650()*-0.511695);
}

double rdNNEB::synapse0x14abe840() {
   return (neuron0x14abc990()*-0.0604903);
}

double rdNNEB::synapse0x14abe880() {
   return (neuron0x14abccd0()*0.296835);
}

double rdNNEB::synapse0x14abe8c0() {
   return (neuron0x14abd010()*0.151318);
}

double rdNNEB::synapse0x14abec40() {
   return (neuron0x14abb950()*0.45892);
}

double rdNNEB::synapse0x14abb880() {
   return (neuron0x14abbc90()*-0.113319);
}

double rdNNEB::synapse0x14ac4590() {
   return (neuron0x14abbfd0()*-0.0432273);
}

double rdNNEB::synapse0x14319b50() {
   return (neuron0x14abc310()*-1.20745);
}

double rdNNEB::synapse0x14abe170() {
   return (neuron0x14abc650()*-0.414526);
}

double rdNNEB::synapse0x14abe1b0() {
   return (neuron0x14abc990()*0.388562);
}

double rdNNEB::synapse0x14abe1f0() {
   return (neuron0x14abccd0()*0.581199);
}

double rdNNEB::synapse0x14abe230() {
   return (neuron0x14abd010()*-0.242221);
}

double rdNNEB::synapse0x14abefc0() {
   return (neuron0x14abb950()*-1.53883);
}

double rdNNEB::synapse0x14abf000() {
   return (neuron0x14abbc90()*-0.0712372);
}

double rdNNEB::synapse0x14abf040() {
   return (neuron0x14abbfd0()*-0.00473586);
}

double rdNNEB::synapse0x14abf080() {
   return (neuron0x14abc310()*0.202715);
}

double rdNNEB::synapse0x14abf0c0() {
   return (neuron0x14abc650()*0.403952);
}

double rdNNEB::synapse0x14abf100() {
   return (neuron0x14abc990()*0.306872);
}

double rdNNEB::synapse0x14abf140() {
   return (neuron0x14abccd0()*0.256241);
}

double rdNNEB::synapse0x14abf180() {
   return (neuron0x14abd010()*0.203546);
}

double rdNNEB::synapse0x14abf500() {
   return (neuron0x14abb950()*0.749902);
}

double rdNNEB::synapse0x14abf540() {
   return (neuron0x14abbc90()*0.186212);
}

double rdNNEB::synapse0x14abf580() {
   return (neuron0x14abbfd0()*0.641923);
}

double rdNNEB::synapse0x14abf5c0() {
   return (neuron0x14abc310()*0.0616216);
}

double rdNNEB::synapse0x14abf600() {
   return (neuron0x14abc650()*0.0547814);
}

double rdNNEB::synapse0x14abf640() {
   return (neuron0x14abc990()*0.150325);
}

double rdNNEB::synapse0x14abf680() {
   return (neuron0x14abccd0()*-0.037531);
}

double rdNNEB::synapse0x14abf6c0() {
   return (neuron0x14abd010()*0.00331658);
}

double rdNNEB::synapse0x14abfa40() {
   return (neuron0x14abb950()*-0.0260585);
}

double rdNNEB::synapse0x14abfa80() {
   return (neuron0x14abbc90()*0.243857);
}

double rdNNEB::synapse0x14abfac0() {
   return (neuron0x14abbfd0()*0.556623);
}

double rdNNEB::synapse0x14abfb00() {
   return (neuron0x14abc310()*0.786342);
}

double rdNNEB::synapse0x14abfb40() {
   return (neuron0x14abc650()*-1.22138);
}

double rdNNEB::synapse0x14abfb80() {
   return (neuron0x14abc990()*-0.180283);
}

double rdNNEB::synapse0x14abfbc0() {
   return (neuron0x14abccd0()*-0.557386);
}

double rdNNEB::synapse0x14abfc00() {
   return (neuron0x14abd010()*0.194623);
}

double rdNNEB::synapse0x14208f70() {
   return (neuron0x14abb950()*-0.560909);
}

double rdNNEB::synapse0x14208fb0() {
   return (neuron0x14abbc90()*0.618483);
}

double rdNNEB::synapse0x14334920() {
   return (neuron0x14abbfd0()*1.56332);
}

double rdNNEB::synapse0x14334960() {
   return (neuron0x14abc310()*-0.543328);
}

double rdNNEB::synapse0x143349a0() {
   return (neuron0x14abc650()*-0.183581);
}

double rdNNEB::synapse0x143349e0() {
   return (neuron0x14abc990()*0.7814);
}

double rdNNEB::synapse0x14334a20() {
   return (neuron0x14abccd0()*-0.104609);
}

double rdNNEB::synapse0x14334a60() {
   return (neuron0x14abd010()*-0.376717);
}

double rdNNEB::synapse0x14ac0750() {
   return (neuron0x14abb950()*-1.3224);
}

double rdNNEB::synapse0x14ac0790() {
   return (neuron0x14abbc90()*0.05359);
}

double rdNNEB::synapse0x14ac07d0() {
   return (neuron0x14abbfd0()*0.682048);
}

double rdNNEB::synapse0x14ac0810() {
   return (neuron0x14abc310()*-0.249621);
}

double rdNNEB::synapse0x14ac0850() {
   return (neuron0x14abc650()*-0.0417721);
}

double rdNNEB::synapse0x14ac0890() {
   return (neuron0x14abc990()*0.099426);
}

double rdNNEB::synapse0x14ac08d0() {
   return (neuron0x14abccd0()*0.417192);
}

double rdNNEB::synapse0x14ac0910() {
   return (neuron0x14abd010()*0.23391);
}

double rdNNEB::synapse0x14ac0c90() {
   return (neuron0x14abd480()*0.0948073);
}

double rdNNEB::synapse0x14ac0cd0() {
   return (neuron0x14abd8b0()*-1.02893);
}

double rdNNEB::synapse0x14ac0d10() {
   return (neuron0x14abddf0()*-0.509918);
}

double rdNNEB::synapse0x14ac0d50() {
   return (neuron0x14abe3c0()*0.867766);
}

double rdNNEB::synapse0x14ac0d90() {
   return (neuron0x14abe900()*0.664667);
}

double rdNNEB::synapse0x14ac0dd0() {
   return (neuron0x14abec80()*0.659037);
}

double rdNNEB::synapse0x14ac0e10() {
   return (neuron0x14abf1c0()*-1.63301);
}

double rdNNEB::synapse0x14ac0e50() {
   return (neuron0x14abf700()*0.626759);
}

double rdNNEB::synapse0x14ac0e90() {
   return (neuron0x14abfc40()*-0.251753);
}

double rdNNEB::synapse0x14ac0ed0() {
   return (neuron0x14ac04a0()*1.72268);
}

double rdNNEB::synapse0x14ac1250() {
   return (neuron0x14abd480()*0.86684);
}

double rdNNEB::synapse0x14ac1290() {
   return (neuron0x14abd8b0()*-0.228328);
}

double rdNNEB::synapse0x14ac12d0() {
   return (neuron0x14abddf0()*-2.27991);
}

double rdNNEB::synapse0x14ac1310() {
   return (neuron0x14abe3c0()*2.19628);
}

double rdNNEB::synapse0x14ac1350() {
   return (neuron0x14abe900()*-1.12465);
}

double rdNNEB::synapse0x14ac1390() {
   return (neuron0x14abec80()*-2.69582);
}

double rdNNEB::synapse0x14ac13d0() {
   return (neuron0x14abf1c0()*0.388994);
}

double rdNNEB::synapse0x14ac1410() {
   return (neuron0x14abf700()*-1.59652);
}

double rdNNEB::synapse0x14ac1450() {
   return (neuron0x14abfc40()*1.86835);
}

double rdNNEB::synapse0x14ac1490() {
   return (neuron0x14ac04a0()*1.49056);
}

double rdNNEB::synapse0x14ac1810() {
   return (neuron0x14abd480()*-0.290069);
}

double rdNNEB::synapse0x14ac1850() {
   return (neuron0x14abd8b0()*-0.803451);
}

double rdNNEB::synapse0x14ac1890() {
   return (neuron0x14abddf0()*0.00307815);
}

double rdNNEB::synapse0x14ac18d0() {
   return (neuron0x14abe3c0()*-0.777897);
}

double rdNNEB::synapse0x14ac1910() {
   return (neuron0x14abe900()*-0.291359);
}

double rdNNEB::synapse0x14ac1950() {
   return (neuron0x14abec80()*0.501869);
}

double rdNNEB::synapse0x14ac1990() {
   return (neuron0x14abf1c0()*0.991883);
}

double rdNNEB::synapse0x14ac19d0() {
   return (neuron0x14abf700()*-0.601744);
}

double rdNNEB::synapse0x14ac1a10() {
   return (neuron0x14abfc40()*-0.705386);
}

double rdNNEB::synapse0x14ac1a50() {
   return (neuron0x14ac04a0()*-1.02517);
}

double rdNNEB::synapse0x14ac1dd0() {
   return (neuron0x14abd480()*1.05875);
}

double rdNNEB::synapse0x14ac1e10() {
   return (neuron0x14abd8b0()*0.0280678);
}

double rdNNEB::synapse0x14ac1e50() {
   return (neuron0x14abddf0()*0.310589);
}

double rdNNEB::synapse0x14ac1e90() {
   return (neuron0x14abe3c0()*0.950605);
}

double rdNNEB::synapse0x14ac1ed0() {
   return (neuron0x14abe900()*-0.141197);
}

double rdNNEB::synapse0x14ac1f10() {
   return (neuron0x14abec80()*-0.781974);
}

double rdNNEB::synapse0x14ac1f50() {
   return (neuron0x14abf1c0()*1.51498);
}

double rdNNEB::synapse0x14ac1f90() {
   return (neuron0x14abf700()*-0.23685);
}

double rdNNEB::synapse0x14ac1fd0() {
   return (neuron0x14abfc40()*0.881958);
}

double rdNNEB::synapse0x14ac2010() {
   return (neuron0x14ac04a0()*-0.707959);
}

double rdNNEB::synapse0x14ac2390() {
   return (neuron0x14abd480()*-1.16238);
}

double rdNNEB::synapse0x14ac23d0() {
   return (neuron0x14abd8b0()*-0.98238);
}

double rdNNEB::synapse0x14ac2410() {
   return (neuron0x14abddf0()*1.1459);
}

double rdNNEB::synapse0x14ac2450() {
   return (neuron0x14abe3c0()*-1.25397);
}

double rdNNEB::synapse0x14ac2490() {
   return (neuron0x14abe900()*1.4953);
}

double rdNNEB::synapse0x14ac24d0() {
   return (neuron0x14abec80()*1.63361);
}

double rdNNEB::synapse0x14ac2510() {
   return (neuron0x14abf1c0()*-0.665746);
}

double rdNNEB::synapse0x14ac2550() {
   return (neuron0x14abf700()*0.569317);
}

double rdNNEB::synapse0x14ac2590() {
   return (neuron0x14abfc40()*-0.375751);
}

double rdNNEB::synapse0x14ac0090() {
   return (neuron0x14ac04a0()*-0.472168);
}

double rdNNEB::synapse0x14ac0410() {
   return (neuron0x14ac0950()*-2.71126);
}

double rdNNEB::synapse0x14ac0450() {
   return (neuron0x14ac0f10()*5.31703);
}

double rdNNEB::synapse0x14abd350() {
   return (neuron0x14ac14d0()*0.943595);
}

double rdNNEB::synapse0x14abd390() {
   return (neuron0x14ac1a90()*4.36333);
}

double rdNNEB::synapse0x14abd3d0() {
   return (neuron0x14ac2050()*-4.62187);
}


double rrNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.14956)/1.69316;
   input2 = (in2 - 1.89563)/1.49913;
   input3 = (in3 - 1.90666)/1.49442;
   input4 = (in4 - 0.3065)/1.46726;
   input5 = (in5 - 0.318454)/1.50742;
   input6 = (in6 - 0.305354)/1.51455;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0xb0b3110();
     default:
         return 0.;
   }
}

double rrNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.14956)/1.69316;
   input2 = (input[2] - 1.89563)/1.49913;
   input3 = (input[3] - 1.90666)/1.49442;
   input4 = (input[4] - 0.3065)/1.46726;
   input5 = (input[5] - 0.318454)/1.50742;
   input6 = (input[6] - 0.305354)/1.51455;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0xb0b3110();
     default:
         return 0.;
   }
}

double rrNNEB::neuron0xb0ae990() {
   return input0;
}

double rrNNEB::neuron0xb0aecd0() {
   return input1;
}

double rrNNEB::neuron0xb0af010() {
   return input2;
}

double rrNNEB::neuron0xb0af350() {
   return input3;
}

double rrNNEB::neuron0xb0af690() {
   return input4;
}

double rrNNEB::neuron0xb0af9d0() {
   return input5;
}

double rrNNEB::neuron0xb0afd10() {
   return input6;
}

double rrNNEB::neuron0xb0b0050() {
   return input7;
}

double rrNNEB::input0xb0b04c0() {
   double input = 1.16126;
   input += synapse0xb00f390();
   input += synapse0xb0b7590();
   input += synapse0xb0b0770();
   input += synapse0xb0b07b0();
   input += synapse0xb0b07f0();
   input += synapse0xb0b0830();
   input += synapse0xb0b0870();
   input += synapse0xb0b08b0();
   return input;
}

double rrNNEB::neuron0xb0b04c0() {
   double input = input0xb0b04c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b08f0() {
   double input = 0.288451;
   input += synapse0xb0b0c30();
   input += synapse0xb0b0c70();
   input += synapse0xb0b0cb0();
   input += synapse0xb0b0cf0();
   input += synapse0xb0b0d30();
   input += synapse0xb0b0d70();
   input += synapse0xb0b0db0();
   input += synapse0xb0b0df0();
   return input;
}

double rrNNEB::neuron0xb0b08f0() {
   double input = input0xb0b08f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b0e30() {
   double input = 0.390796;
   input += synapse0xb0b1170();
   input += synapse0xa90e050();
   input += synapse0xa90e090();
   input += synapse0xb0b12c0();
   input += synapse0xb0b1300();
   input += synapse0xb0b1340();
   input += synapse0xb0b1380();
   input += synapse0xb0b13c0();
   return input;
}

double rrNNEB::neuron0xb0b0e30() {
   double input = input0xb0b0e30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b1400() {
   double input = -0.706336;
   input += synapse0xb0b1740();
   input += synapse0xb0b1780();
   input += synapse0xb0b17c0();
   input += synapse0xb0b1800();
   input += synapse0xb0b1840();
   input += synapse0xb0b1880();
   input += synapse0xb0b18c0();
   input += synapse0xb0b1900();
   return input;
}

double rrNNEB::neuron0xb0b1400() {
   double input = input0xb0b1400();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b1940() {
   double input = -1.43078;
   input += synapse0xb0b1c80();
   input += synapse0xb0ae8c0();
   input += synapse0xb0b75d0();
   input += synapse0xa90cb90();
   input += synapse0xb0b11b0();
   input += synapse0xb0b11f0();
   input += synapse0xb0b1230();
   input += synapse0xb0b1270();
   return input;
}

double rrNNEB::neuron0xb0b1940() {
   double input = input0xb0b1940();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b1cc0() {
   double input = 0.686351;
   input += synapse0xb0b2000();
   input += synapse0xb0b2040();
   input += synapse0xb0b2080();
   input += synapse0xb0b20c0();
   input += synapse0xb0b2100();
   input += synapse0xb0b2140();
   input += synapse0xb0b2180();
   input += synapse0xb0b21c0();
   return input;
}

double rrNNEB::neuron0xb0b1cc0() {
   double input = input0xb0b1cc0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b2200() {
   double input = 1.51256;
   input += synapse0xb0b2540();
   input += synapse0xb0b2580();
   input += synapse0xb0b25c0();
   input += synapse0xb0b2600();
   input += synapse0xb0b2640();
   input += synapse0xb0b2680();
   input += synapse0xb0b26c0();
   input += synapse0xb0b2700();
   return input;
}

double rrNNEB::neuron0xb0b2200() {
   double input = input0xb0b2200();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b2740() {
   double input = 0.163221;
   input += synapse0xb0b2a80();
   input += synapse0xb0b2ac0();
   input += synapse0xb0b2b00();
   input += synapse0xb0b2b40();
   input += synapse0xb0b2b80();
   input += synapse0xb0b2bc0();
   input += synapse0xb0b2c00();
   input += synapse0xb0b2c40();
   return input;
}

double rrNNEB::neuron0xb0b2740() {
   double input = input0xb0b2740();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b2c80() {
   double input = 0.103629;
   input += synapse0xa7fbfb0();
   input += synapse0xa7fbff0();
   input += synapse0xa927960();
   input += synapse0xa9279a0();
   input += synapse0xa9279e0();
   input += synapse0xa927a20();
   input += synapse0xa927a60();
   input += synapse0xa927aa0();
   return input;
}

double rrNNEB::neuron0xb0b2c80() {
   double input = input0xb0b2c80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b34e0() {
   double input = 1.15988;
   input += synapse0xb0b3790();
   input += synapse0xb0b37d0();
   input += synapse0xb0b3810();
   input += synapse0xb0b3850();
   input += synapse0xb0b3890();
   input += synapse0xb0b38d0();
   input += synapse0xb0b3910();
   input += synapse0xb0b3950();
   return input;
}

double rrNNEB::neuron0xb0b34e0() {
   double input = input0xb0b34e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b3990() {
   double input = 0.596451;
   input += synapse0xb0b3cd0();
   input += synapse0xb0b3d10();
   input += synapse0xb0b3d50();
   input += synapse0xb0b3d90();
   input += synapse0xb0b3dd0();
   input += synapse0xb0b3e10();
   input += synapse0xb0b3e50();
   input += synapse0xb0b3e90();
   input += synapse0xb0b3ed0();
   input += synapse0xb0b3f10();
   return input;
}

double rrNNEB::neuron0xb0b3990() {
   double input = input0xb0b3990();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b3f50() {
   double input = -0.30359;
   input += synapse0xb0b4290();
   input += synapse0xb0b42d0();
   input += synapse0xb0b4310();
   input += synapse0xb0b4350();
   input += synapse0xb0b4390();
   input += synapse0xb0b43d0();
   input += synapse0xb0b4410();
   input += synapse0xb0b4450();
   input += synapse0xb0b4490();
   input += synapse0xb0b44d0();
   return input;
}

double rrNNEB::neuron0xb0b3f50() {
   double input = input0xb0b3f50();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b4510() {
   double input = 0.0520657;
   input += synapse0xb0b4850();
   input += synapse0xb0b4890();
   input += synapse0xb0b48d0();
   input += synapse0xb0b4910();
   input += synapse0xb0b4950();
   input += synapse0xb0b4990();
   input += synapse0xb0b49d0();
   input += synapse0xb0b4a10();
   input += synapse0xb0b4a50();
   input += synapse0xb0b4a90();
   return input;
}

double rrNNEB::neuron0xb0b4510() {
   double input = input0xb0b4510();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b4ad0() {
   double input = -0.196957;
   input += synapse0xb0b4e10();
   input += synapse0xb0b4e50();
   input += synapse0xb0b4e90();
   input += synapse0xb0b4ed0();
   input += synapse0xb0b4f10();
   input += synapse0xb0b4f50();
   input += synapse0xb0b4f90();
   input += synapse0xb0b4fd0();
   input += synapse0xb0b5010();
   input += synapse0xb0b5050();
   return input;
}

double rrNNEB::neuron0xb0b4ad0() {
   double input = input0xb0b4ad0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b5090() {
   double input = -0.692853;
   input += synapse0xb0b53d0();
   input += synapse0xb0b5410();
   input += synapse0xb0b5450();
   input += synapse0xb0b5490();
   input += synapse0xb0b54d0();
   input += synapse0xb0b5510();
   input += synapse0xb0b5550();
   input += synapse0xb0b5590();
   input += synapse0xb0b55d0();
   input += synapse0xb0b30d0();
   return input;
}

double rrNNEB::neuron0xb0b5090() {
   double input = input0xb0b5090();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEB::input0xb0b3110() {
   double input = -5.05639;
   input += synapse0xb0b3450();
   input += synapse0xb0b3490();
   input += synapse0xb0b0390();
   input += synapse0xb0b03d0();
   input += synapse0xb0b0410();
   return input;
}

double rrNNEB::neuron0xb0b3110() {
   double input = input0xb0b3110();
   return (input * 1)+0;
}

double rrNNEB::synapse0xb00f390() {
   return (neuron0xb0ae990()*0.605152);
}

double rrNNEB::synapse0xb0b7590() {
   return (neuron0xb0aecd0()*-0.0922336);
}

double rrNNEB::synapse0xb0b0770() {
   return (neuron0xb0af010()*0.149301);
}

double rrNNEB::synapse0xb0b07b0() {
   return (neuron0xb0af350()*0.0738282);
}

double rrNNEB::synapse0xb0b07f0() {
   return (neuron0xb0af690()*-0.352642);
}

double rrNNEB::synapse0xb0b0830() {
   return (neuron0xb0af9d0()*0.0919073);
}

double rrNNEB::synapse0xb0b0870() {
   return (neuron0xb0afd10()*-0.750544);
}

double rrNNEB::synapse0xb0b08b0() {
   return (neuron0xb0b0050()*-0.880366);
}

double rrNNEB::synapse0xb0b0c30() {
   return (neuron0xb0ae990()*1.36946);
}

double rrNNEB::synapse0xb0b0c70() {
   return (neuron0xb0aecd0()*2.02988);
}

double rrNNEB::synapse0xb0b0cb0() {
   return (neuron0xb0af010()*0.455045);
}

double rrNNEB::synapse0xb0b0cf0() {
   return (neuron0xb0af350()*0.449721);
}

double rrNNEB::synapse0xb0b0d30() {
   return (neuron0xb0af690()*-2.14025);
}

double rrNNEB::synapse0xb0b0d70() {
   return (neuron0xb0af9d0()*-1.43845);
}

double rrNNEB::synapse0xb0b0db0() {
   return (neuron0xb0afd10()*-0.384525);
}

double rrNNEB::synapse0xb0b0df0() {
   return (neuron0xb0b0050()*-0.370368);
}

double rrNNEB::synapse0xb0b1170() {
   return (neuron0xb0ae990()*-1.38509);
}

double rrNNEB::synapse0xa90e050() {
   return (neuron0xb0aecd0()*0.276106);
}

double rrNNEB::synapse0xa90e090() {
   return (neuron0xb0af010()*-0.356619);
}

double rrNNEB::synapse0xb0b12c0() {
   return (neuron0xb0af350()*-0.0844072);
}

double rrNNEB::synapse0xb0b1300() {
   return (neuron0xb0af690()*0.207364);
}

double rrNNEB::synapse0xb0b1340() {
   return (neuron0xb0af9d0()*0.672194);
}

double rrNNEB::synapse0xb0b1380() {
   return (neuron0xb0afd10()*0.320889);
}

double rrNNEB::synapse0xb0b13c0() {
   return (neuron0xb0b0050()*0.134055);
}

double rrNNEB::synapse0xb0b1740() {
   return (neuron0xb0ae990()*0.127745);
}

double rrNNEB::synapse0xb0b1780() {
   return (neuron0xb0aecd0()*-0.324372);
}

double rrNNEB::synapse0xb0b17c0() {
   return (neuron0xb0af010()*0.227847);
}

double rrNNEB::synapse0xb0b1800() {
   return (neuron0xb0af350()*0.307894);
}

double rrNNEB::synapse0xb0b1840() {
   return (neuron0xb0af690()*-0.718459);
}

double rrNNEB::synapse0xb0b1880() {
   return (neuron0xb0af9d0()*0.513562);
}

double rrNNEB::synapse0xb0b18c0() {
   return (neuron0xb0afd10()*0.140791);
}

double rrNNEB::synapse0xb0b1900() {
   return (neuron0xb0b0050()*0.177697);
}

double rrNNEB::synapse0xb0b1c80() {
   return (neuron0xb0ae990()*-0.825028);
}

double rrNNEB::synapse0xb0ae8c0() {
   return (neuron0xb0aecd0()*-0.543806);
}

double rrNNEB::synapse0xb0b75d0() {
   return (neuron0xb0af010()*1.0514);
}

double rrNNEB::synapse0xa90cb90() {
   return (neuron0xb0af350()*0.223959);
}

double rrNNEB::synapse0xb0b11b0() {
   return (neuron0xb0af690()*-1.47321);
}

double rrNNEB::synapse0xb0b11f0() {
   return (neuron0xb0af9d0()*1.05252);
}

double rrNNEB::synapse0xb0b1230() {
   return (neuron0xb0afd10()*0.195496);
}

double rrNNEB::synapse0xb0b1270() {
   return (neuron0xb0b0050()*0.184128);
}

double rrNNEB::synapse0xb0b2000() {
   return (neuron0xb0ae990()*0.720829);
}

double rrNNEB::synapse0xb0b2040() {
   return (neuron0xb0aecd0()*0.724649);
}

double rrNNEB::synapse0xb0b2080() {
   return (neuron0xb0af010()*0.995317);
}

double rrNNEB::synapse0xb0b20c0() {
   return (neuron0xb0af350()*1.3153);
}

double rrNNEB::synapse0xb0b2100() {
   return (neuron0xb0af690()*-0.703115);
}

double rrNNEB::synapse0xb0b2140() {
   return (neuron0xb0af9d0()*-1.02371);
}

double rrNNEB::synapse0xb0b2180() {
   return (neuron0xb0afd10()*0.102528);
}

double rrNNEB::synapse0xb0b21c0() {
   return (neuron0xb0b0050()*0.012124);
}

double rrNNEB::synapse0xb0b2540() {
   return (neuron0xb0ae990()*-0.20377);
}

double rrNNEB::synapse0xb0b2580() {
   return (neuron0xb0aecd0()*0.174654);
}

double rrNNEB::synapse0xb0b25c0() {
   return (neuron0xb0af010()*0.00425452);
}

double rrNNEB::synapse0xb0b2600() {
   return (neuron0xb0af350()*-0.153449);
}

double rrNNEB::synapse0xb0b2640() {
   return (neuron0xb0af690()*0.859321);
}

double rrNNEB::synapse0xb0b2680() {
   return (neuron0xb0af9d0()*0.878773);
}

double rrNNEB::synapse0xb0b26c0() {
   return (neuron0xb0afd10()*-0.393275);
}

double rrNNEB::synapse0xb0b2700() {
   return (neuron0xb0b0050()*-0.673188);
}

double rrNNEB::synapse0xb0b2a80() {
   return (neuron0xb0ae990()*0.541441);
}

double rrNNEB::synapse0xb0b2ac0() {
   return (neuron0xb0aecd0()*0.265352);
}

double rrNNEB::synapse0xb0b2b00() {
   return (neuron0xb0af010()*-1.66637);
}

double rrNNEB::synapse0xb0b2b40() {
   return (neuron0xb0af350()*-0.211802);
}

double rrNNEB::synapse0xb0b2b80() {
   return (neuron0xb0af690()*0.570811);
}

double rrNNEB::synapse0xb0b2bc0() {
   return (neuron0xb0af9d0()*1.19751);
}

double rrNNEB::synapse0xb0b2c00() {
   return (neuron0xb0afd10()*-0.280904);
}

double rrNNEB::synapse0xb0b2c40() {
   return (neuron0xb0b0050()*-0.12274);
}

double rrNNEB::synapse0xa7fbfb0() {
   return (neuron0xb0ae990()*4.05102);
}

double rrNNEB::synapse0xa7fbff0() {
   return (neuron0xb0aecd0()*-0.875368);
}

double rrNNEB::synapse0xa927960() {
   return (neuron0xb0af010()*1.21106);
}

double rrNNEB::synapse0xa9279a0() {
   return (neuron0xb0af350()*0.0400104);
}

double rrNNEB::synapse0xa9279e0() {
   return (neuron0xb0af690()*-2.51356);
}

double rrNNEB::synapse0xa927a20() {
   return (neuron0xb0af9d0()*-1.47699);
}

double rrNNEB::synapse0xa927a60() {
   return (neuron0xb0afd10()*-0.787877);
}

double rrNNEB::synapse0xa927aa0() {
   return (neuron0xb0b0050()*-0.402396);
}

double rrNNEB::synapse0xb0b3790() {
   return (neuron0xb0ae990()*-0.0941635);
}

double rrNNEB::synapse0xb0b37d0() {
   return (neuron0xb0aecd0()*0.15048);
}

double rrNNEB::synapse0xb0b3810() {
   return (neuron0xb0af010()*1.18917);
}

double rrNNEB::synapse0xb0b3850() {
   return (neuron0xb0af350()*-1.94427);
}

double rrNNEB::synapse0xb0b3890() {
   return (neuron0xb0af690()*-0.783299);
}

double rrNNEB::synapse0xb0b38d0() {
   return (neuron0xb0af9d0()*1.74181);
}

double rrNNEB::synapse0xb0b3910() {
   return (neuron0xb0afd10()*0.0131229);
}

double rrNNEB::synapse0xb0b3950() {
   return (neuron0xb0b0050()*-0.212452);
}

double rrNNEB::synapse0xb0b3cd0() {
   return (neuron0xb0b04c0()*-1.31458);
}

double rrNNEB::synapse0xb0b3d10() {
   return (neuron0xb0b08f0()*-1.47201);
}

double rrNNEB::synapse0xb0b3d50() {
   return (neuron0xb0b0e30()*-2.1513);
}

double rrNNEB::synapse0xb0b3d90() {
   return (neuron0xb0b1400()*0.611078);
}

double rrNNEB::synapse0xb0b3dd0() {
   return (neuron0xb0b1940()*-0.932359);
}

double rrNNEB::synapse0xb0b3e10() {
   return (neuron0xb0b1cc0()*-0.271423);
}

double rrNNEB::synapse0xb0b3e50() {
   return (neuron0xb0b2200()*1.84772);
}

double rrNNEB::synapse0xb0b3e90() {
   return (neuron0xb0b2740()*0.0107637);
}

double rrNNEB::synapse0xb0b3ed0() {
   return (neuron0xb0b2c80()*0.161287);
}

double rrNNEB::synapse0xb0b3f10() {
   return (neuron0xb0b34e0()*0.544009);
}

double rrNNEB::synapse0xb0b4290() {
   return (neuron0xb0b04c0()*-1.58107);
}

double rrNNEB::synapse0xb0b42d0() {
   return (neuron0xb0b08f0()*-0.50392);
}

double rrNNEB::synapse0xb0b4310() {
   return (neuron0xb0b0e30()*-0.850074);
}

double rrNNEB::synapse0xb0b4350() {
   return (neuron0xb0b1400()*1.62414);
}

double rrNNEB::synapse0xb0b4390() {
   return (neuron0xb0b1940()*0.379895);
}

double rrNNEB::synapse0xb0b43d0() {
   return (neuron0xb0b1cc0()*-0.421928);
}

double rrNNEB::synapse0xb0b4410() {
   return (neuron0xb0b2200()*1.94032);
}

double rrNNEB::synapse0xb0b4450() {
   return (neuron0xb0b2740()*0.40022);
}

double rrNNEB::synapse0xb0b4490() {
   return (neuron0xb0b2c80()*-0.381851);
}

double rrNNEB::synapse0xb0b44d0() {
   return (neuron0xb0b34e0()*-0.213928);
}

double rrNNEB::synapse0xb0b4850() {
   return (neuron0xb0b04c0()*-0.875178);
}

double rrNNEB::synapse0xb0b4890() {
   return (neuron0xb0b08f0()*0.763808);
}

double rrNNEB::synapse0xb0b48d0() {
   return (neuron0xb0b0e30()*-2.06516);
}

double rrNNEB::synapse0xb0b4910() {
   return (neuron0xb0b1400()*2.18368);
}

double rrNNEB::synapse0xb0b4950() {
   return (neuron0xb0b1940()*-0.676228);
}

double rrNNEB::synapse0xb0b4990() {
   return (neuron0xb0b1cc0()*0.936987);
}

double rrNNEB::synapse0xb0b49d0() {
   return (neuron0xb0b2200()*1.28271);
}

double rrNNEB::synapse0xb0b4a10() {
   return (neuron0xb0b2740()*0.279625);
}

double rrNNEB::synapse0xb0b4a50() {
   return (neuron0xb0b2c80()*0.277457);
}

double rrNNEB::synapse0xb0b4a90() {
   return (neuron0xb0b34e0()*0.203106);
}

double rrNNEB::synapse0xb0b4e10() {
   return (neuron0xb0b04c0()*-0.448124);
}

double rrNNEB::synapse0xb0b4e50() {
   return (neuron0xb0b08f0()*-0.367835);
}

double rrNNEB::synapse0xb0b4e90() {
   return (neuron0xb0b0e30()*-0.670159);
}

double rrNNEB::synapse0xb0b4ed0() {
   return (neuron0xb0b1400()*0.927953);
}

double rrNNEB::synapse0xb0b4f10() {
   return (neuron0xb0b1940()*-0.04733);
}

double rrNNEB::synapse0xb0b4f50() {
   return (neuron0xb0b1cc0()*-0.508311);
}

double rrNNEB::synapse0xb0b4f90() {
   return (neuron0xb0b2200()*0.804349);
}

double rrNNEB::synapse0xb0b4fd0() {
   return (neuron0xb0b2740()*0.480625);
}

double rrNNEB::synapse0xb0b5010() {
   return (neuron0xb0b2c80()*0.187565);
}

double rrNNEB::synapse0xb0b5050() {
   return (neuron0xb0b34e0()*-0.588078);
}

double rrNNEB::synapse0xb0b53d0() {
   return (neuron0xb0b04c0()*-1.98538);
}

double rrNNEB::synapse0xb0b5410() {
   return (neuron0xb0b08f0()*-2.25635);
}

double rrNNEB::synapse0xb0b5450() {
   return (neuron0xb0b0e30()*0.363142);
}

double rrNNEB::synapse0xb0b5490() {
   return (neuron0xb0b1400()*0.689364);
}

double rrNNEB::synapse0xb0b54d0() {
   return (neuron0xb0b1940()*-2.41042);
}

double rrNNEB::synapse0xb0b5510() {
   return (neuron0xb0b1cc0()*-0.94643);
}

double rrNNEB::synapse0xb0b5550() {
   return (neuron0xb0b2200()*-0.166018);
}

double rrNNEB::synapse0xb0b5590() {
   return (neuron0xb0b2740()*1.13809);
}

double rrNNEB::synapse0xb0b55d0() {
   return (neuron0xb0b2c80()*-2.84646);
}

double rrNNEB::synapse0xb0b30d0() {
   return (neuron0xb0b34e0()*2.69381);
}

double rrNNEB::synapse0xb0b3450() {
   return (neuron0xb0b3990()*2.7719);
}

double rrNNEB::synapse0xb0b3490() {
   return (neuron0xb0b3f50()*3.24816);
}

double rrNNEB::synapse0xb0b0390() {
   return (neuron0xb0b4510()*4.16451);
}

double rrNNEB::synapse0xb0b03d0() {
   return (neuron0xb0b4ad0()*1.17779);
}

double rrNNEB::synapse0xb0b0410() {
   return (neuron0xb0b5090()*3.92439);
}

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

