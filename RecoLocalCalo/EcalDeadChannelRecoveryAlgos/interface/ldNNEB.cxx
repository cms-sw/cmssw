#include "ldNNEB.h"
#include <cmath>

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

