#include "ldNNEE.h"
#include <cmath>

double ldNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.40353)/1.40003;
   input2 = (in2 - 2.41121)/1.41004;
   input3 = (in3 - 2.42657)/1.40629;
   input4 = (in4 - 2.42619)/1.40466;
   input5 = (in5 - 1.33856)/1.28698;
   input6 = (in6 - 1.33177)/1.28879;
   input7 = (in7 - 1.33367)/1.29347;
   switch(index) {
     case 0:
         return neuron0x1e95bcc0();
     default:
         return 0.;
   }
}

double ldNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.40353)/1.40003;
   input2 = (input[2] - 2.41121)/1.41004;
   input3 = (input[3] - 2.42657)/1.40629;
   input4 = (input[4] - 2.42619)/1.40466;
   input5 = (input[5] - 1.33856)/1.28698;
   input6 = (input[6] - 1.33177)/1.28879;
   input7 = (input[7] - 1.33367)/1.29347;
   switch(index) {
     case 0:
         return neuron0x1e95bcc0();
     default:
         return 0.;
   }
}

double ldNNEE::neuron0x1e957540() {
   return input0;
}

double ldNNEE::neuron0x1e957880() {
   return input1;
}

double ldNNEE::neuron0x1e957bc0() {
   return input2;
}

double ldNNEE::neuron0x1e957f00() {
   return input3;
}

double ldNNEE::neuron0x1e958240() {
   return input4;
}

double ldNNEE::neuron0x1e958580() {
   return input5;
}

double ldNNEE::neuron0x1e9588c0() {
   return input6;
}

double ldNNEE::neuron0x1e958c00() {
   return input7;
}

double ldNNEE::input0x1e959070() {
   double input = -1.56243;
   input += synapse0x1e8b7880();
   input += synapse0x1e960140();
   input += synapse0x1e959320();
   input += synapse0x1e959360();
   input += synapse0x1e9593a0();
   input += synapse0x1e9593e0();
   input += synapse0x1e959420();
   input += synapse0x1e959460();
   return input;
}

double ldNNEE::neuron0x1e959070() {
   double input = input0x1e959070();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e9594a0() {
   double input = 0.277395;
   input += synapse0x1e9597e0();
   input += synapse0x1e959820();
   input += synapse0x1e959860();
   input += synapse0x1e9598a0();
   input += synapse0x1e9598e0();
   input += synapse0x1e959920();
   input += synapse0x1e959960();
   input += synapse0x1e9599a0();
   return input;
}

double ldNNEE::neuron0x1e9594a0() {
   double input = input0x1e9594a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e9599e0() {
   double input = -2.09409;
   input += synapse0x1e959d20();
   input += synapse0x1e885e20();
   input += synapse0x1e885e60();
   input += synapse0x1e959e70();
   input += synapse0x1e959eb0();
   input += synapse0x1e959ef0();
   input += synapse0x1e959f30();
   input += synapse0x1e959f70();
   return input;
}

double ldNNEE::neuron0x1e9599e0() {
   double input = input0x1e9599e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e959fb0() {
   double input = -0.665299;
   input += synapse0x1e95a2f0();
   input += synapse0x1e95a330();
   input += synapse0x1e95a370();
   input += synapse0x1e95a3b0();
   input += synapse0x1e95a3f0();
   input += synapse0x1e95a430();
   input += synapse0x1e95a470();
   input += synapse0x1e95a4b0();
   return input;
}

double ldNNEE::neuron0x1e959fb0() {
   double input = input0x1e959fb0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95a4f0() {
   double input = -0.112128;
   input += synapse0x1e95a830();
   input += synapse0x1e957470();
   input += synapse0x1e960180();
   input += synapse0x1e8a2190();
   input += synapse0x1e959d60();
   input += synapse0x1e959da0();
   input += synapse0x1e959de0();
   input += synapse0x1e959e20();
   return input;
}

double ldNNEE::neuron0x1e95a4f0() {
   double input = input0x1e95a4f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95a870() {
   double input = 0.0424555;
   input += synapse0x1e95abb0();
   input += synapse0x1e95abf0();
   input += synapse0x1e95ac30();
   input += synapse0x1e95ac70();
   input += synapse0x1e95acb0();
   input += synapse0x1e95acf0();
   input += synapse0x1e95ad30();
   input += synapse0x1e95ad70();
   return input;
}

double ldNNEE::neuron0x1e95a870() {
   double input = input0x1e95a870();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95adb0() {
   double input = 0.783657;
   input += synapse0x1e95b0f0();
   input += synapse0x1e95b130();
   input += synapse0x1e95b170();
   input += synapse0x1e95b1b0();
   input += synapse0x1e95b1f0();
   input += synapse0x1e95b230();
   input += synapse0x1e95b270();
   input += synapse0x1e95b2b0();
   return input;
}

double ldNNEE::neuron0x1e95adb0() {
   double input = input0x1e95adb0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95b2f0() {
   double input = -0.327327;
   input += synapse0x1e95b630();
   input += synapse0x1e95b670();
   input += synapse0x1e95b6b0();
   input += synapse0x1e95b6f0();
   input += synapse0x1e95b730();
   input += synapse0x1e95b770();
   input += synapse0x1e95b7b0();
   input += synapse0x1e95b7f0();
   return input;
}

double ldNNEE::neuron0x1e95b2f0() {
   double input = input0x1e95b2f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95b830() {
   double input = 2.23142;
   input += synapse0x1e883a80();
   input += synapse0x1e883ac0();
   input += synapse0x1e89e980();
   input += synapse0x1e89e9c0();
   input += synapse0x1e89ea00();
   input += synapse0x1e89ea40();
   input += synapse0x1e89ea80();
   input += synapse0x1e89eac0();
   return input;
}

double ldNNEE::neuron0x1e95b830() {
   double input = input0x1e95b830();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95c090() {
   double input = 2.37996;
   input += synapse0x1e95c340();
   input += synapse0x1e95c380();
   input += synapse0x1e95c3c0();
   input += synapse0x1e95c400();
   input += synapse0x1e95c440();
   input += synapse0x1e95c480();
   input += synapse0x1e95c4c0();
   input += synapse0x1e95c500();
   return input;
}

double ldNNEE::neuron0x1e95c090() {
   double input = input0x1e95c090();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95c540() {
   double input = -0.959941;
   input += synapse0x1e95c880();
   input += synapse0x1e95c8c0();
   input += synapse0x1e95c900();
   input += synapse0x1e95c940();
   input += synapse0x1e95c980();
   input += synapse0x1e95c9c0();
   input += synapse0x1e95ca00();
   input += synapse0x1e95ca40();
   input += synapse0x1e95ca80();
   input += synapse0x1e95cac0();
   return input;
}

double ldNNEE::neuron0x1e95c540() {
   double input = input0x1e95c540();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95cb00() {
   double input = -0.0545535;
   input += synapse0x1e95ce40();
   input += synapse0x1e95ce80();
   input += synapse0x1e95cec0();
   input += synapse0x1e95cf00();
   input += synapse0x1e95cf40();
   input += synapse0x1e95cf80();
   input += synapse0x1e95cfc0();
   input += synapse0x1e95d000();
   input += synapse0x1e95d040();
   input += synapse0x1e95d080();
   return input;
}

double ldNNEE::neuron0x1e95cb00() {
   double input = input0x1e95cb00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95d0c0() {
   double input = 0.108088;
   input += synapse0x1e95d400();
   input += synapse0x1e95d440();
   input += synapse0x1e95d480();
   input += synapse0x1e95d4c0();
   input += synapse0x1e95d500();
   input += synapse0x1e95d540();
   input += synapse0x1e95d580();
   input += synapse0x1e95d5c0();
   input += synapse0x1e95d600();
   input += synapse0x1e95d640();
   return input;
}

double ldNNEE::neuron0x1e95d0c0() {
   double input = input0x1e95d0c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95d680() {
   double input = -0.0100983;
   input += synapse0x1e95d9c0();
   input += synapse0x1e95da00();
   input += synapse0x1e95da40();
   input += synapse0x1e95da80();
   input += synapse0x1e95dac0();
   input += synapse0x1e95db00();
   input += synapse0x1e95db40();
   input += synapse0x1e95db80();
   input += synapse0x1e95dbc0();
   input += synapse0x1e95dc00();
   return input;
}

double ldNNEE::neuron0x1e95d680() {
   double input = input0x1e95d680();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95dc40() {
   double input = -0.534084;
   input += synapse0x1e95df80();
   input += synapse0x1e95dfc0();
   input += synapse0x1e95e000();
   input += synapse0x1e95e040();
   input += synapse0x1e95e080();
   input += synapse0x1e95e0c0();
   input += synapse0x1e95e100();
   input += synapse0x1e95e140();
   input += synapse0x1e95e180();
   input += synapse0x1e95bc80();
   return input;
}

double ldNNEE::neuron0x1e95dc40() {
   double input = input0x1e95dc40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95bcc0() {
   double input = -1.46932;
   input += synapse0x1e95c000();
   input += synapse0x1e95c040();
   input += synapse0x1e958f40();
   input += synapse0x1e958f80();
   input += synapse0x1e958fc0();
   return input;
}

double ldNNEE::neuron0x1e95bcc0() {
   double input = input0x1e95bcc0();
   return (input * 1)+0;
}

double ldNNEE::synapse0x1e8b7880() {
   return (neuron0x1e957540()*1.46564);
}

double ldNNEE::synapse0x1e960140() {
   return (neuron0x1e957880()*0.428584);
}

double ldNNEE::synapse0x1e959320() {
   return (neuron0x1e957bc0()*0.582542);
}

double ldNNEE::synapse0x1e959360() {
   return (neuron0x1e957f00()*-0.787778);
}

double ldNNEE::synapse0x1e9593a0() {
   return (neuron0x1e958240()*-1.76293);
}

double ldNNEE::synapse0x1e9593e0() {
   return (neuron0x1e958580()*0.173838);
}

double ldNNEE::synapse0x1e959420() {
   return (neuron0x1e9588c0()*-0.902033);
}

double ldNNEE::synapse0x1e959460() {
   return (neuron0x1e958c00()*-0.089348);
}

double ldNNEE::synapse0x1e9597e0() {
   return (neuron0x1e957540()*1.41247);
}

double ldNNEE::synapse0x1e959820() {
   return (neuron0x1e957880()*-0.042349);
}

double ldNNEE::synapse0x1e959860() {
   return (neuron0x1e957bc0()*-0.7785);
}

double ldNNEE::synapse0x1e9598a0() {
   return (neuron0x1e957f00()*-0.210301);
}

double ldNNEE::synapse0x1e9598e0() {
   return (neuron0x1e958240()*-1.17315);
}

double ldNNEE::synapse0x1e959920() {
   return (neuron0x1e958580()*-0.195878);
}

double ldNNEE::synapse0x1e959960() {
   return (neuron0x1e9588c0()*1.3596);
}

double ldNNEE::synapse0x1e9599a0() {
   return (neuron0x1e958c00()*1.28309);
}

double ldNNEE::synapse0x1e959d20() {
   return (neuron0x1e957540()*1.15124);
}

double ldNNEE::synapse0x1e885e20() {
   return (neuron0x1e957880()*-0.0668577);
}

double ldNNEE::synapse0x1e885e60() {
   return (neuron0x1e957bc0()*0.408767);
}

double ldNNEE::synapse0x1e959e70() {
   return (neuron0x1e957f00()*-0.383822);
}

double ldNNEE::synapse0x1e959eb0() {
   return (neuron0x1e958240()*-0.350844);
}

double ldNNEE::synapse0x1e959ef0() {
   return (neuron0x1e958580()*-0.658633);
}

double ldNNEE::synapse0x1e959f30() {
   return (neuron0x1e9588c0()*0.253153);
}

double ldNNEE::synapse0x1e959f70() {
   return (neuron0x1e958c00()*0.450355);
}

double ldNNEE::synapse0x1e95a2f0() {
   return (neuron0x1e957540()*0.533386);
}

double ldNNEE::synapse0x1e95a330() {
   return (neuron0x1e957880()*-0.0825323);
}

double ldNNEE::synapse0x1e95a370() {
   return (neuron0x1e957bc0()*-0.690911);
}

double ldNNEE::synapse0x1e95a3b0() {
   return (neuron0x1e957f00()*0.216972);
}

double ldNNEE::synapse0x1e95a3f0() {
   return (neuron0x1e958240()*0.753513);
}

double ldNNEE::synapse0x1e95a430() {
   return (neuron0x1e958580()*0.0971258);
}

double ldNNEE::synapse0x1e95a470() {
   return (neuron0x1e9588c0()*0.355891);
}

double ldNNEE::synapse0x1e95a4b0() {
   return (neuron0x1e958c00()*0.62749);
}

double ldNNEE::synapse0x1e95a830() {
   return (neuron0x1e957540()*2.40699);
}

double ldNNEE::synapse0x1e957470() {
   return (neuron0x1e957880()*-1.59664);
}

double ldNNEE::synapse0x1e960180() {
   return (neuron0x1e957bc0()*-0.865839);
}

double ldNNEE::synapse0x1e8a2190() {
   return (neuron0x1e957f00()*-1.03403);
}

double ldNNEE::synapse0x1e959d60() {
   return (neuron0x1e958240()*-0.0327801);
}

double ldNNEE::synapse0x1e959da0() {
   return (neuron0x1e958580()*-0.883803);
}

double ldNNEE::synapse0x1e959de0() {
   return (neuron0x1e9588c0()*0.919519);
}

double ldNNEE::synapse0x1e959e20() {
   return (neuron0x1e958c00()*1.463);
}

double ldNNEE::synapse0x1e95abb0() {
   return (neuron0x1e957540()*-0.0746568);
}

double ldNNEE::synapse0x1e95abf0() {
   return (neuron0x1e957880()*-0.332818);
}

double ldNNEE::synapse0x1e95ac30() {
   return (neuron0x1e957bc0()*-1.35063);
}

double ldNNEE::synapse0x1e95ac70() {
   return (neuron0x1e957f00()*0.183133);
}

double ldNNEE::synapse0x1e95acb0() {
   return (neuron0x1e958240()*0.134358);
}

double ldNNEE::synapse0x1e95acf0() {
   return (neuron0x1e958580()*-0.491049);
}

double ldNNEE::synapse0x1e95ad30() {
   return (neuron0x1e9588c0()*1.64453);
}

double ldNNEE::synapse0x1e95ad70() {
   return (neuron0x1e958c00()*0.67541);
}

double ldNNEE::synapse0x1e95b0f0() {
   return (neuron0x1e957540()*0.434697);
}

double ldNNEE::synapse0x1e95b130() {
   return (neuron0x1e957880()*-0.360692);
}

double ldNNEE::synapse0x1e95b170() {
   return (neuron0x1e957bc0()*0.217958);
}

double ldNNEE::synapse0x1e95b1b0() {
   return (neuron0x1e957f00()*0.0690149);
}

double ldNNEE::synapse0x1e95b1f0() {
   return (neuron0x1e958240()*1.07911);
}

double ldNNEE::synapse0x1e95b230() {
   return (neuron0x1e958580()*0.461768);
}

double ldNNEE::synapse0x1e95b270() {
   return (neuron0x1e9588c0()*-0.778333);
}

double ldNNEE::synapse0x1e95b2b0() {
   return (neuron0x1e958c00()*-1.37221);
}

double ldNNEE::synapse0x1e95b630() {
   return (neuron0x1e957540()*-0.885185);
}

double ldNNEE::synapse0x1e95b670() {
   return (neuron0x1e957880()*-0.146735);
}

double ldNNEE::synapse0x1e95b6b0() {
   return (neuron0x1e957bc0()*1.0352);
}

double ldNNEE::synapse0x1e95b6f0() {
   return (neuron0x1e957f00()*0.00272632);
}

double ldNNEE::synapse0x1e95b730() {
   return (neuron0x1e958240()*0.761924);
}

double ldNNEE::synapse0x1e95b770() {
   return (neuron0x1e958580()*0.0672013);
}

double ldNNEE::synapse0x1e95b7b0() {
   return (neuron0x1e9588c0()*-0.640816);
}

double ldNNEE::synapse0x1e95b7f0() {
   return (neuron0x1e958c00()*-0.691704);
}

double ldNNEE::synapse0x1e883a80() {
   return (neuron0x1e957540()*-0.384238);
}

double ldNNEE::synapse0x1e883ac0() {
   return (neuron0x1e957880()*0.397007);
}

double ldNNEE::synapse0x1e89e980() {
   return (neuron0x1e957bc0()*-0.39839);
}

double ldNNEE::synapse0x1e89e9c0() {
   return (neuron0x1e957f00()*-0.00745285);
}

double ldNNEE::synapse0x1e89ea00() {
   return (neuron0x1e958240()*-0.413446);
}

double ldNNEE::synapse0x1e89ea40() {
   return (neuron0x1e958580()*-0.243978);
}

double ldNNEE::synapse0x1e89ea80() {
   return (neuron0x1e9588c0()*-0.500086);
}

double ldNNEE::synapse0x1e89eac0() {
   return (neuron0x1e958c00()*-0.178044);
}

double ldNNEE::synapse0x1e95c340() {
   return (neuron0x1e957540()*-0.323586);
}

double ldNNEE::synapse0x1e95c380() {
   return (neuron0x1e957880()*0.501998);
}

double ldNNEE::synapse0x1e95c3c0() {
   return (neuron0x1e957bc0()*1.51388);
}

double ldNNEE::synapse0x1e95c400() {
   return (neuron0x1e957f00()*-0.879529);
}

double ldNNEE::synapse0x1e95c440() {
   return (neuron0x1e958240()*-0.805979);
}

double ldNNEE::synapse0x1e95c480() {
   return (neuron0x1e958580()*-0.461876);
}

double ldNNEE::synapse0x1e95c4c0() {
   return (neuron0x1e9588c0()*0.0814574);
}

double ldNNEE::synapse0x1e95c500() {
   return (neuron0x1e958c00()*0.690482);
}

double ldNNEE::synapse0x1e95c880() {
   return (neuron0x1e959070()*-0.92281);
}

double ldNNEE::synapse0x1e95c8c0() {
   return (neuron0x1e9594a0()*-1.65121);
}

double ldNNEE::synapse0x1e95c900() {
   return (neuron0x1e9599e0()*1.7076);
}

double ldNNEE::synapse0x1e95c940() {
   return (neuron0x1e959fb0()*0.799588);
}

double ldNNEE::synapse0x1e95c980() {
   return (neuron0x1e95a4f0()*-1.21932);
}

double ldNNEE::synapse0x1e95c9c0() {
   return (neuron0x1e95a870()*-0.265466);
}

double ldNNEE::synapse0x1e95ca00() {
   return (neuron0x1e95adb0()*1.20421);
}

double ldNNEE::synapse0x1e95ca40() {
   return (neuron0x1e95b2f0()*0.585932);
}

double ldNNEE::synapse0x1e95ca80() {
   return (neuron0x1e95b830()*-1.8946);
}

double ldNNEE::synapse0x1e95cac0() {
   return (neuron0x1e95c090()*1.08486);
}

double ldNNEE::synapse0x1e95ce40() {
   return (neuron0x1e959070()*0.764949);
}

double ldNNEE::synapse0x1e95ce80() {
   return (neuron0x1e9594a0()*-1.77857);
}

double ldNNEE::synapse0x1e95cec0() {
   return (neuron0x1e9599e0()*-1.44957);
}

double ldNNEE::synapse0x1e95cf00() {
   return (neuron0x1e959fb0()*-1.41676);
}

double ldNNEE::synapse0x1e95cf40() {
   return (neuron0x1e95a4f0()*-0.458544);
}

double ldNNEE::synapse0x1e95cf80() {
   return (neuron0x1e95a870()*0.528898);
}

double ldNNEE::synapse0x1e95cfc0() {
   return (neuron0x1e95adb0()*-0.750037);
}

double ldNNEE::synapse0x1e95d000() {
   return (neuron0x1e95b2f0()*-0.764525);
}

double ldNNEE::synapse0x1e95d040() {
   return (neuron0x1e95b830()*0.371361);
}

double ldNNEE::synapse0x1e95d080() {
   return (neuron0x1e95c090()*-0.399181);
}

double ldNNEE::synapse0x1e95d400() {
   return (neuron0x1e959070()*-0.376096);
}

double ldNNEE::synapse0x1e95d440() {
   return (neuron0x1e9594a0()*0.137658);
}

double ldNNEE::synapse0x1e95d480() {
   return (neuron0x1e9599e0()*0.298105);
}

double ldNNEE::synapse0x1e95d4c0() {
   return (neuron0x1e959fb0()*0.105747);
}

double ldNNEE::synapse0x1e95d500() {
   return (neuron0x1e95a4f0()*-0.738776);
}

double ldNNEE::synapse0x1e95d540() {
   return (neuron0x1e95a870()*-0.216621);
}

double ldNNEE::synapse0x1e95d580() {
   return (neuron0x1e95adb0()*0.522074);
}

double ldNNEE::synapse0x1e95d5c0() {
   return (neuron0x1e95b2f0()*-0.133003);
}

double ldNNEE::synapse0x1e95d600() {
   return (neuron0x1e95b830()*-0.116397);
}

double ldNNEE::synapse0x1e95d640() {
   return (neuron0x1e95c090()*0.643543);
}

double ldNNEE::synapse0x1e95d9c0() {
   return (neuron0x1e959070()*-0.684852);
}

double ldNNEE::synapse0x1e95da00() {
   return (neuron0x1e9594a0()*0.556556);
}

double ldNNEE::synapse0x1e95da40() {
   return (neuron0x1e9599e0()*1.16285);
}

double ldNNEE::synapse0x1e95da80() {
   return (neuron0x1e959fb0()*1.13189);
}

double ldNNEE::synapse0x1e95dac0() {
   return (neuron0x1e95a4f0()*-1.5021);
}

double ldNNEE::synapse0x1e95db00() {
   return (neuron0x1e95a870()*-0.509703);
}

double ldNNEE::synapse0x1e95db40() {
   return (neuron0x1e95adb0()*0.980433);
}

double ldNNEE::synapse0x1e95db80() {
   return (neuron0x1e95b2f0()*0.240291);
}

double ldNNEE::synapse0x1e95dbc0() {
   return (neuron0x1e95b830()*-1.16458);
}

double ldNNEE::synapse0x1e95dc00() {
   return (neuron0x1e95c090()*0.598338);
}

double ldNNEE::synapse0x1e95df80() {
   return (neuron0x1e959070()*-1.05336);
}

double ldNNEE::synapse0x1e95dfc0() {
   return (neuron0x1e9594a0()*0.897677);
}

double ldNNEE::synapse0x1e95e000() {
   return (neuron0x1e9599e0()*2.32913);
}

double ldNNEE::synapse0x1e95e040() {
   return (neuron0x1e959fb0()*0.837162);
}

double ldNNEE::synapse0x1e95e080() {
   return (neuron0x1e95a4f0()*0.0385715);
}

double ldNNEE::synapse0x1e95e0c0() {
   return (neuron0x1e95a870()*-1.22258);
}

double ldNNEE::synapse0x1e95e100() {
   return (neuron0x1e95adb0()*0.946005);
}

double ldNNEE::synapse0x1e95e140() {
   return (neuron0x1e95b2f0()*1.22504);
}

double ldNNEE::synapse0x1e95e180() {
   return (neuron0x1e95b830()*-1.85501);
}

double ldNNEE::synapse0x1e95bc80() {
   return (neuron0x1e95c090()*1.77983);
}

double ldNNEE::synapse0x1e95c000() {
   return (neuron0x1e95c540()*4.10013);
}

double ldNNEE::synapse0x1e95c040() {
   return (neuron0x1e95cb00()*-3.6466);
}

double ldNNEE::synapse0x1e958f40() {
   return (neuron0x1e95d0c0()*-0.333399);
}

double ldNNEE::synapse0x1e958f80() {
   return (neuron0x1e95d680()*1.42134);
}

double ldNNEE::synapse0x1e958fc0() {
   return (neuron0x1e95dc40()*3.14812);
}

