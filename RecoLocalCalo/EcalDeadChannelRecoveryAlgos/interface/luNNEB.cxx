#include "luNNEB.h"
#include <cmath>

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

