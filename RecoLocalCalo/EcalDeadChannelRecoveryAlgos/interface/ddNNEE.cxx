#include "ddNNEE.h"
#include <cmath>

double ddNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.40353)/1.40003;
   input2 = (in2 - 2.41121)/1.41004;
   input3 = (in3 - 2.42657)/1.40629;
   input4 = (in4 - 1.33856)/1.28698;
   input5 = (in5 - 1.33177)/1.28879;
   input6 = (in6 - 1.33367)/1.29347;
   input7 = (in7 - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x8316d00();
     default:
         return 0.;
   }
}

double ddNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.40353)/1.40003;
   input2 = (input[2] - 2.41121)/1.41004;
   input3 = (input[3] - 2.42657)/1.40629;
   input4 = (input[4] - 1.33856)/1.28698;
   input5 = (input[5] - 1.33177)/1.28879;
   input6 = (input[6] - 1.33367)/1.29347;
   input7 = (input[7] - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x8316d00();
     default:
         return 0.;
   }
}

double ddNNEE::neuron0x8312580() {
   return input0;
}

double ddNNEE::neuron0x83128c0() {
   return input1;
}

double ddNNEE::neuron0x8312c00() {
   return input2;
}

double ddNNEE::neuron0x8312f40() {
   return input3;
}

double ddNNEE::neuron0x8313280() {
   return input4;
}

double ddNNEE::neuron0x83135c0() {
   return input5;
}

double ddNNEE::neuron0x8313900() {
   return input6;
}

double ddNNEE::neuron0x8313c40() {
   return input7;
}

double ddNNEE::input0x83140b0() {
   double input = 1.05719;
   input += synapse0x82728c0();
   input += synapse0x831b180();
   input += synapse0x8314360();
   input += synapse0x83143a0();
   input += synapse0x83143e0();
   input += synapse0x8314420();
   input += synapse0x8314460();
   input += synapse0x83144a0();
   return input;
}

double ddNNEE::neuron0x83140b0() {
   double input = input0x83140b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x83144e0() {
   double input = -1.37516;
   input += synapse0x8314820();
   input += synapse0x8314860();
   input += synapse0x83148a0();
   input += synapse0x83148e0();
   input += synapse0x8314920();
   input += synapse0x8314960();
   input += synapse0x83149a0();
   input += synapse0x83149e0();
   return input;
}

double ddNNEE::neuron0x83144e0() {
   double input = input0x83144e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8314a20() {
   double input = 0.753304;
   input += synapse0x8314d60();
   input += synapse0x8240e60();
   input += synapse0x8240ea0();
   input += synapse0x8314eb0();
   input += synapse0x8314ef0();
   input += synapse0x8314f30();
   input += synapse0x8314f70();
   input += synapse0x8314fb0();
   return input;
}

double ddNNEE::neuron0x8314a20() {
   double input = input0x8314a20();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8314ff0() {
   double input = -1.4179;
   input += synapse0x8315330();
   input += synapse0x8315370();
   input += synapse0x83153b0();
   input += synapse0x83153f0();
   input += synapse0x8315430();
   input += synapse0x8315470();
   input += synapse0x83154b0();
   input += synapse0x83154f0();
   return input;
}

double ddNNEE::neuron0x8314ff0() {
   double input = input0x8314ff0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8315530() {
   double input = 1.67886;
   input += synapse0x8315870();
   input += synapse0x83124b0();
   input += synapse0x831b1c0();
   input += synapse0x825d1d0();
   input += synapse0x8314da0();
   input += synapse0x8314de0();
   input += synapse0x8314e20();
   input += synapse0x8314e60();
   return input;
}

double ddNNEE::neuron0x8315530() {
   double input = input0x8315530();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x83158b0() {
   double input = 0.142783;
   input += synapse0x8315bf0();
   input += synapse0x8315c30();
   input += synapse0x8315c70();
   input += synapse0x8315cb0();
   input += synapse0x8315cf0();
   input += synapse0x8315d30();
   input += synapse0x8315d70();
   input += synapse0x8315db0();
   return input;
}

double ddNNEE::neuron0x83158b0() {
   double input = input0x83158b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8315df0() {
   double input = -1.49169;
   input += synapse0x8316130();
   input += synapse0x8316170();
   input += synapse0x83161b0();
   input += synapse0x83161f0();
   input += synapse0x8316230();
   input += synapse0x8316270();
   input += synapse0x83162b0();
   input += synapse0x83162f0();
   return input;
}

double ddNNEE::neuron0x8315df0() {
   double input = input0x8315df0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8316330() {
   double input = -0.0621605;
   input += synapse0x8316670();
   input += synapse0x83166b0();
   input += synapse0x83166f0();
   input += synapse0x8316730();
   input += synapse0x8316770();
   input += synapse0x83167b0();
   input += synapse0x83167f0();
   input += synapse0x8316830();
   return input;
}

double ddNNEE::neuron0x8316330() {
   double input = input0x8316330();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8316870() {
   double input = 0.517246;
   input += synapse0x823eac0();
   input += synapse0x823eb00();
   input += synapse0x82599c0();
   input += synapse0x8259a00();
   input += synapse0x8259a40();
   input += synapse0x8259a80();
   input += synapse0x8259ac0();
   input += synapse0x8259b00();
   return input;
}

double ddNNEE::neuron0x8316870() {
   double input = input0x8316870();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x83170d0() {
   double input = 0.63036;
   input += synapse0x8317380();
   input += synapse0x83173c0();
   input += synapse0x8317400();
   input += synapse0x8317440();
   input += synapse0x8317480();
   input += synapse0x83174c0();
   input += synapse0x8317500();
   input += synapse0x8317540();
   return input;
}

double ddNNEE::neuron0x83170d0() {
   double input = input0x83170d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8317580() {
   double input = 0.536046;
   input += synapse0x83178c0();
   input += synapse0x8317900();
   input += synapse0x8317940();
   input += synapse0x8317980();
   input += synapse0x83179c0();
   input += synapse0x8317a00();
   input += synapse0x8317a40();
   input += synapse0x8317a80();
   input += synapse0x8317ac0();
   input += synapse0x8317b00();
   return input;
}

double ddNNEE::neuron0x8317580() {
   double input = input0x8317580();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8317b40() {
   double input = 0.28105;
   input += synapse0x8317e80();
   input += synapse0x8317ec0();
   input += synapse0x8317f00();
   input += synapse0x8317f40();
   input += synapse0x8317f80();
   input += synapse0x8317fc0();
   input += synapse0x8318000();
   input += synapse0x8318040();
   input += synapse0x8318080();
   input += synapse0x83180c0();
   return input;
}

double ddNNEE::neuron0x8317b40() {
   double input = input0x8317b40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8318100() {
   double input = 0.253693;
   input += synapse0x8318440();
   input += synapse0x8318480();
   input += synapse0x83184c0();
   input += synapse0x8318500();
   input += synapse0x8318540();
   input += synapse0x8318580();
   input += synapse0x83185c0();
   input += synapse0x8318600();
   input += synapse0x8318640();
   input += synapse0x8318680();
   return input;
}

double ddNNEE::neuron0x8318100() {
   double input = input0x8318100();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x83186c0() {
   double input = 0.558697;
   input += synapse0x8318a00();
   input += synapse0x8318a40();
   input += synapse0x8318a80();
   input += synapse0x8318ac0();
   input += synapse0x8318b00();
   input += synapse0x8318b40();
   input += synapse0x8318b80();
   input += synapse0x8318bc0();
   input += synapse0x8318c00();
   input += synapse0x8318c40();
   return input;
}

double ddNNEE::neuron0x83186c0() {
   double input = input0x83186c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8318c80() {
   double input = -2.24623;
   input += synapse0x8318fc0();
   input += synapse0x8319000();
   input += synapse0x8319040();
   input += synapse0x8319080();
   input += synapse0x83190c0();
   input += synapse0x8319100();
   input += synapse0x8319140();
   input += synapse0x8319180();
   input += synapse0x83191c0();
   input += synapse0x8316cc0();
   return input;
}

double ddNNEE::neuron0x8318c80() {
   double input = input0x8318c80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ddNNEE::input0x8316d00() {
   double input = -2.15788;
   input += synapse0x8317040();
   input += synapse0x8317080();
   input += synapse0x8313f80();
   input += synapse0x8313fc0();
   input += synapse0x8314000();
   return input;
}

double ddNNEE::neuron0x8316d00() {
   double input = input0x8316d00();
   return (input * 1)+0;
}

double ddNNEE::synapse0x82728c0() {
   return (neuron0x8312580()*-1.58825);
}

double ddNNEE::synapse0x831b180() {
   return (neuron0x83128c0()*-1.10229);
}

double ddNNEE::synapse0x8314360() {
   return (neuron0x8312c00()*-0.565096);
}

double ddNNEE::synapse0x83143a0() {
   return (neuron0x8312f40()*0.122835);
}

double ddNNEE::synapse0x83143e0() {
   return (neuron0x8313280()*-0.105586);
}

double ddNNEE::synapse0x8314420() {
   return (neuron0x83135c0()*3.72349);
}

double ddNNEE::synapse0x8314460() {
   return (neuron0x8313900()*0.517663);
}

double ddNNEE::synapse0x83144a0() {
   return (neuron0x8313c40()*-0.0567961);
}

double ddNNEE::synapse0x8314820() {
   return (neuron0x8312580()*0.394706);
}

double ddNNEE::synapse0x8314860() {
   return (neuron0x83128c0()*0.342424);
}

double ddNNEE::synapse0x83148a0() {
   return (neuron0x8312c00()*0.436855);
}

double ddNNEE::synapse0x83148e0() {
   return (neuron0x8312f40()*-0.294598);
}

double ddNNEE::synapse0x8314920() {
   return (neuron0x8313280()*0.492127);
}

double ddNNEE::synapse0x8314960() {
   return (neuron0x83135c0()*0.00187448);
}

double ddNNEE::synapse0x83149a0() {
   return (neuron0x8313900()*-0.377788);
}

double ddNNEE::synapse0x83149e0() {
   return (neuron0x8313c40()*-0.349046);
}

double ddNNEE::synapse0x8314d60() {
   return (neuron0x8312580()*-0.197605);
}

double ddNNEE::synapse0x8240e60() {
   return (neuron0x83128c0()*-0.373985);
}

double ddNNEE::synapse0x8240ea0() {
   return (neuron0x8312c00()*-0.0505447);
}

double ddNNEE::synapse0x8314eb0() {
   return (neuron0x8312f40()*-0.246537);
}

double ddNNEE::synapse0x8314ef0() {
   return (neuron0x8313280()*0.140793);
}

double ddNNEE::synapse0x8314f30() {
   return (neuron0x83135c0()*0.411811);
}

double ddNNEE::synapse0x8314f70() {
   return (neuron0x8313900()*-0.209583);
}

double ddNNEE::synapse0x8314fb0() {
   return (neuron0x8313c40()*-0.453906);
}

double ddNNEE::synapse0x8315330() {
   return (neuron0x8312580()*1.64781);
}

double ddNNEE::synapse0x8315370() {
   return (neuron0x83128c0()*0.921551);
}

double ddNNEE::synapse0x83153b0() {
   return (neuron0x8312c00()*0.941424);
}

double ddNNEE::synapse0x83153f0() {
   return (neuron0x8312f40()*1.15033);
}

double ddNNEE::synapse0x8315430() {
   return (neuron0x8313280()*0.978414);
}

double ddNNEE::synapse0x8315470() {
   return (neuron0x83135c0()*1.03646);
}

double ddNNEE::synapse0x83154b0() {
   return (neuron0x8313900()*0.992893);
}

double ddNNEE::synapse0x83154f0() {
   return (neuron0x8313c40()*1.07173);
}

double ddNNEE::synapse0x8315870() {
   return (neuron0x8312580()*-0.487263);
}

double ddNNEE::synapse0x83124b0() {
   return (neuron0x83128c0()*-0.415371);
}

double ddNNEE::synapse0x831b1c0() {
   return (neuron0x8312c00()*-0.179726);
}

double ddNNEE::synapse0x825d1d0() {
   return (neuron0x8312f40()*1.23975);
}

double ddNNEE::synapse0x8314da0() {
   return (neuron0x8313280()*-0.218843);
}

double ddNNEE::synapse0x8314de0() {
   return (neuron0x83135c0()*0.239369);
}

double ddNNEE::synapse0x8314e20() {
   return (neuron0x8313900()*-0.584394);
}

double ddNNEE::synapse0x8314e60() {
   return (neuron0x8313c40()*0.198253);
}

double ddNNEE::synapse0x8315bf0() {
   return (neuron0x8312580()*0.740868);
}

double ddNNEE::synapse0x8315c30() {
   return (neuron0x83128c0()*-0.217136);
}

double ddNNEE::synapse0x8315c70() {
   return (neuron0x8312c00()*0.561292);
}

double ddNNEE::synapse0x8315cb0() {
   return (neuron0x8312f40()*0.114271);
}

double ddNNEE::synapse0x8315cf0() {
   return (neuron0x8313280()*0.381374);
}

double ddNNEE::synapse0x8315d30() {
   return (neuron0x83135c0()*-0.466769);
}

double ddNNEE::synapse0x8315d70() {
   return (neuron0x8313900()*0.405474);
}

double ddNNEE::synapse0x8315db0() {
   return (neuron0x8313c40()*-1.04062);
}

double ddNNEE::synapse0x8316130() {
   return (neuron0x8312580()*-0.402604);
}

double ddNNEE::synapse0x8316170() {
   return (neuron0x83128c0()*0.220286);
}

double ddNNEE::synapse0x83161b0() {
   return (neuron0x8312c00()*0.331519);
}

double ddNNEE::synapse0x83161f0() {
   return (neuron0x8312f40()*-0.646217);
}

double ddNNEE::synapse0x8316230() {
   return (neuron0x8313280()*0.198439);
}

double ddNNEE::synapse0x8316270() {
   return (neuron0x83135c0()*-0.70732);
}

double ddNNEE::synapse0x83162b0() {
   return (neuron0x8313900()*-0.584387);
}

double ddNNEE::synapse0x83162f0() {
   return (neuron0x8313c40()*2.29453);
}

double ddNNEE::synapse0x8316670() {
   return (neuron0x8312580()*-1.93044);
}

double ddNNEE::synapse0x83166b0() {
   return (neuron0x83128c0()*-2.45384);
}

double ddNNEE::synapse0x83166f0() {
   return (neuron0x8312c00()*-0.423168);
}

double ddNNEE::synapse0x8316730() {
   return (neuron0x8312f40()*0.126811);
}

double ddNNEE::synapse0x8316770() {
   return (neuron0x8313280()*0.46297);
}

double ddNNEE::synapse0x83167b0() {
   return (neuron0x83135c0()*2.33621);
}

double ddNNEE::synapse0x83167f0() {
   return (neuron0x8313900()*-0.0213368);
}

double ddNNEE::synapse0x8316830() {
   return (neuron0x8313c40()*1.38537);
}

double ddNNEE::synapse0x823eac0() {
   return (neuron0x8312580()*-0.53831);
}

double ddNNEE::synapse0x823eb00() {
   return (neuron0x83128c0()*0.329635);
}

double ddNNEE::synapse0x82599c0() {
   return (neuron0x8312c00()*-0.795528);
}

double ddNNEE::synapse0x8259a00() {
   return (neuron0x8312f40()*0.387663);
}

double ddNNEE::synapse0x8259a40() {
   return (neuron0x8313280()*0.0506177);
}

double ddNNEE::synapse0x8259a80() {
   return (neuron0x83135c0()*-0.619213);
}

double ddNNEE::synapse0x8259ac0() {
   return (neuron0x8313900()*-0.00450464);
}

double ddNNEE::synapse0x8259b00() {
   return (neuron0x8313c40()*1.74116);
}

double ddNNEE::synapse0x8317380() {
   return (neuron0x8312580()*1.39179);
}

double ddNNEE::synapse0x83173c0() {
   return (neuron0x83128c0()*-1.33156);
}

double ddNNEE::synapse0x8317400() {
   return (neuron0x8312c00()*0.412971);
}

double ddNNEE::synapse0x8317440() {
   return (neuron0x8312f40()*-0.100225);
}

double ddNNEE::synapse0x8317480() {
   return (neuron0x8313280()*-0.155398);
}

double ddNNEE::synapse0x83174c0() {
   return (neuron0x83135c0()*0.898231);
}

double ddNNEE::synapse0x8317500() {
   return (neuron0x8313900()*-0.238382);
}

double ddNNEE::synapse0x8317540() {
   return (neuron0x8313c40()*-1.06072);
}

double ddNNEE::synapse0x83178c0() {
   return (neuron0x83140b0()*0.914963);
}

double ddNNEE::synapse0x8317900() {
   return (neuron0x83144e0()*1.44698);
}

double ddNNEE::synapse0x8317940() {
   return (neuron0x8314a20()*-0.928438);
}

double ddNNEE::synapse0x8317980() {
   return (neuron0x8314ff0()*0.470698);
}

double ddNNEE::synapse0x83179c0() {
   return (neuron0x8315530()*-0.69716);
}

double ddNNEE::synapse0x8317a00() {
   return (neuron0x83158b0()*-0.57454);
}

double ddNNEE::synapse0x8317a40() {
   return (neuron0x8315df0()*0.438625);
}

double ddNNEE::synapse0x8317a80() {
   return (neuron0x8316330()*-0.170681);
}

double ddNNEE::synapse0x8317ac0() {
   return (neuron0x8316870()*1.08005);
}

double ddNNEE::synapse0x8317b00() {
   return (neuron0x83170d0()*0.711833);
}

double ddNNEE::synapse0x8317e80() {
   return (neuron0x83140b0()*0.647346);
}

double ddNNEE::synapse0x8317ec0() {
   return (neuron0x83144e0()*2.39181);
}

double ddNNEE::synapse0x8317f00() {
   return (neuron0x8314a20()*-1.55821);
}

double ddNNEE::synapse0x8317f40() {
   return (neuron0x8314ff0()*-0.215333);
}

double ddNNEE::synapse0x8317f80() {
   return (neuron0x8315530()*-0.366795);
}

double ddNNEE::synapse0x8317fc0() {
   return (neuron0x83158b0()*0.215596);
}

double ddNNEE::synapse0x8318000() {
   return (neuron0x8315df0()*-0.572009);
}

double ddNNEE::synapse0x8318040() {
   return (neuron0x8316330()*-1.12707);
}

double ddNNEE::synapse0x8318080() {
   return (neuron0x8316870()*0.874066);
}

double ddNNEE::synapse0x83180c0() {
   return (neuron0x83170d0()*1.09344);
}

double ddNNEE::synapse0x8318440() {
   return (neuron0x83140b0()*-0.283404);
}

double ddNNEE::synapse0x8318480() {
   return (neuron0x83144e0()*0.0407104);
}

double ddNNEE::synapse0x83184c0() {
   return (neuron0x8314a20()*-0.714295);
}

double ddNNEE::synapse0x8318500() {
   return (neuron0x8314ff0()*-0.168094);
}

double ddNNEE::synapse0x8318540() {
   return (neuron0x8315530()*-2.60967);
}

double ddNNEE::synapse0x8318580() {
   return (neuron0x83158b0()*-1.08998);
}

double ddNNEE::synapse0x83185c0() {
   return (neuron0x8315df0()*1.00207);
}

double ddNNEE::synapse0x8318600() {
   return (neuron0x8316330()*0.792278);
}

double ddNNEE::synapse0x8318640() {
   return (neuron0x8316870()*0.718949);
}

double ddNNEE::synapse0x8318680() {
   return (neuron0x83170d0()*2.36826);
}

double ddNNEE::synapse0x8318a00() {
   return (neuron0x83140b0()*0.478507);
}

double ddNNEE::synapse0x8318a40() {
   return (neuron0x83144e0()*0.13425);
}

double ddNNEE::synapse0x8318a80() {
   return (neuron0x8314a20()*0.281792);
}

double ddNNEE::synapse0x8318ac0() {
   return (neuron0x8314ff0()*0.769649);
}

double ddNNEE::synapse0x8318b00() {
   return (neuron0x8315530()*0.893873);
}

double ddNNEE::synapse0x8318b40() {
   return (neuron0x83158b0()*0.401025);
}

double ddNNEE::synapse0x8318b80() {
   return (neuron0x8315df0()*-0.726698);
}

double ddNNEE::synapse0x8318bc0() {
   return (neuron0x8316330()*0.377856);
}

double ddNNEE::synapse0x8318c00() {
   return (neuron0x8316870()*-0.0124937);
}

double ddNNEE::synapse0x8318c40() {
   return (neuron0x83170d0()*0.413426);
}

double ddNNEE::synapse0x8318fc0() {
   return (neuron0x83140b0()*0.910933);
}

double ddNNEE::synapse0x8319000() {
   return (neuron0x83144e0()*1.12521);
}

double ddNNEE::synapse0x8319040() {
   return (neuron0x8314a20()*-1.36901);
}

double ddNNEE::synapse0x8319080() {
   return (neuron0x8314ff0()*0.494542);
}

double ddNNEE::synapse0x83190c0() {
   return (neuron0x8315530()*-0.775266);
}

double ddNNEE::synapse0x8319100() {
   return (neuron0x83158b0()*-2.12784);
}

double ddNNEE::synapse0x8319140() {
   return (neuron0x8315df0()*-2.08337);
}

double ddNNEE::synapse0x8319180() {
   return (neuron0x8316330()*1.98218);
}

double ddNNEE::synapse0x83191c0() {
   return (neuron0x8316870()*2.82563);
}

double ddNNEE::synapse0x8316cc0() {
   return (neuron0x83170d0()*0.404478);
}

double ddNNEE::synapse0x8317040() {
   return (neuron0x8317580()*2.06702);
}

double ddNNEE::synapse0x8317080() {
   return (neuron0x8317b40()*3.65618);
}

double ddNNEE::synapse0x8313f80() {
   return (neuron0x8318100()*3.54663);
}

double ddNNEE::synapse0x8313fc0() {
   return (neuron0x83186c0()*-1.53354);
}

double ddNNEE::synapse0x8314000() {
   return (neuron0x8318c80()*4.25567);
}

