#include "rrNNEB.h"
#include <cmath>

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

