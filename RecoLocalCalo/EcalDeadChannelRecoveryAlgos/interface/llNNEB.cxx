#include "llNNEB.h"
#include <cmath>

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

