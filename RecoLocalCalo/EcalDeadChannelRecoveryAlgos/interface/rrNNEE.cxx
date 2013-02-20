#include "rrNNEE.h"
#include <cmath>

double rrNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.41121)/1.41004;
   input2 = (in2 - 2.42657)/1.40629;
   input3 = (in3 - 2.42619)/1.40466;
   input4 = (in4 - 1.33856)/1.28698;
   input5 = (in5 - 1.33177)/1.28879;
   input6 = (in6 - 1.33367)/1.29347;
   input7 = (in7 - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x672db80();
     default:
         return 0.;
   }
}

double rrNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.41121)/1.41004;
   input2 = (input[2] - 2.42657)/1.40629;
   input3 = (input[3] - 2.42619)/1.40466;
   input4 = (input[4] - 1.33856)/1.28698;
   input5 = (input[5] - 1.33177)/1.28879;
   input6 = (input[6] - 1.33367)/1.29347;
   input7 = (input[7] - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x672db80();
     default:
         return 0.;
   }
}

double rrNNEE::neuron0x6729400() {
   return input0;
}

double rrNNEE::neuron0x6729740() {
   return input1;
}

double rrNNEE::neuron0x6729a80() {
   return input2;
}

double rrNNEE::neuron0x6729dc0() {
   return input3;
}

double rrNNEE::neuron0x672a100() {
   return input4;
}

double rrNNEE::neuron0x672a440() {
   return input5;
}

double rrNNEE::neuron0x672a780() {
   return input6;
}

double rrNNEE::neuron0x672aac0() {
   return input7;
}

double rrNNEE::input0x672af30() {
   double input = 0.653896;
   input += synapse0x6689740();
   input += synapse0x6732000();
   input += synapse0x672b1e0();
   input += synapse0x672b220();
   input += synapse0x672b260();
   input += synapse0x672b2a0();
   input += synapse0x672b2e0();
   input += synapse0x672b320();
   return input;
}

double rrNNEE::neuron0x672af30() {
   double input = input0x672af30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672b360() {
   double input = 0.707726;
   input += synapse0x672b6a0();
   input += synapse0x672b6e0();
   input += synapse0x672b720();
   input += synapse0x672b760();
   input += synapse0x672b7a0();
   input += synapse0x672b7e0();
   input += synapse0x672b820();
   input += synapse0x672b860();
   return input;
}

double rrNNEE::neuron0x672b360() {
   double input = input0x672b360();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672b8a0() {
   double input = -0.441893;
   input += synapse0x672bbe0();
   input += synapse0x6657ce0();
   input += synapse0x6657d20();
   input += synapse0x672bd30();
   input += synapse0x672bd70();
   input += synapse0x672bdb0();
   input += synapse0x672bdf0();
   input += synapse0x672be30();
   return input;
}

double rrNNEE::neuron0x672b8a0() {
   double input = input0x672b8a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672be70() {
   double input = -0.634536;
   input += synapse0x672c1b0();
   input += synapse0x672c1f0();
   input += synapse0x672c230();
   input += synapse0x672c270();
   input += synapse0x672c2b0();
   input += synapse0x672c2f0();
   input += synapse0x672c330();
   input += synapse0x672c370();
   return input;
}

double rrNNEE::neuron0x672be70() {
   double input = input0x672be70();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672c3b0() {
   double input = 0.53862;
   input += synapse0x672c6f0();
   input += synapse0x6729330();
   input += synapse0x6732040();
   input += synapse0x6674050();
   input += synapse0x672bc20();
   input += synapse0x672bc60();
   input += synapse0x672bca0();
   input += synapse0x672bce0();
   return input;
}

double rrNNEE::neuron0x672c3b0() {
   double input = input0x672c3b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672c730() {
   double input = -1.38965;
   input += synapse0x672ca70();
   input += synapse0x672cab0();
   input += synapse0x672caf0();
   input += synapse0x672cb30();
   input += synapse0x672cb70();
   input += synapse0x672cbb0();
   input += synapse0x672cbf0();
   input += synapse0x672cc30();
   return input;
}

double rrNNEE::neuron0x672c730() {
   double input = input0x672c730();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672cc70() {
   double input = -2.69757;
   input += synapse0x672cfb0();
   input += synapse0x672cff0();
   input += synapse0x672d030();
   input += synapse0x672d070();
   input += synapse0x672d0b0();
   input += synapse0x672d0f0();
   input += synapse0x672d130();
   input += synapse0x672d170();
   return input;
}

double rrNNEE::neuron0x672cc70() {
   double input = input0x672cc70();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672d1b0() {
   double input = 0.552602;
   input += synapse0x672d4f0();
   input += synapse0x672d530();
   input += synapse0x672d570();
   input += synapse0x672d5b0();
   input += synapse0x672d5f0();
   input += synapse0x672d630();
   input += synapse0x672d670();
   input += synapse0x672d6b0();
   return input;
}

double rrNNEE::neuron0x672d1b0() {
   double input = input0x672d1b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672d6f0() {
   double input = -0.289258;
   input += synapse0x6655940();
   input += synapse0x6655980();
   input += synapse0x6670840();
   input += synapse0x6670880();
   input += synapse0x66708c0();
   input += synapse0x6670900();
   input += synapse0x6670940();
   input += synapse0x6670980();
   return input;
}

double rrNNEE::neuron0x672d6f0() {
   double input = input0x672d6f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672df50() {
   double input = 0.0539445;
   input += synapse0x672e200();
   input += synapse0x672e240();
   input += synapse0x672e280();
   input += synapse0x672e2c0();
   input += synapse0x672e300();
   input += synapse0x672e340();
   input += synapse0x672e380();
   input += synapse0x672e3c0();
   return input;
}

double rrNNEE::neuron0x672df50() {
   double input = input0x672df50();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672e400() {
   double input = 0.422267;
   input += synapse0x672e740();
   input += synapse0x672e780();
   input += synapse0x672e7c0();
   input += synapse0x672e800();
   input += synapse0x672e840();
   input += synapse0x672e880();
   input += synapse0x672e8c0();
   input += synapse0x672e900();
   input += synapse0x672e940();
   input += synapse0x672e980();
   return input;
}

double rrNNEE::neuron0x672e400() {
   double input = input0x672e400();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672e9c0() {
   double input = -0.298858;
   input += synapse0x672ed00();
   input += synapse0x672ed40();
   input += synapse0x672ed80();
   input += synapse0x672edc0();
   input += synapse0x672ee00();
   input += synapse0x672ee40();
   input += synapse0x672ee80();
   input += synapse0x672eec0();
   input += synapse0x672ef00();
   input += synapse0x672ef40();
   return input;
}

double rrNNEE::neuron0x672e9c0() {
   double input = input0x672e9c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672ef80() {
   double input = 0.87923;
   input += synapse0x672f2c0();
   input += synapse0x672f300();
   input += synapse0x672f340();
   input += synapse0x672f380();
   input += synapse0x672f3c0();
   input += synapse0x672f400();
   input += synapse0x672f440();
   input += synapse0x672f480();
   input += synapse0x672f4c0();
   input += synapse0x672f500();
   return input;
}

double rrNNEE::neuron0x672ef80() {
   double input = input0x672ef80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672f540() {
   double input = -1.51041;
   input += synapse0x672f880();
   input += synapse0x672f8c0();
   input += synapse0x672f900();
   input += synapse0x672f940();
   input += synapse0x672f980();
   input += synapse0x672f9c0();
   input += synapse0x672fa00();
   input += synapse0x672fa40();
   input += synapse0x672fa80();
   input += synapse0x672fac0();
   return input;
}

double rrNNEE::neuron0x672f540() {
   double input = input0x672f540();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672fb00() {
   double input = -3.09485;
   input += synapse0x672fe40();
   input += synapse0x672fe80();
   input += synapse0x672fec0();
   input += synapse0x672ff00();
   input += synapse0x672ff40();
   input += synapse0x672ff80();
   input += synapse0x672ffc0();
   input += synapse0x6730000();
   input += synapse0x6730040();
   input += synapse0x672db40();
   return input;
}

double rrNNEE::neuron0x672fb00() {
   double input = input0x672fb00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rrNNEE::input0x672db80() {
   double input = 3.93219;
   input += synapse0x672dec0();
   input += synapse0x672df00();
   input += synapse0x672ae00();
   input += synapse0x672ae40();
   input += synapse0x672ae80();
   return input;
}

double rrNNEE::neuron0x672db80() {
   double input = input0x672db80();
   return (input * 1)+0;
}

double rrNNEE::synapse0x6689740() {
   return (neuron0x6729400()*0.719572);
}

double rrNNEE::synapse0x6732000() {
   return (neuron0x6729740()*-0.0692008);
}

double rrNNEE::synapse0x672b1e0() {
   return (neuron0x6729a80()*-1.18787);
}

double rrNNEE::synapse0x672b220() {
   return (neuron0x6729dc0()*0.707363);
}

double rrNNEE::synapse0x672b260() {
   return (neuron0x672a100()*1.16964);
}

double rrNNEE::synapse0x672b2a0() {
   return (neuron0x672a440()*-1.01254);
}

double rrNNEE::synapse0x672b2e0() {
   return (neuron0x672a780()*0.129117);
}

double rrNNEE::synapse0x672b320() {
   return (neuron0x672aac0()*-0.166791);
}

double rrNNEE::synapse0x672b6a0() {
   return (neuron0x6729400()*-0.529282);
}

double rrNNEE::synapse0x672b6e0() {
   return (neuron0x6729740()*0.732823);
}

double rrNNEE::synapse0x672b720() {
   return (neuron0x6729a80()*-0.361648);
}

double rrNNEE::synapse0x672b760() {
   return (neuron0x6729dc0()*-0.610316);
}

double rrNNEE::synapse0x672b7a0() {
   return (neuron0x672a100()*0.188769);
}

double rrNNEE::synapse0x672b7e0() {
   return (neuron0x672a440()*0.542069);
}

double rrNNEE::synapse0x672b820() {
   return (neuron0x672a780()*-0.600285);
}

double rrNNEE::synapse0x672b860() {
   return (neuron0x672aac0()*-0.0701344);
}

double rrNNEE::synapse0x672bbe0() {
   return (neuron0x6729400()*1.54523);
}

double rrNNEE::synapse0x6657ce0() {
   return (neuron0x6729740()*-0.498099);
}

double rrNNEE::synapse0x6657d20() {
   return (neuron0x6729a80()*-0.726753);
}

double rrNNEE::synapse0x672bd30() {
   return (neuron0x6729dc0()*-0.842911);
}

double rrNNEE::synapse0x672bd70() {
   return (neuron0x672a100()*-0.469895);
}

double rrNNEE::synapse0x672bdb0() {
   return (neuron0x672a440()*-0.377609);
}

double rrNNEE::synapse0x672bdf0() {
   return (neuron0x672a780()*0.813695);
}

double rrNNEE::synapse0x672be30() {
   return (neuron0x672aac0()*0.636629);
}

double rrNNEE::synapse0x672c1b0() {
   return (neuron0x6729400()*1.65108);
}

double rrNNEE::synapse0x672c1f0() {
   return (neuron0x6729740()*-0.190199);
}

double rrNNEE::synapse0x672c230() {
   return (neuron0x6729a80()*0.982778);
}

double rrNNEE::synapse0x672c270() {
   return (neuron0x6729dc0()*0.711544);
}

double rrNNEE::synapse0x672c2b0() {
   return (neuron0x672a100()*-3.26145);
}

double rrNNEE::synapse0x672c2f0() {
   return (neuron0x672a440()*0.162657);
}

double rrNNEE::synapse0x672c330() {
   return (neuron0x672a780()*-0.089699);
}

double rrNNEE::synapse0x672c370() {
   return (neuron0x672aac0()*-0.851981);
}

double rrNNEE::synapse0x672c6f0() {
   return (neuron0x6729400()*0.772435);
}

double rrNNEE::synapse0x6729330() {
   return (neuron0x6729740()*0.152775);
}

double rrNNEE::synapse0x6732040() {
   return (neuron0x6729a80()*0.723079);
}

double rrNNEE::synapse0x6674050() {
   return (neuron0x6729dc0()*-0.917971);
}

double rrNNEE::synapse0x672bc20() {
   return (neuron0x672a100()*-1.09432);
}

double rrNNEE::synapse0x672bc60() {
   return (neuron0x672a440()*0.928471);
}

double rrNNEE::synapse0x672bca0() {
   return (neuron0x672a780()*-0.358788);
}

double rrNNEE::synapse0x672bce0() {
   return (neuron0x672aac0()*-0.0812723);
}

double rrNNEE::synapse0x672ca70() {
   return (neuron0x6729400()*0.374882);
}

double rrNNEE::synapse0x672cab0() {
   return (neuron0x6729740()*-1.11654);
}

double rrNNEE::synapse0x672caf0() {
   return (neuron0x6729a80()*0.847504);
}

double rrNNEE::synapse0x672cb30() {
   return (neuron0x6729dc0()*0.602074);
}

double rrNNEE::synapse0x672cb70() {
   return (neuron0x672a100()*-0.28634);
}

double rrNNEE::synapse0x672cbb0() {
   return (neuron0x672a440()*-0.371769);
}

double rrNNEE::synapse0x672cbf0() {
   return (neuron0x672a780()*-0.900869);
}

double rrNNEE::synapse0x672cc30() {
   return (neuron0x672aac0()*1.50983);
}

double rrNNEE::synapse0x672cfb0() {
   return (neuron0x6729400()*0.370821);
}

double rrNNEE::synapse0x672cff0() {
   return (neuron0x6729740()*1.07235);
}

double rrNNEE::synapse0x672d030() {
   return (neuron0x6729a80()*-0.367159);
}

double rrNNEE::synapse0x672d070() {
   return (neuron0x6729dc0()*-0.68921);
}

double rrNNEE::synapse0x672d0b0() {
   return (neuron0x672a100()*1.15195);
}

double rrNNEE::synapse0x672d0f0() {
   return (neuron0x672a440()*1.46823);
}

double rrNNEE::synapse0x672d130() {
   return (neuron0x672a780()*0.0745679);
}

double rrNNEE::synapse0x672d170() {
   return (neuron0x672aac0()*-0.00508042);
}

double rrNNEE::synapse0x672d4f0() {
   return (neuron0x6729400()*-1.81515);
}

double rrNNEE::synapse0x672d530() {
   return (neuron0x6729740()*0.478004);
}

double rrNNEE::synapse0x672d570() {
   return (neuron0x6729a80()*-0.609589);
}

double rrNNEE::synapse0x672d5b0() {
   return (neuron0x6729dc0()*-1.08779);
}

double rrNNEE::synapse0x672d5f0() {
   return (neuron0x672a100()*-0.318204);
}

double rrNNEE::synapse0x672d630() {
   return (neuron0x672a440()*3.54867);
}

double rrNNEE::synapse0x672d670() {
   return (neuron0x672a780()*0.79003);
}

double rrNNEE::synapse0x672d6b0() {
   return (neuron0x672aac0()*0.0066591);
}

double rrNNEE::synapse0x6655940() {
   return (neuron0x6729400()*-0.603721);
}

double rrNNEE::synapse0x6655980() {
   return (neuron0x6729740()*-0.0150205);
}

double rrNNEE::synapse0x6670840() {
   return (neuron0x6729a80()*0.0899062);
}

double rrNNEE::synapse0x6670880() {
   return (neuron0x6729dc0()*-2.32862);
}

double rrNNEE::synapse0x66708c0() {
   return (neuron0x672a100()*0.392175);
}

double rrNNEE::synapse0x6670900() {
   return (neuron0x672a440()*2.26292);
}

double rrNNEE::synapse0x6670940() {
   return (neuron0x672a780()*-0.117487);
}

double rrNNEE::synapse0x6670980() {
   return (neuron0x672aac0()*0.0187756);
}

double rrNNEE::synapse0x672e200() {
   return (neuron0x6729400()*0.0326351);
}

double rrNNEE::synapse0x672e240() {
   return (neuron0x6729740()*0.0883082);
}

double rrNNEE::synapse0x672e280() {
   return (neuron0x6729a80()*-2.62164);
}

double rrNNEE::synapse0x672e2c0() {
   return (neuron0x6729dc0()*-0.626323);
}

double rrNNEE::synapse0x672e300() {
   return (neuron0x672a100()*1.77677);
}

double rrNNEE::synapse0x672e340() {
   return (neuron0x672a440()*0.771757);
}

double rrNNEE::synapse0x672e380() {
   return (neuron0x672a780()*0.0133546);
}

double rrNNEE::synapse0x672e3c0() {
   return (neuron0x672aac0()*-0.108335);
}

double rrNNEE::synapse0x672e740() {
   return (neuron0x672af30()*-1.46721);
}

double rrNNEE::synapse0x672e780() {
   return (neuron0x672b360()*1.19058);
}

double rrNNEE::synapse0x672e7c0() {
   return (neuron0x672b8a0()*-0.719918);
}

double rrNNEE::synapse0x672e800() {
   return (neuron0x672be70()*-0.0205937);
}

double rrNNEE::synapse0x672e840() {
   return (neuron0x672c3b0()*-1.98777);
}

double rrNNEE::synapse0x672e880() {
   return (neuron0x672c730()*-0.0982577);
}

double rrNNEE::synapse0x672e8c0() {
   return (neuron0x672cc70()*-1.46517);
}

double rrNNEE::synapse0x672e900() {
   return (neuron0x672d1b0()*0.113774);
}

double rrNNEE::synapse0x672e940() {
   return (neuron0x672d6f0()*0.045319);
}

double rrNNEE::synapse0x672e980() {
   return (neuron0x672df50()*0.421127);
}

double rrNNEE::synapse0x672ed00() {
   return (neuron0x672af30()*-0.974081);
}

double rrNNEE::synapse0x672ed40() {
   return (neuron0x672b360()*-0.351462);
}

double rrNNEE::synapse0x672ed80() {
   return (neuron0x672b8a0()*-0.342592);
}

double rrNNEE::synapse0x672edc0() {
   return (neuron0x672be70()*0.225037);
}

double rrNNEE::synapse0x672ee00() {
   return (neuron0x672c3b0()*-0.192674);
}

double rrNNEE::synapse0x672ee40() {
   return (neuron0x672c730()*-0.82117);
}

double rrNNEE::synapse0x672ee80() {
   return (neuron0x672cc70()*-0.175556);
}

double rrNNEE::synapse0x672eec0() {
   return (neuron0x672d1b0()*0.0352347);
}

double rrNNEE::synapse0x672ef00() {
   return (neuron0x672d6f0()*0.155098);
}

double rrNNEE::synapse0x672ef40() {
   return (neuron0x672df50()*0.465587);
}

double rrNNEE::synapse0x672f2c0() {
   return (neuron0x672af30()*-1.93436);
}

double rrNNEE::synapse0x672f300() {
   return (neuron0x672b360()*1.73157);
}

double rrNNEE::synapse0x672f340() {
   return (neuron0x672b8a0()*1.47908);
}

double rrNNEE::synapse0x672f380() {
   return (neuron0x672be70()*1.01792);
}

double rrNNEE::synapse0x672f3c0() {
   return (neuron0x672c3b0()*-2.70061);
}

double rrNNEE::synapse0x672f400() {
   return (neuron0x672c730()*-0.351462);
}

double rrNNEE::synapse0x672f440() {
   return (neuron0x672cc70()*0.330296);
}

double rrNNEE::synapse0x672f480() {
   return (neuron0x672d1b0()*-0.941561);
}

double rrNNEE::synapse0x672f4c0() {
   return (neuron0x672d6f0()*-0.0761283);
}

double rrNNEE::synapse0x672f500() {
   return (neuron0x672df50()*-0.32878);
}

double rrNNEE::synapse0x672f880() {
   return (neuron0x672af30()*0.700144);
}

double rrNNEE::synapse0x672f8c0() {
   return (neuron0x672b360()*-0.962227);
}

double rrNNEE::synapse0x672f900() {
   return (neuron0x672b8a0()*-2.48208);
}

double rrNNEE::synapse0x672f940() {
   return (neuron0x672be70()*-1.33986);
}

double rrNNEE::synapse0x672f980() {
   return (neuron0x672c3b0()*-1.63243);
}

double rrNNEE::synapse0x672f9c0() {
   return (neuron0x672c730()*0.559641);
}

double rrNNEE::synapse0x672fa00() {
   return (neuron0x672cc70()*-0.537616);
}

double rrNNEE::synapse0x672fa40() {
   return (neuron0x672d1b0()*1.76524);
}

double rrNNEE::synapse0x672fa80() {
   return (neuron0x672d6f0()*2.37791);
}

double rrNNEE::synapse0x672fac0() {
   return (neuron0x672df50()*1.21161);
}

double rrNNEE::synapse0x672fe40() {
   return (neuron0x672af30()*0.775653);
}

double rrNNEE::synapse0x672fe80() {
   return (neuron0x672b360()*-2.31679);
}

double rrNNEE::synapse0x672fec0() {
   return (neuron0x672b8a0()*-0.314825);
}

double rrNNEE::synapse0x672ff00() {
   return (neuron0x672be70()*0.281125);
}

double rrNNEE::synapse0x672ff40() {
   return (neuron0x672c3b0()*1.0574);
}

double rrNNEE::synapse0x672ff80() {
   return (neuron0x672c730()*0.183422);
}

double rrNNEE::synapse0x672ffc0() {
   return (neuron0x672cc70()*0.923959);
}

double rrNNEE::synapse0x6730000() {
   return (neuron0x672d1b0()*-0.495954);
}

double rrNNEE::synapse0x6730040() {
   return (neuron0x672d6f0()*1.13259);
}

double rrNNEE::synapse0x672db40() {
   return (neuron0x672df50()*1.30696);
}

double rrNNEE::synapse0x672dec0() {
   return (neuron0x672e400()*-3.12656);
}

double rrNNEE::synapse0x672df00() {
   return (neuron0x672e9c0()*-1.07836);
}

double rrNNEE::synapse0x672ae00() {
   return (neuron0x672ef80()*-4.12287);
}

double rrNNEE::synapse0x672ae40() {
   return (neuron0x672f540()*2.16816);
}

double rrNNEE::synapse0x672ae80() {
   return (neuron0x672fb00()*4.26021);
}

