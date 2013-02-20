#include "ruNNEE.h"
#include <cmath>

double ruNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.40353)/1.40003;
   input2 = (in2 - 2.41121)/1.41004;
   input3 = (in3 - 2.42657)/1.40629;
   input4 = (in4 - 2.42619)/1.40466;
   input5 = (in5 - 1.33177)/1.28879;
   input6 = (in6 - 1.33367)/1.29347;
   input7 = (in7 - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0xa82cc40();
     default:
         return 0.;
   }
}

double ruNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.40353)/1.40003;
   input2 = (input[2] - 2.41121)/1.41004;
   input3 = (input[3] - 2.42657)/1.40629;
   input4 = (input[4] - 2.42619)/1.40466;
   input5 = (input[5] - 1.33177)/1.28879;
   input6 = (input[6] - 1.33367)/1.29347;
   input7 = (input[7] - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0xa82cc40();
     default:
         return 0.;
   }
}

double ruNNEE::neuron0xa8284c0() {
   return input0;
}

double ruNNEE::neuron0xa828800() {
   return input1;
}

double ruNNEE::neuron0xa828b40() {
   return input2;
}

double ruNNEE::neuron0xa828e80() {
   return input3;
}

double ruNNEE::neuron0xa8291c0() {
   return input4;
}

double ruNNEE::neuron0xa829500() {
   return input5;
}

double ruNNEE::neuron0xa829840() {
   return input6;
}

double ruNNEE::neuron0xa829b80() {
   return input7;
}

double ruNNEE::input0xa829ff0() {
   double input = -0.739082;
   input += synapse0xa788800();
   input += synapse0xa8310c0();
   input += synapse0xa82a2a0();
   input += synapse0xa82a2e0();
   input += synapse0xa82a320();
   input += synapse0xa82a360();
   input += synapse0xa82a3a0();
   input += synapse0xa82a3e0();
   return input;
}

double ruNNEE::neuron0xa829ff0() {
   double input = input0xa829ff0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82a420() {
   double input = -0.228431;
   input += synapse0xa82a760();
   input += synapse0xa82a7a0();
   input += synapse0xa82a7e0();
   input += synapse0xa82a820();
   input += synapse0xa82a860();
   input += synapse0xa82a8a0();
   input += synapse0xa82a8e0();
   input += synapse0xa82a920();
   return input;
}

double ruNNEE::neuron0xa82a420() {
   double input = input0xa82a420();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82a960() {
   double input = -2.50648;
   input += synapse0xa82aca0();
   input += synapse0xa756da0();
   input += synapse0xa756de0();
   input += synapse0xa82adf0();
   input += synapse0xa82ae30();
   input += synapse0xa82ae70();
   input += synapse0xa82aeb0();
   input += synapse0xa82aef0();
   return input;
}

double ruNNEE::neuron0xa82a960() {
   double input = input0xa82a960();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82af30() {
   double input = 2.40472;
   input += synapse0xa82b270();
   input += synapse0xa82b2b0();
   input += synapse0xa82b2f0();
   input += synapse0xa82b330();
   input += synapse0xa82b370();
   input += synapse0xa82b3b0();
   input += synapse0xa82b3f0();
   input += synapse0xa82b430();
   return input;
}

double ruNNEE::neuron0xa82af30() {
   double input = input0xa82af30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82b470() {
   double input = -2.81377;
   input += synapse0xa82b7b0();
   input += synapse0xa8283f0();
   input += synapse0xa831100();
   input += synapse0xa773110();
   input += synapse0xa82ace0();
   input += synapse0xa82ad20();
   input += synapse0xa82ad60();
   input += synapse0xa82ada0();
   return input;
}

double ruNNEE::neuron0xa82b470() {
   double input = input0xa82b470();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82b7f0() {
   double input = 0.360115;
   input += synapse0xa82bb30();
   input += synapse0xa82bb70();
   input += synapse0xa82bbb0();
   input += synapse0xa82bbf0();
   input += synapse0xa82bc30();
   input += synapse0xa82bc70();
   input += synapse0xa82bcb0();
   input += synapse0xa82bcf0();
   return input;
}

double ruNNEE::neuron0xa82b7f0() {
   double input = input0xa82b7f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82bd30() {
   double input = -1.41676;
   input += synapse0xa82c070();
   input += synapse0xa82c0b0();
   input += synapse0xa82c0f0();
   input += synapse0xa82c130();
   input += synapse0xa82c170();
   input += synapse0xa82c1b0();
   input += synapse0xa82c1f0();
   input += synapse0xa82c230();
   return input;
}

double ruNNEE::neuron0xa82bd30() {
   double input = input0xa82bd30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82c270() {
   double input = 0.979543;
   input += synapse0xa82c5b0();
   input += synapse0xa82c5f0();
   input += synapse0xa82c630();
   input += synapse0xa82c670();
   input += synapse0xa82c6b0();
   input += synapse0xa82c6f0();
   input += synapse0xa82c730();
   input += synapse0xa82c770();
   return input;
}

double ruNNEE::neuron0xa82c270() {
   double input = input0xa82c270();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82c7b0() {
   double input = 2.28198;
   input += synapse0xa754a00();
   input += synapse0xa754a40();
   input += synapse0xa76f900();
   input += synapse0xa76f940();
   input += synapse0xa76f980();
   input += synapse0xa76f9c0();
   input += synapse0xa76fa00();
   input += synapse0xa76fa40();
   return input;
}

double ruNNEE::neuron0xa82c7b0() {
   double input = input0xa82c7b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82d010() {
   double input = -2.40913;
   input += synapse0xa82d2c0();
   input += synapse0xa82d300();
   input += synapse0xa82d340();
   input += synapse0xa82d380();
   input += synapse0xa82d3c0();
   input += synapse0xa82d400();
   input += synapse0xa82d440();
   input += synapse0xa82d480();
   return input;
}

double ruNNEE::neuron0xa82d010() {
   double input = input0xa82d010();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82d4c0() {
   double input = 0.0880209;
   input += synapse0xa82d800();
   input += synapse0xa82d840();
   input += synapse0xa82d880();
   input += synapse0xa82d8c0();
   input += synapse0xa82d900();
   input += synapse0xa82d940();
   input += synapse0xa82d980();
   input += synapse0xa82d9c0();
   input += synapse0xa82da00();
   input += synapse0xa82da40();
   return input;
}

double ruNNEE::neuron0xa82d4c0() {
   double input = input0xa82d4c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82da80() {
   double input = 0.645116;
   input += synapse0xa82ddc0();
   input += synapse0xa82de00();
   input += synapse0xa82de40();
   input += synapse0xa82de80();
   input += synapse0xa82dec0();
   input += synapse0xa82df00();
   input += synapse0xa82df40();
   input += synapse0xa82df80();
   input += synapse0xa82dfc0();
   input += synapse0xa82e000();
   return input;
}

double ruNNEE::neuron0xa82da80() {
   double input = input0xa82da80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82e040() {
   double input = 0.00539517;
   input += synapse0xa82e380();
   input += synapse0xa82e3c0();
   input += synapse0xa82e400();
   input += synapse0xa82e440();
   input += synapse0xa82e480();
   input += synapse0xa82e4c0();
   input += synapse0xa82e500();
   input += synapse0xa82e540();
   input += synapse0xa82e580();
   input += synapse0xa82e5c0();
   return input;
}

double ruNNEE::neuron0xa82e040() {
   double input = input0xa82e040();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82e600() {
   double input = 0.810706;
   input += synapse0xa82e940();
   input += synapse0xa82e980();
   input += synapse0xa82e9c0();
   input += synapse0xa82ea00();
   input += synapse0xa82ea40();
   input += synapse0xa82ea80();
   input += synapse0xa82eac0();
   input += synapse0xa82eb00();
   input += synapse0xa82eb40();
   input += synapse0xa82eb80();
   return input;
}

double ruNNEE::neuron0xa82e600() {
   double input = input0xa82e600();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82ebc0() {
   double input = -0.83505;
   input += synapse0xa82ef00();
   input += synapse0xa82ef40();
   input += synapse0xa82ef80();
   input += synapse0xa82efc0();
   input += synapse0xa82f000();
   input += synapse0xa82f040();
   input += synapse0xa82f080();
   input += synapse0xa82f0c0();
   input += synapse0xa82f100();
   input += synapse0xa82cc00();
   return input;
}

double ruNNEE::neuron0xa82ebc0() {
   double input = input0xa82ebc0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ruNNEE::input0xa82cc40() {
   double input = 0.0469036;
   input += synapse0xa82cf80();
   input += synapse0xa82cfc0();
   input += synapse0xa829ec0();
   input += synapse0xa829f00();
   input += synapse0xa829f40();
   return input;
}

double ruNNEE::neuron0xa82cc40() {
   double input = input0xa82cc40();
   return (input * 1)+0;
}

double ruNNEE::synapse0xa788800() {
   return (neuron0xa8284c0()*-1.47017);
}

double ruNNEE::synapse0xa8310c0() {
   return (neuron0xa828800()*0.300555);
}

double ruNNEE::synapse0xa82a2a0() {
   return (neuron0xa828b40()*-0.153933);
}

double ruNNEE::synapse0xa82a2e0() {
   return (neuron0xa828e80()*0.287892);
}

double ruNNEE::synapse0xa82a320() {
   return (neuron0xa8291c0()*0.0337064);
}

double ruNNEE::synapse0xa82a360() {
   return (neuron0xa829500()*0.235817);
}

double ruNNEE::synapse0xa82a3a0() {
   return (neuron0xa829840()*0.123549);
}

double ruNNEE::synapse0xa82a3e0() {
   return (neuron0xa829b80()*-0.124928);
}

double ruNNEE::synapse0xa82a760() {
   return (neuron0xa8284c0()*0.274245);
}

double ruNNEE::synapse0xa82a7a0() {
   return (neuron0xa828800()*-1.08182);
}

double ruNNEE::synapse0xa82a7e0() {
   return (neuron0xa828b40()*-0.82647);
}

double ruNNEE::synapse0xa82a820() {
   return (neuron0xa828e80()*-1.00043);
}

double ruNNEE::synapse0xa82a860() {
   return (neuron0xa8291c0()*0.143352);
}

double ruNNEE::synapse0xa82a8a0() {
   return (neuron0xa829500()*0.345801);
}

double ruNNEE::synapse0xa82a8e0() {
   return (neuron0xa829840()*2.88779);
}

double ruNNEE::synapse0xa82a920() {
   return (neuron0xa829b80()*-0.550332);
}

double ruNNEE::synapse0xa82aca0() {
   return (neuron0xa8284c0()*1.06424);
}

double ruNNEE::synapse0xa756da0() {
   return (neuron0xa828800()*-2.4593);
}

double ruNNEE::synapse0xa756de0() {
   return (neuron0xa828b40()*-0.716967);
}

double ruNNEE::synapse0xa82adf0() {
   return (neuron0xa828e80()*0.821676);
}

double ruNNEE::synapse0xa82ae30() {
   return (neuron0xa8291c0()*0.888099);
}

double ruNNEE::synapse0xa82ae70() {
   return (neuron0xa829500()*-0.430376);
}

double ruNNEE::synapse0xa82aeb0() {
   return (neuron0xa829840()*-0.0471971);
}

double ruNNEE::synapse0xa82aef0() {
   return (neuron0xa829b80()*0.214872);
}

double ruNNEE::synapse0xa82b270() {
   return (neuron0xa8284c0()*-0.175604);
}

double ruNNEE::synapse0xa82b2b0() {
   return (neuron0xa828800()*-0.391058);
}

double ruNNEE::synapse0xa82b2f0() {
   return (neuron0xa828b40()*-0.0195448);
}

double ruNNEE::synapse0xa82b330() {
   return (neuron0xa828e80()*-0.409259);
}

double ruNNEE::synapse0xa82b370() {
   return (neuron0xa8291c0()*-0.290356);
}

double ruNNEE::synapse0xa82b3b0() {
   return (neuron0xa829500()*0.0849884);
}

double ruNNEE::synapse0xa82b3f0() {
   return (neuron0xa829840()*-0.286946);
}

double ruNNEE::synapse0xa82b430() {
   return (neuron0xa829b80()*0.0658396);
}

double ruNNEE::synapse0xa82b7b0() {
   return (neuron0xa8284c0()*1.86726);
}

double ruNNEE::synapse0xa8283f0() {
   return (neuron0xa828800()*0.673606);
}

double ruNNEE::synapse0xa831100() {
   return (neuron0xa828b40()*1.28299);
}

double ruNNEE::synapse0xa773110() {
   return (neuron0xa828e80()*-3.07066);
}

double ruNNEE::synapse0xa82ace0() {
   return (neuron0xa8291c0()*-1.19893);
}

double ruNNEE::synapse0xa82ad20() {
   return (neuron0xa829500()*0.0879598);
}

double ruNNEE::synapse0xa82ad60() {
   return (neuron0xa829840()*-1.15324);
}

double ruNNEE::synapse0xa82ada0() {
   return (neuron0xa829b80()*0.282287);
}

double ruNNEE::synapse0xa82bb30() {
   return (neuron0xa8284c0()*-0.798174);
}

double ruNNEE::synapse0xa82bb70() {
   return (neuron0xa828800()*-0.23108);
}

double ruNNEE::synapse0xa82bbb0() {
   return (neuron0xa828b40()*-0.0903257);
}

double ruNNEE::synapse0xa82bbf0() {
   return (neuron0xa828e80()*-0.330533);
}

double ruNNEE::synapse0xa82bc30() {
   return (neuron0xa8291c0()*0.0800251);
}

double ruNNEE::synapse0xa82bc70() {
   return (neuron0xa829500()*-0.0635542);
}

double ruNNEE::synapse0xa82bcb0() {
   return (neuron0xa829840()*-0.0161712);
}

double ruNNEE::synapse0xa82bcf0() {
   return (neuron0xa829b80()*-0.270878);
}

double ruNNEE::synapse0xa82c070() {
   return (neuron0xa8284c0()*1.50949);
}

double ruNNEE::synapse0xa82c0b0() {
   return (neuron0xa828800()*0.0766836);
}

double ruNNEE::synapse0xa82c0f0() {
   return (neuron0xa828b40()*-1.30968);
}

double ruNNEE::synapse0xa82c130() {
   return (neuron0xa828e80()*0.193006);
}

double ruNNEE::synapse0xa82c170() {
   return (neuron0xa8291c0()*-1.36285);
}

double ruNNEE::synapse0xa82c1b0() {
   return (neuron0xa829500()*0.951778);
}

double ruNNEE::synapse0xa82c1f0() {
   return (neuron0xa829840()*0.71593);
}

double ruNNEE::synapse0xa82c230() {
   return (neuron0xa829b80()*-0.586066);
}

double ruNNEE::synapse0xa82c5b0() {
   return (neuron0xa8284c0()*1.28256);
}

double ruNNEE::synapse0xa82c5f0() {
   return (neuron0xa828800()*0.534974);
}

double ruNNEE::synapse0xa82c630() {
   return (neuron0xa828b40()*-0.443663);
}

double ruNNEE::synapse0xa82c670() {
   return (neuron0xa828e80()*0.850094);
}

double ruNNEE::synapse0xa82c6b0() {
   return (neuron0xa8291c0()*-0.180367);
}

double ruNNEE::synapse0xa82c6f0() {
   return (neuron0xa829500()*-2.35516);
}

double ruNNEE::synapse0xa82c730() {
   return (neuron0xa829840()*-0.806304);
}

double ruNNEE::synapse0xa82c770() {
   return (neuron0xa829b80()*0.573719);
}

double ruNNEE::synapse0xa754a00() {
   return (neuron0xa8284c0()*1.13865);
}

double ruNNEE::synapse0xa754a40() {
   return (neuron0xa828800()*0.327888);
}

double ruNNEE::synapse0xa76f900() {
   return (neuron0xa828b40()*0.0724785);
}

double ruNNEE::synapse0xa76f940() {
   return (neuron0xa828e80()*0.382394);
}

double ruNNEE::synapse0xa76f980() {
   return (neuron0xa8291c0()*-0.00753103);
}

double ruNNEE::synapse0xa76f9c0() {
   return (neuron0xa829500()*-0.190601);
}

double ruNNEE::synapse0xa76fa00() {
   return (neuron0xa829840()*-0.234588);
}

double ruNNEE::synapse0xa76fa40() {
   return (neuron0xa829b80()*-0.00238543);
}

double ruNNEE::synapse0xa82d2c0() {
   return (neuron0xa8284c0()*0.859022);
}

double ruNNEE::synapse0xa82d300() {
   return (neuron0xa828800()*0.239251);
}

double ruNNEE::synapse0xa82d340() {
   return (neuron0xa828b40()*-0.627504);
}

double ruNNEE::synapse0xa82d380() {
   return (neuron0xa828e80()*0.230238);
}

double ruNNEE::synapse0xa82d3c0() {
   return (neuron0xa8291c0()*-0.417699);
}

double ruNNEE::synapse0xa82d400() {
   return (neuron0xa829500()*0.0416927);
}

double ruNNEE::synapse0xa82d440() {
   return (neuron0xa829840()*0.320059);
}

double ruNNEE::synapse0xa82d480() {
   return (neuron0xa829b80()*-0.367003);
}

double ruNNEE::synapse0xa82d800() {
   return (neuron0xa829ff0()*0.182052);
}

double ruNNEE::synapse0xa82d840() {
   return (neuron0xa82a420()*-0.139437);
}

double ruNNEE::synapse0xa82d880() {
   return (neuron0xa82a960()*-0.598619);
}

double ruNNEE::synapse0xa82d8c0() {
   return (neuron0xa82af30()*-1.35137);
}

double ruNNEE::synapse0xa82d900() {
   return (neuron0xa82b470()*-0.272138);
}

double ruNNEE::synapse0xa82d940() {
   return (neuron0xa82b7f0()*-1.22362);
}

double ruNNEE::synapse0xa82d980() {
   return (neuron0xa82bd30()*0.34401);
}

double ruNNEE::synapse0xa82d9c0() {
   return (neuron0xa82c270()*-0.533939);
}

double ruNNEE::synapse0xa82da00() {
   return (neuron0xa82c7b0()*0.477494);
}

double ruNNEE::synapse0xa82da40() {
   return (neuron0xa82d010()*1.42208);
}

double ruNNEE::synapse0xa82ddc0() {
   return (neuron0xa829ff0()*0.0972746);
}

double ruNNEE::synapse0xa82de00() {
   return (neuron0xa82a420()*1.17598);
}

double ruNNEE::synapse0xa82de40() {
   return (neuron0xa82a960()*-0.752091);
}

double ruNNEE::synapse0xa82de80() {
   return (neuron0xa82af30()*-0.744279);
}

double ruNNEE::synapse0xa82dec0() {
   return (neuron0xa82b470()*-0.120409);
}

double ruNNEE::synapse0xa82df00() {
   return (neuron0xa82b7f0()*-0.209274);
}

double ruNNEE::synapse0xa82df40() {
   return (neuron0xa82bd30()*1.09003);
}

double ruNNEE::synapse0xa82df80() {
   return (neuron0xa82c270()*0.333017);
}

double ruNNEE::synapse0xa82dfc0() {
   return (neuron0xa82c7b0()*2.06864);
}

double ruNNEE::synapse0xa82e000() {
   return (neuron0xa82d010()*1.36074);
}

double ruNNEE::synapse0xa82e380() {
   return (neuron0xa829ff0()*0.383234);
}

double ruNNEE::synapse0xa82e3c0() {
   return (neuron0xa82a420()*0.722152);
}

double ruNNEE::synapse0xa82e400() {
   return (neuron0xa82a960()*0.993275);
}

double ruNNEE::synapse0xa82e440() {
   return (neuron0xa82af30()*-0.812513);
}

double ruNNEE::synapse0xa82e480() {
   return (neuron0xa82b470()*0.429425);
}

double ruNNEE::synapse0xa82e4c0() {
   return (neuron0xa82b7f0()*-1.53475);
}

double ruNNEE::synapse0xa82e500() {
   return (neuron0xa82bd30()*-0.122715);
}

double ruNNEE::synapse0xa82e540() {
   return (neuron0xa82c270()*-0.938325);
}

double ruNNEE::synapse0xa82e580() {
   return (neuron0xa82c7b0()*1.7371);
}

double ruNNEE::synapse0xa82e5c0() {
   return (neuron0xa82d010()*2.06163);
}

double ruNNEE::synapse0xa82e940() {
   return (neuron0xa829ff0()*0.454973);
}

double ruNNEE::synapse0xa82e980() {
   return (neuron0xa82a420()*-1.40592);
}

double ruNNEE::synapse0xa82e9c0() {
   return (neuron0xa82a960()*-1.58567);
}

double ruNNEE::synapse0xa82ea00() {
   return (neuron0xa82af30()*-0.880453);
}

double ruNNEE::synapse0xa82ea40() {
   return (neuron0xa82b470()*-0.359768);
}

double ruNNEE::synapse0xa82ea80() {
   return (neuron0xa82b7f0()*-0.733199);
}

double ruNNEE::synapse0xa82eac0() {
   return (neuron0xa82bd30()*0.688791);
}

double ruNNEE::synapse0xa82eb00() {
   return (neuron0xa82c270()*0.6616);
}

double ruNNEE::synapse0xa82eb40() {
   return (neuron0xa82c7b0()*0.878811);
}

double ruNNEE::synapse0xa82eb80() {
   return (neuron0xa82d010()*1.86126);
}

double ruNNEE::synapse0xa82ef00() {
   return (neuron0xa829ff0()*-1.51421);
}

double ruNNEE::synapse0xa82ef40() {
   return (neuron0xa82a420()*1.22434);
}

double ruNNEE::synapse0xa82ef80() {
   return (neuron0xa82a960()*1.51179);
}

double ruNNEE::synapse0xa82efc0() {
   return (neuron0xa82af30()*1.54355);
}

double ruNNEE::synapse0xa82f000() {
   return (neuron0xa82b470()*1.36086);
}

double ruNNEE::synapse0xa82f040() {
   return (neuron0xa82b7f0()*0.500821);
}

double ruNNEE::synapse0xa82f080() {
   return (neuron0xa82bd30()*1.77085);
}

double ruNNEE::synapse0xa82f0c0() {
   return (neuron0xa82c270()*-1.79602);
}

double ruNNEE::synapse0xa82f100() {
   return (neuron0xa82c7b0()*1.15052);
}

double ruNNEE::synapse0xa82cc00() {
   return (neuron0xa82d010()*-1.15544);
}

double ruNNEE::synapse0xa82cf80() {
   return (neuron0xa82d4c0()*1.74873);
}

double ruNNEE::synapse0xa82cfc0() {
   return (neuron0xa82da80()*2.23277);
}

double ruNNEE::synapse0xa829ec0() {
   return (neuron0xa82e040()*3.82441);
}

double ruNNEE::synapse0xa829f00() {
   return (neuron0xa82e600()*1.89862);
}

double ruNNEE::synapse0xa829f40() {
   return (neuron0xa82ebc0()*-5.32595);
}

