#include "luNNEE.h"
#include <cmath>

double luNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.40353)/1.40003;
   input2 = (in2 - 2.41121)/1.41004;
   input3 = (in3 - 2.42657)/1.40629;
   input4 = (in4 - 2.42619)/1.40466;
   input5 = (in5 - 1.33856)/1.28698;
   input6 = (in6 - 1.33177)/1.28879;
   input7 = (in7 - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x1d243c80();
     default:
         return 0.;
   }
}

double luNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.40353)/1.40003;
   input2 = (input[2] - 2.41121)/1.41004;
   input3 = (input[3] - 2.42657)/1.40629;
   input4 = (input[4] - 2.42619)/1.40466;
   input5 = (input[5] - 1.33856)/1.28698;
   input6 = (input[6] - 1.33177)/1.28879;
   input7 = (input[7] - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x1d243c80();
     default:
         return 0.;
   }
}

double luNNEE::neuron0x1d23f500() {
   return input0;
}

double luNNEE::neuron0x1d23f840() {
   return input1;
}

double luNNEE::neuron0x1d23fb80() {
   return input2;
}

double luNNEE::neuron0x1d23fec0() {
   return input3;
}

double luNNEE::neuron0x1d240200() {
   return input4;
}

double luNNEE::neuron0x1d240540() {
   return input5;
}

double luNNEE::neuron0x1d240880() {
   return input6;
}

double luNNEE::neuron0x1d240bc0() {
   return input7;
}

double luNNEE::input0x1d241030() {
   double input = 1.882;
   input += synapse0x1d19f840();
   input += synapse0x1d248100();
   input += synapse0x1d2412e0();
   input += synapse0x1d241320();
   input += synapse0x1d241360();
   input += synapse0x1d2413a0();
   input += synapse0x1d2413e0();
   input += synapse0x1d241420();
   return input;
}

double luNNEE::neuron0x1d241030() {
   double input = input0x1d241030();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d241460() {
   double input = 1.23049;
   input += synapse0x1d2417a0();
   input += synapse0x1d2417e0();
   input += synapse0x1d241820();
   input += synapse0x1d241860();
   input += synapse0x1d2418a0();
   input += synapse0x1d2418e0();
   input += synapse0x1d241920();
   input += synapse0x1d241960();
   return input;
}

double luNNEE::neuron0x1d241460() {
   double input = input0x1d241460();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d2419a0() {
   double input = 0.649317;
   input += synapse0x1d241ce0();
   input += synapse0x1d16dde0();
   input += synapse0x1d16de20();
   input += synapse0x1d241e30();
   input += synapse0x1d241e70();
   input += synapse0x1d241eb0();
   input += synapse0x1d241ef0();
   input += synapse0x1d241f30();
   return input;
}

double luNNEE::neuron0x1d2419a0() {
   double input = input0x1d2419a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d241f70() {
   double input = -0.288646;
   input += synapse0x1d2422b0();
   input += synapse0x1d2422f0();
   input += synapse0x1d242330();
   input += synapse0x1d242370();
   input += synapse0x1d2423b0();
   input += synapse0x1d2423f0();
   input += synapse0x1d242430();
   input += synapse0x1d242470();
   return input;
}

double luNNEE::neuron0x1d241f70() {
   double input = input0x1d241f70();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d2424b0() {
   double input = 1.72265;
   input += synapse0x1d2427f0();
   input += synapse0x1d23f430();
   input += synapse0x1d248140();
   input += synapse0x1d18a150();
   input += synapse0x1d241d20();
   input += synapse0x1d241d60();
   input += synapse0x1d241da0();
   input += synapse0x1d241de0();
   return input;
}

double luNNEE::neuron0x1d2424b0() {
   double input = input0x1d2424b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d242830() {
   double input = 0.45399;
   input += synapse0x1d242b70();
   input += synapse0x1d242bb0();
   input += synapse0x1d242bf0();
   input += synapse0x1d242c30();
   input += synapse0x1d242c70();
   input += synapse0x1d242cb0();
   input += synapse0x1d242cf0();
   input += synapse0x1d242d30();
   return input;
}

double luNNEE::neuron0x1d242830() {
   double input = input0x1d242830();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d242d70() {
   double input = 1.79327;
   input += synapse0x1d2430b0();
   input += synapse0x1d2430f0();
   input += synapse0x1d243130();
   input += synapse0x1d243170();
   input += synapse0x1d2431b0();
   input += synapse0x1d2431f0();
   input += synapse0x1d243230();
   input += synapse0x1d243270();
   return input;
}

double luNNEE::neuron0x1d242d70() {
   double input = input0x1d242d70();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d2432b0() {
   double input = -1.3481;
   input += synapse0x1d2435f0();
   input += synapse0x1d243630();
   input += synapse0x1d243670();
   input += synapse0x1d2436b0();
   input += synapse0x1d2436f0();
   input += synapse0x1d243730();
   input += synapse0x1d243770();
   input += synapse0x1d2437b0();
   return input;
}

double luNNEE::neuron0x1d2432b0() {
   double input = input0x1d2432b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d2437f0() {
   double input = 2.24534;
   input += synapse0x1d16ba40();
   input += synapse0x1d16ba80();
   input += synapse0x1d186940();
   input += synapse0x1d186980();
   input += synapse0x1d1869c0();
   input += synapse0x1d186a00();
   input += synapse0x1d186a40();
   input += synapse0x1d186a80();
   return input;
}

double luNNEE::neuron0x1d2437f0() {
   double input = input0x1d2437f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d244050() {
   double input = -1.10491;
   input += synapse0x1d244300();
   input += synapse0x1d244340();
   input += synapse0x1d244380();
   input += synapse0x1d2443c0();
   input += synapse0x1d244400();
   input += synapse0x1d244440();
   input += synapse0x1d244480();
   input += synapse0x1d2444c0();
   return input;
}

double luNNEE::neuron0x1d244050() {
   double input = input0x1d244050();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d244500() {
   double input = 0.086249;
   input += synapse0x1d244840();
   input += synapse0x1d244880();
   input += synapse0x1d2448c0();
   input += synapse0x1d244900();
   input += synapse0x1d244940();
   input += synapse0x1d244980();
   input += synapse0x1d2449c0();
   input += synapse0x1d244a00();
   input += synapse0x1d244a40();
   input += synapse0x1d244a80();
   return input;
}

double luNNEE::neuron0x1d244500() {
   double input = input0x1d244500();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d244ac0() {
   double input = 0.536111;
   input += synapse0x1d244e00();
   input += synapse0x1d244e40();
   input += synapse0x1d244e80();
   input += synapse0x1d244ec0();
   input += synapse0x1d244f00();
   input += synapse0x1d244f40();
   input += synapse0x1d244f80();
   input += synapse0x1d244fc0();
   input += synapse0x1d245000();
   input += synapse0x1d245040();
   return input;
}

double luNNEE::neuron0x1d244ac0() {
   double input = input0x1d244ac0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d245080() {
   double input = 1.05668;
   input += synapse0x1d2453c0();
   input += synapse0x1d245400();
   input += synapse0x1d245440();
   input += synapse0x1d245480();
   input += synapse0x1d2454c0();
   input += synapse0x1d245500();
   input += synapse0x1d245540();
   input += synapse0x1d245580();
   input += synapse0x1d2455c0();
   input += synapse0x1d245600();
   return input;
}

double luNNEE::neuron0x1d245080() {
   double input = input0x1d245080();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d245640() {
   double input = 0.63205;
   input += synapse0x1d245980();
   input += synapse0x1d2459c0();
   input += synapse0x1d245a00();
   input += synapse0x1d245a40();
   input += synapse0x1d245a80();
   input += synapse0x1d245ac0();
   input += synapse0x1d245b00();
   input += synapse0x1d245b40();
   input += synapse0x1d245b80();
   input += synapse0x1d245bc0();
   return input;
}

double luNNEE::neuron0x1d245640() {
   double input = input0x1d245640();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d245c00() {
   double input = 0.23401;
   input += synapse0x1d245f40();
   input += synapse0x1d245f80();
   input += synapse0x1d245fc0();
   input += synapse0x1d246000();
   input += synapse0x1d246040();
   input += synapse0x1d246080();
   input += synapse0x1d2460c0();
   input += synapse0x1d246100();
   input += synapse0x1d246140();
   input += synapse0x1d243c40();
   return input;
}

double luNNEE::neuron0x1d245c00() {
   double input = input0x1d245c00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double luNNEE::input0x1d243c80() {
   double input = -1.22764;
   input += synapse0x1d243fc0();
   input += synapse0x1d244000();
   input += synapse0x1d240f00();
   input += synapse0x1d240f40();
   input += synapse0x1d240f80();
   return input;
}

double luNNEE::neuron0x1d243c80() {
   double input = input0x1d243c80();
   return (input * 1)+0;
}

double luNNEE::synapse0x1d19f840() {
   return (neuron0x1d23f500()*-0.92939);
}

double luNNEE::synapse0x1d248100() {
   return (neuron0x1d23f840()*0.320184);
}

double luNNEE::synapse0x1d2412e0() {
   return (neuron0x1d23fb80()*0.932357);
}

double luNNEE::synapse0x1d241320() {
   return (neuron0x1d23fec0()*-0.374119);
}

double luNNEE::synapse0x1d241360() {
   return (neuron0x1d240200()*-0.384557);
}

double luNNEE::synapse0x1d2413a0() {
   return (neuron0x1d240540()*0.753054);
}

double luNNEE::synapse0x1d2413e0() {
   return (neuron0x1d240880()*-0.36931);
}

double luNNEE::synapse0x1d241420() {
   return (neuron0x1d240bc0()*0.984306);
}

double luNNEE::synapse0x1d2417a0() {
   return (neuron0x1d23f500()*0.990952);
}

double luNNEE::synapse0x1d2417e0() {
   return (neuron0x1d23f840()*0.323254);
}

double luNNEE::synapse0x1d241820() {
   return (neuron0x1d23fb80()*0.242277);
}

double luNNEE::synapse0x1d241860() {
   return (neuron0x1d23fec0()*0.1361);
}

double luNNEE::synapse0x1d2418a0() {
   return (neuron0x1d240200()*0.19794);
}

double luNNEE::synapse0x1d2418e0() {
   return (neuron0x1d240540()*0.200274);
}

double luNNEE::synapse0x1d241920() {
   return (neuron0x1d240880()*-0.177756);
}

double luNNEE::synapse0x1d241960() {
   return (neuron0x1d240bc0()*0.210928);
}

double luNNEE::synapse0x1d241ce0() {
   return (neuron0x1d23f500()*0.663276);
}

double luNNEE::synapse0x1d16dde0() {
   return (neuron0x1d23f840()*0.198227);
}

double luNNEE::synapse0x1d16de20() {
   return (neuron0x1d23fb80()*-0.526768);
}

double luNNEE::synapse0x1d241e30() {
   return (neuron0x1d23fec0()*-0.59923);
}

double luNNEE::synapse0x1d241e70() {
   return (neuron0x1d240200()*0.152005);
}

double luNNEE::synapse0x1d241eb0() {
   return (neuron0x1d240540()*0.419226);
}

double luNNEE::synapse0x1d241ef0() {
   return (neuron0x1d240880()*-0.106808);
}

double luNNEE::synapse0x1d241f30() {
   return (neuron0x1d240bc0()*0.340483);
}

double luNNEE::synapse0x1d2422b0() {
   return (neuron0x1d23f500()*-1.26401);
}

double luNNEE::synapse0x1d2422f0() {
   return (neuron0x1d23f840()*-0.435347);
}

double luNNEE::synapse0x1d242330() {
   return (neuron0x1d23fb80()*-0.0244965);
}

double luNNEE::synapse0x1d242370() {
   return (neuron0x1d23fec0()*-0.726778);
}

double luNNEE::synapse0x1d2423b0() {
   return (neuron0x1d240200()*-0.0550666);
}

double luNNEE::synapse0x1d2423f0() {
   return (neuron0x1d240540()*-0.274861);
}

double luNNEE::synapse0x1d242430() {
   return (neuron0x1d240880()*-0.723293);
}

double luNNEE::synapse0x1d242470() {
   return (neuron0x1d240bc0()*-1.18553);
}

double luNNEE::synapse0x1d2427f0() {
   return (neuron0x1d23f500()*-1.09586);
}

double luNNEE::synapse0x1d23f430() {
   return (neuron0x1d23f840()*0.60971);
}

double luNNEE::synapse0x1d248140() {
   return (neuron0x1d23fb80()*-0.31137);
}

double luNNEE::synapse0x1d18a150() {
   return (neuron0x1d23fec0()*-0.711958);
}

double luNNEE::synapse0x1d241d20() {
   return (neuron0x1d240200()*0.487486);
}

double luNNEE::synapse0x1d241d60() {
   return (neuron0x1d240540()*0.0659214);
}

double luNNEE::synapse0x1d241da0() {
   return (neuron0x1d240880()*0.157566);
}

double luNNEE::synapse0x1d241de0() {
   return (neuron0x1d240bc0()*0.172181);
}

double luNNEE::synapse0x1d242b70() {
   return (neuron0x1d23f500()*0.535937);
}

double luNNEE::synapse0x1d242bb0() {
   return (neuron0x1d23f840()*-0.0465596);
}

double luNNEE::synapse0x1d242bf0() {
   return (neuron0x1d23fb80()*0.643086);
}

double luNNEE::synapse0x1d242c30() {
   return (neuron0x1d23fec0()*1.17476);
}

double luNNEE::synapse0x1d242c70() {
   return (neuron0x1d240200()*0.288662);
}

double luNNEE::synapse0x1d242cb0() {
   return (neuron0x1d240540()*-0.641959);
}

double luNNEE::synapse0x1d242cf0() {
   return (neuron0x1d240880()*0.537438);
}

double luNNEE::synapse0x1d242d30() {
   return (neuron0x1d240bc0()*-2.73611);
}

double luNNEE::synapse0x1d2430b0() {
   return (neuron0x1d23f500()*-0.366689);
}

double luNNEE::synapse0x1d2430f0() {
   return (neuron0x1d23f840()*-0.183007);
}

double luNNEE::synapse0x1d243130() {
   return (neuron0x1d23fb80()*-0.380738);
}

double luNNEE::synapse0x1d243170() {
   return (neuron0x1d23fec0()*-0.00161081);
}

double luNNEE::synapse0x1d2431b0() {
   return (neuron0x1d240200()*-0.18148);
}

double luNNEE::synapse0x1d2431f0() {
   return (neuron0x1d240540()*-0.557646);
}

double luNNEE::synapse0x1d243230() {
   return (neuron0x1d240880()*0.042859);
}

double luNNEE::synapse0x1d243270() {
   return (neuron0x1d240bc0()*-0.349465);
}

double luNNEE::synapse0x1d2435f0() {
   return (neuron0x1d23f500()*1.49823);
}

double luNNEE::synapse0x1d243630() {
   return (neuron0x1d23f840()*-1.15228);
}

double luNNEE::synapse0x1d243670() {
   return (neuron0x1d23fb80()*0.381668);
}

double luNNEE::synapse0x1d2436b0() {
   return (neuron0x1d23fec0()*0.363835);
}

double luNNEE::synapse0x1d2436f0() {
   return (neuron0x1d240200()*-1.34539);
}

double luNNEE::synapse0x1d243730() {
   return (neuron0x1d240540()*0.563932);
}

double luNNEE::synapse0x1d243770() {
   return (neuron0x1d240880()*-0.418145);
}

double luNNEE::synapse0x1d2437b0() {
   return (neuron0x1d240bc0()*0.480876);
}

double luNNEE::synapse0x1d16ba40() {
   return (neuron0x1d23f500()*-0.606071);
}

double luNNEE::synapse0x1d16ba80() {
   return (neuron0x1d23f840()*-1.29562);
}

double luNNEE::synapse0x1d186940() {
   return (neuron0x1d23fb80()*-1.32919);
}

double luNNEE::synapse0x1d186980() {
   return (neuron0x1d23fec0()*2.38038);
}

double luNNEE::synapse0x1d1869c0() {
   return (neuron0x1d240200()*1.1447);
}

double luNNEE::synapse0x1d186a00() {
   return (neuron0x1d240540()*0.428176);
}

double luNNEE::synapse0x1d186a40() {
   return (neuron0x1d240880()*-0.0504862);
}

double luNNEE::synapse0x1d186a80() {
   return (neuron0x1d240bc0()*-0.429001);
}

double luNNEE::synapse0x1d244300() {
   return (neuron0x1d23f500()*-0.516443);
}

double luNNEE::synapse0x1d244340() {
   return (neuron0x1d23f840()*-0.156578);
}

double luNNEE::synapse0x1d244380() {
   return (neuron0x1d23fb80()*-1.24439);
}

double luNNEE::synapse0x1d2443c0() {
   return (neuron0x1d23fec0()*-0.0514229);
}

double luNNEE::synapse0x1d244400() {
   return (neuron0x1d240200()*0.443304);
}

double luNNEE::synapse0x1d244440() {
   return (neuron0x1d240540()*1.62769);
}

double luNNEE::synapse0x1d244480() {
   return (neuron0x1d240880()*-0.374873);
}

double luNNEE::synapse0x1d2444c0() {
   return (neuron0x1d240bc0()*0.560535);
}

double luNNEE::synapse0x1d244840() {
   return (neuron0x1d241030()*0.767992);
}

double luNNEE::synapse0x1d244880() {
   return (neuron0x1d241460()*0.50302);
}

double luNNEE::synapse0x1d2448c0() {
   return (neuron0x1d2419a0()*-0.472644);
}

double luNNEE::synapse0x1d244900() {
   return (neuron0x1d241f70()*0.806902);
}

double luNNEE::synapse0x1d244940() {
   return (neuron0x1d2424b0()*-1.81099);
}

double luNNEE::synapse0x1d244980() {
   return (neuron0x1d242830()*-0.458441);
}

double luNNEE::synapse0x1d2449c0() {
   return (neuron0x1d242d70()*-1.38208);
}

double luNNEE::synapse0x1d244a00() {
   return (neuron0x1d2432b0()*-0.310336);
}

double luNNEE::synapse0x1d244a40() {
   return (neuron0x1d2437f0()*-0.324322);
}

double luNNEE::synapse0x1d244a80() {
   return (neuron0x1d244050()*0.0540683);
}

double luNNEE::synapse0x1d244e00() {
   return (neuron0x1d241030()*0.786979);
}

double luNNEE::synapse0x1d244e40() {
   return (neuron0x1d241460()*1.0236);
}

double luNNEE::synapse0x1d244e80() {
   return (neuron0x1d2419a0()*-0.8535);
}

double luNNEE::synapse0x1d244ec0() {
   return (neuron0x1d241f70()*-0.215959);
}

double luNNEE::synapse0x1d244f00() {
   return (neuron0x1d2424b0()*-0.854814);
}

double luNNEE::synapse0x1d244f40() {
   return (neuron0x1d242830()*-0.108652);
}

double luNNEE::synapse0x1d244f80() {
   return (neuron0x1d242d70()*-0.943531);
}

double luNNEE::synapse0x1d244fc0() {
   return (neuron0x1d2432b0()*0.509758);
}

double luNNEE::synapse0x1d245000() {
   return (neuron0x1d2437f0()*0.0385086);
}

double luNNEE::synapse0x1d245040() {
   return (neuron0x1d244050()*0.296752);
}

double luNNEE::synapse0x1d2453c0() {
   return (neuron0x1d241030()*0.600361);
}

double luNNEE::synapse0x1d245400() {
   return (neuron0x1d241460()*2.23698);
}

double luNNEE::synapse0x1d245440() {
   return (neuron0x1d2419a0()*-1.06472);
}

double luNNEE::synapse0x1d245480() {
   return (neuron0x1d241f70()*0.0920412);
}

double luNNEE::synapse0x1d2454c0() {
   return (neuron0x1d2424b0()*-0.834027);
}

double luNNEE::synapse0x1d245500() {
   return (neuron0x1d242830()*0.75298);
}

double luNNEE::synapse0x1d245540() {
   return (neuron0x1d242d70()*-0.396142);
}

double luNNEE::synapse0x1d245580() {
   return (neuron0x1d2432b0()*0.892455);
}

double luNNEE::synapse0x1d2455c0() {
   return (neuron0x1d2437f0()*0.968262);
}

double luNNEE::synapse0x1d245600() {
   return (neuron0x1d244050()*-0.709875);
}

double luNNEE::synapse0x1d245980() {
   return (neuron0x1d241030()*-2.51911);
}

double luNNEE::synapse0x1d2459c0() {
   return (neuron0x1d241460()*1.31583);
}

double luNNEE::synapse0x1d245a00() {
   return (neuron0x1d2419a0()*2.66054);
}

double luNNEE::synapse0x1d245a40() {
   return (neuron0x1d241f70()*1.22613);
}

double luNNEE::synapse0x1d245a80() {
   return (neuron0x1d2424b0()*-0.0846529);
}

double luNNEE::synapse0x1d245ac0() {
   return (neuron0x1d242830()*-1.80604);
}

double luNNEE::synapse0x1d245b00() {
   return (neuron0x1d242d70()*1.2746);
}

double luNNEE::synapse0x1d245b40() {
   return (neuron0x1d2432b0()*3.04159);
}

double luNNEE::synapse0x1d245b80() {
   return (neuron0x1d2437f0()*-2.38866);
}

double luNNEE::synapse0x1d245bc0() {
   return (neuron0x1d244050()*2.92037);
}

double luNNEE::synapse0x1d245f40() {
   return (neuron0x1d241030()*0.467129);
}

double luNNEE::synapse0x1d245f80() {
   return (neuron0x1d241460()*0.933566);
}

double luNNEE::synapse0x1d245fc0() {
   return (neuron0x1d2419a0()*-0.411388);
}

double luNNEE::synapse0x1d246000() {
   return (neuron0x1d241f70()*-0.254523);
}

double luNNEE::synapse0x1d246040() {
   return (neuron0x1d2424b0()*-1.34352);
}

double luNNEE::synapse0x1d246080() {
   return (neuron0x1d242830()*0.155716);
}

double luNNEE::synapse0x1d2460c0() {
   return (neuron0x1d242d70()*-0.833839);
}

double luNNEE::synapse0x1d246100() {
   return (neuron0x1d2432b0()*0.113311);
}

double luNNEE::synapse0x1d246140() {
   return (neuron0x1d2437f0()*-0.465655);
}

double luNNEE::synapse0x1d243c40() {
   return (neuron0x1d244050()*0.0637556);
}

double luNNEE::synapse0x1d243fc0() {
   return (neuron0x1d244500()*3.50251);
}

double luNNEE::synapse0x1d244000() {
   return (neuron0x1d244ac0()*2.51474);
}

double luNNEE::synapse0x1d240f00() {
   return (neuron0x1d245080()*2.96117);
}

double luNNEE::synapse0x1d240f40() {
   return (neuron0x1d245640()*-3.30225);
}

double luNNEE::synapse0x1d240f80() {
   return (neuron0x1d245c00()*2.72949);
}

