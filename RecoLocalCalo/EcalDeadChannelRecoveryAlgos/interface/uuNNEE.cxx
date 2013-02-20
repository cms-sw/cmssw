#include "uuNNEE.h"
#include <cmath>

double uuNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.40353)/1.40003;
   input2 = (in2 - 2.41121)/1.41004;
   input3 = (in3 - 2.42619)/1.40466;
   input4 = (in4 - 1.33856)/1.28698;
   input5 = (in5 - 1.33177)/1.28879;
   input6 = (in6 - 1.33367)/1.29347;
   input7 = (in7 - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x1e0e4b40();
     default:
         return 0.;
   }
}

double uuNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.40353)/1.40003;
   input2 = (input[2] - 2.41121)/1.41004;
   input3 = (input[3] - 2.42619)/1.40466;
   input4 = (input[4] - 1.33856)/1.28698;
   input5 = (input[5] - 1.33177)/1.28879;
   input6 = (input[6] - 1.33367)/1.29347;
   input7 = (input[7] - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x1e0e4b40();
     default:
         return 0.;
   }
}

double uuNNEE::neuron0x1e0e03c0() {
   return input0;
}

double uuNNEE::neuron0x1e0e0700() {
   return input1;
}

double uuNNEE::neuron0x1e0e0a40() {
   return input2;
}

double uuNNEE::neuron0x1e0e0d80() {
   return input3;
}

double uuNNEE::neuron0x1e0e10c0() {
   return input4;
}

double uuNNEE::neuron0x1e0e1400() {
   return input5;
}

double uuNNEE::neuron0x1e0e1740() {
   return input6;
}

double uuNNEE::neuron0x1e0e1a80() {
   return input7;
}

double uuNNEE::input0x1e0e1ef0() {
   double input = 0.132124;
   input += synapse0x1e040700();
   input += synapse0x1e0e8fc0();
   input += synapse0x1e0e21a0();
   input += synapse0x1e0e21e0();
   input += synapse0x1e0e2220();
   input += synapse0x1e0e2260();
   input += synapse0x1e0e22a0();
   input += synapse0x1e0e22e0();
   return input;
}

double uuNNEE::neuron0x1e0e1ef0() {
   double input = input0x1e0e1ef0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e2320() {
   double input = -1.56833;
   input += synapse0x1e0e2660();
   input += synapse0x1e0e26a0();
   input += synapse0x1e0e26e0();
   input += synapse0x1e0e2720();
   input += synapse0x1e0e2760();
   input += synapse0x1e0e27a0();
   input += synapse0x1e0e27e0();
   input += synapse0x1e0e2820();
   return input;
}

double uuNNEE::neuron0x1e0e2320() {
   double input = input0x1e0e2320();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e2860() {
   double input = 2.47862;
   input += synapse0x1e0e2ba0();
   input += synapse0x1e00eca0();
   input += synapse0x1e00ece0();
   input += synapse0x1e0e2cf0();
   input += synapse0x1e0e2d30();
   input += synapse0x1e0e2d70();
   input += synapse0x1e0e2db0();
   input += synapse0x1e0e2df0();
   return input;
}

double uuNNEE::neuron0x1e0e2860() {
   double input = input0x1e0e2860();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e2e30() {
   double input = -0.411867;
   input += synapse0x1e0e3170();
   input += synapse0x1e0e31b0();
   input += synapse0x1e0e31f0();
   input += synapse0x1e0e3230();
   input += synapse0x1e0e3270();
   input += synapse0x1e0e32b0();
   input += synapse0x1e0e32f0();
   input += synapse0x1e0e3330();
   return input;
}

double uuNNEE::neuron0x1e0e2e30() {
   double input = input0x1e0e2e30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e3370() {
   double input = -0.466911;
   input += synapse0x1e0e36b0();
   input += synapse0x1e0e02f0();
   input += synapse0x1e0e9000();
   input += synapse0x1e02b010();
   input += synapse0x1e0e2be0();
   input += synapse0x1e0e2c20();
   input += synapse0x1e0e2c60();
   input += synapse0x1e0e2ca0();
   return input;
}

double uuNNEE::neuron0x1e0e3370() {
   double input = input0x1e0e3370();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e36f0() {
   double input = 1.76238;
   input += synapse0x1e0e3a30();
   input += synapse0x1e0e3a70();
   input += synapse0x1e0e3ab0();
   input += synapse0x1e0e3af0();
   input += synapse0x1e0e3b30();
   input += synapse0x1e0e3b70();
   input += synapse0x1e0e3bb0();
   input += synapse0x1e0e3bf0();
   return input;
}

double uuNNEE::neuron0x1e0e36f0() {
   double input = input0x1e0e36f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e3c30() {
   double input = 0.269915;
   input += synapse0x1e0e3f70();
   input += synapse0x1e0e3fb0();
   input += synapse0x1e0e3ff0();
   input += synapse0x1e0e4030();
   input += synapse0x1e0e4070();
   input += synapse0x1e0e40b0();
   input += synapse0x1e0e40f0();
   input += synapse0x1e0e4130();
   return input;
}

double uuNNEE::neuron0x1e0e3c30() {
   double input = input0x1e0e3c30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e4170() {
   double input = -1.28328;
   input += synapse0x1e0e44b0();
   input += synapse0x1e0e44f0();
   input += synapse0x1e0e4530();
   input += synapse0x1e0e4570();
   input += synapse0x1e0e45b0();
   input += synapse0x1e0e45f0();
   input += synapse0x1e0e4630();
   input += synapse0x1e0e4670();
   return input;
}

double uuNNEE::neuron0x1e0e4170() {
   double input = input0x1e0e4170();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e46b0() {
   double input = 0.991379;
   input += synapse0x1e00c900();
   input += synapse0x1e00c940();
   input += synapse0x1e027800();
   input += synapse0x1e027840();
   input += synapse0x1e027880();
   input += synapse0x1e0278c0();
   input += synapse0x1e027900();
   input += synapse0x1e027940();
   return input;
}

double uuNNEE::neuron0x1e0e46b0() {
   double input = input0x1e0e46b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e4f10() {
   double input = -0.40117;
   input += synapse0x1e0e51c0();
   input += synapse0x1e0e5200();
   input += synapse0x1e0e5240();
   input += synapse0x1e0e5280();
   input += synapse0x1e0e52c0();
   input += synapse0x1e0e5300();
   input += synapse0x1e0e5340();
   input += synapse0x1e0e5380();
   return input;
}

double uuNNEE::neuron0x1e0e4f10() {
   double input = input0x1e0e4f10();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e53c0() {
   double input = -2.43098;
   input += synapse0x1e0e5700();
   input += synapse0x1e0e5740();
   input += synapse0x1e0e5780();
   input += synapse0x1e0e57c0();
   input += synapse0x1e0e5800();
   input += synapse0x1e0e5840();
   input += synapse0x1e0e5880();
   input += synapse0x1e0e58c0();
   input += synapse0x1e0e5900();
   input += synapse0x1e0e5940();
   return input;
}

double uuNNEE::neuron0x1e0e53c0() {
   double input = input0x1e0e53c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e5980() {
   double input = -0.49089;
   input += synapse0x1e0e5cc0();
   input += synapse0x1e0e5d00();
   input += synapse0x1e0e5d40();
   input += synapse0x1e0e5d80();
   input += synapse0x1e0e5dc0();
   input += synapse0x1e0e5e00();
   input += synapse0x1e0e5e40();
   input += synapse0x1e0e5e80();
   input += synapse0x1e0e5ec0();
   input += synapse0x1e0e5f00();
   return input;
}

double uuNNEE::neuron0x1e0e5980() {
   double input = input0x1e0e5980();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e5f40() {
   double input = -1.40155;
   input += synapse0x1e0e6280();
   input += synapse0x1e0e62c0();
   input += synapse0x1e0e6300();
   input += synapse0x1e0e6340();
   input += synapse0x1e0e6380();
   input += synapse0x1e0e63c0();
   input += synapse0x1e0e6400();
   input += synapse0x1e0e6440();
   input += synapse0x1e0e6480();
   input += synapse0x1e0e64c0();
   return input;
}

double uuNNEE::neuron0x1e0e5f40() {
   double input = input0x1e0e5f40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e6500() {
   double input = 0.856813;
   input += synapse0x1e0e6840();
   input += synapse0x1e0e6880();
   input += synapse0x1e0e68c0();
   input += synapse0x1e0e6900();
   input += synapse0x1e0e6940();
   input += synapse0x1e0e6980();
   input += synapse0x1e0e69c0();
   input += synapse0x1e0e6a00();
   input += synapse0x1e0e6a40();
   input += synapse0x1e0e6a80();
   return input;
}

double uuNNEE::neuron0x1e0e6500() {
   double input = input0x1e0e6500();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e6ac0() {
   double input = -2.19246;
   input += synapse0x1e0e6e00();
   input += synapse0x1e0e6e40();
   input += synapse0x1e0e6e80();
   input += synapse0x1e0e6ec0();
   input += synapse0x1e0e6f00();
   input += synapse0x1e0e6f40();
   input += synapse0x1e0e6f80();
   input += synapse0x1e0e6fc0();
   input += synapse0x1e0e7000();
   input += synapse0x1e0e4b00();
   return input;
}

double uuNNEE::neuron0x1e0e6ac0() {
   double input = input0x1e0e6ac0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double uuNNEE::input0x1e0e4b40() {
   double input = 1.19185;
   input += synapse0x1e0e4e80();
   input += synapse0x1e0e4ec0();
   input += synapse0x1e0e1dc0();
   input += synapse0x1e0e1e00();
   input += synapse0x1e0e1e40();
   return input;
}

double uuNNEE::neuron0x1e0e4b40() {
   double input = input0x1e0e4b40();
   return (input * 1)+0;
}

double uuNNEE::synapse0x1e040700() {
   return (neuron0x1e0e03c0()*-1.62498);
}

double uuNNEE::synapse0x1e0e8fc0() {
   return (neuron0x1e0e0700()*-1.12984);
}

double uuNNEE::synapse0x1e0e21a0() {
   return (neuron0x1e0e0a40()*-1.65077);
}

double uuNNEE::synapse0x1e0e21e0() {
   return (neuron0x1e0e0d80()*1.10939);
}

double uuNNEE::synapse0x1e0e2220() {
   return (neuron0x1e0e10c0()*-0.662375);
}

double uuNNEE::synapse0x1e0e2260() {
   return (neuron0x1e0e1400()*0.722792);
}

double uuNNEE::synapse0x1e0e22a0() {
   return (neuron0x1e0e1740()*2.60651);
}

double uuNNEE::synapse0x1e0e22e0() {
   return (neuron0x1e0e1a80()*0.205855);
}

double uuNNEE::synapse0x1e0e2660() {
   return (neuron0x1e0e03c0()*0.867588);
}

double uuNNEE::synapse0x1e0e26a0() {
   return (neuron0x1e0e0700()*0.986362);
}

double uuNNEE::synapse0x1e0e26e0() {
   return (neuron0x1e0e0a40()*0.215262);
}

double uuNNEE::synapse0x1e0e2720() {
   return (neuron0x1e0e0d80()*-0.367279);
}

double uuNNEE::synapse0x1e0e2760() {
   return (neuron0x1e0e10c0()*-1.55761);
}

double uuNNEE::synapse0x1e0e27a0() {
   return (neuron0x1e0e1400()*-0.685611);
}

double uuNNEE::synapse0x1e0e27e0() {
   return (neuron0x1e0e1740()*0.331329);
}

double uuNNEE::synapse0x1e0e2820() {
   return (neuron0x1e0e1a80()*0.810081);
}

double uuNNEE::synapse0x1e0e2ba0() {
   return (neuron0x1e0e03c0()*1.3436);
}

double uuNNEE::synapse0x1e00eca0() {
   return (neuron0x1e0e0700()*0.733122);
}

double uuNNEE::synapse0x1e00ece0() {
   return (neuron0x1e0e0a40()*-0.413899);
}

double uuNNEE::synapse0x1e0e2cf0() {
   return (neuron0x1e0e0d80()*0.705015);
}

double uuNNEE::synapse0x1e0e2d30() {
   return (neuron0x1e0e10c0()*-0.95841);
}

double uuNNEE::synapse0x1e0e2d70() {
   return (neuron0x1e0e1400()*-0.2698);
}

double uuNNEE::synapse0x1e0e2db0() {
   return (neuron0x1e0e1740()*0.00991537);
}

double uuNNEE::synapse0x1e0e2df0() {
   return (neuron0x1e0e1a80()*-0.176317);
}

double uuNNEE::synapse0x1e0e3170() {
   return (neuron0x1e0e03c0()*0.605556);
}

double uuNNEE::synapse0x1e0e31b0() {
   return (neuron0x1e0e0700()*-0.0700405);
}

double uuNNEE::synapse0x1e0e31f0() {
   return (neuron0x1e0e0a40()*0.643331);
}

double uuNNEE::synapse0x1e0e3230() {
   return (neuron0x1e0e0d80()*0.0630598);
}

double uuNNEE::synapse0x1e0e3270() {
   return (neuron0x1e0e10c0()*0.649561);
}

double uuNNEE::synapse0x1e0e32b0() {
   return (neuron0x1e0e1400()*0.157865);
}

double uuNNEE::synapse0x1e0e32f0() {
   return (neuron0x1e0e1740()*-0.780671);
}

double uuNNEE::synapse0x1e0e3330() {
   return (neuron0x1e0e1a80()*-0.252903);
}

double uuNNEE::synapse0x1e0e36b0() {
   return (neuron0x1e0e03c0()*-0.913091);
}

double uuNNEE::synapse0x1e0e02f0() {
   return (neuron0x1e0e0700()*-0.177423);
}

double uuNNEE::synapse0x1e0e9000() {
   return (neuron0x1e0e0a40()*2.24212);
}

double uuNNEE::synapse0x1e02b010() {
   return (neuron0x1e0e0d80()*0.458241);
}

double uuNNEE::synapse0x1e0e2be0() {
   return (neuron0x1e0e10c0()*0.57807);
}

double uuNNEE::synapse0x1e0e2c20() {
   return (neuron0x1e0e1400()*-0.0153282);
}

double uuNNEE::synapse0x1e0e2c60() {
   return (neuron0x1e0e1740()*-1.62724);
}

double uuNNEE::synapse0x1e0e2ca0() {
   return (neuron0x1e0e1a80()*-0.128059);
}

double uuNNEE::synapse0x1e0e3a30() {
   return (neuron0x1e0e03c0()*0.307401);
}

double uuNNEE::synapse0x1e0e3a70() {
   return (neuron0x1e0e0700()*-0.0846233);
}

double uuNNEE::synapse0x1e0e3ab0() {
   return (neuron0x1e0e0a40()*-0.796274);
}

double uuNNEE::synapse0x1e0e3af0() {
   return (neuron0x1e0e0d80()*1.84915);
}

double uuNNEE::synapse0x1e0e3b30() {
   return (neuron0x1e0e10c0()*-0.293791);
}

double uuNNEE::synapse0x1e0e3b70() {
   return (neuron0x1e0e1400()*-0.535142);
}

double uuNNEE::synapse0x1e0e3bb0() {
   return (neuron0x1e0e1740()*0.282696);
}

double uuNNEE::synapse0x1e0e3bf0() {
   return (neuron0x1e0e1a80()*-0.43494);
}

double uuNNEE::synapse0x1e0e3f70() {
   return (neuron0x1e0e03c0()*0.641573);
}

double uuNNEE::synapse0x1e0e3fb0() {
   return (neuron0x1e0e0700()*-2.2047);
}

double uuNNEE::synapse0x1e0e3ff0() {
   return (neuron0x1e0e0a40()*0.411344);
}

double uuNNEE::synapse0x1e0e4030() {
   return (neuron0x1e0e0d80()*-0.259468);
}

double uuNNEE::synapse0x1e0e4070() {
   return (neuron0x1e0e10c0()*1.80947);
}

double uuNNEE::synapse0x1e0e40b0() {
   return (neuron0x1e0e1400()*0.0964665);
}

double uuNNEE::synapse0x1e0e40f0() {
   return (neuron0x1e0e1740()*-0.646553);
}

double uuNNEE::synapse0x1e0e4130() {
   return (neuron0x1e0e1a80()*-0.194568);
}

double uuNNEE::synapse0x1e0e44b0() {
   return (neuron0x1e0e03c0()*-0.865571);
}

double uuNNEE::synapse0x1e0e44f0() {
   return (neuron0x1e0e0700()*0.308825);
}

double uuNNEE::synapse0x1e0e4530() {
   return (neuron0x1e0e0a40()*0.0972404);
}

double uuNNEE::synapse0x1e0e4570() {
   return (neuron0x1e0e0d80()*0.747192);
}

double uuNNEE::synapse0x1e0e45b0() {
   return (neuron0x1e0e10c0()*0.259783);
}

double uuNNEE::synapse0x1e0e45f0() {
   return (neuron0x1e0e1400()*0.262404);
}

double uuNNEE::synapse0x1e0e4630() {
   return (neuron0x1e0e1740()*1.1773);
}

double uuNNEE::synapse0x1e0e4670() {
   return (neuron0x1e0e1a80()*0.0180551);
}

double uuNNEE::synapse0x1e00c900() {
   return (neuron0x1e0e03c0()*-1.36178);
}

double uuNNEE::synapse0x1e00c940() {
   return (neuron0x1e0e0700()*0.469672);
}

double uuNNEE::synapse0x1e027800() {
   return (neuron0x1e0e0a40()*-0.517745);
}

double uuNNEE::synapse0x1e027840() {
   return (neuron0x1e0e0d80()*0.132974);
}

double uuNNEE::synapse0x1e027880() {
   return (neuron0x1e0e10c0()*-0.203739);
}

double uuNNEE::synapse0x1e0278c0() {
   return (neuron0x1e0e1400()*-0.44837);
}

double uuNNEE::synapse0x1e027900() {
   return (neuron0x1e0e1740()*2.97524);
}

double uuNNEE::synapse0x1e027940() {
   return (neuron0x1e0e1a80()*-0.316517);
}

double uuNNEE::synapse0x1e0e51c0() {
   return (neuron0x1e0e03c0()*1.82517);
}

double uuNNEE::synapse0x1e0e5200() {
   return (neuron0x1e0e0700()*0.414161);
}

double uuNNEE::synapse0x1e0e5240() {
   return (neuron0x1e0e0a40()*0.251126);
}

double uuNNEE::synapse0x1e0e5280() {
   return (neuron0x1e0e0d80()*-0.507721);
}

double uuNNEE::synapse0x1e0e52c0() {
   return (neuron0x1e0e10c0()*-2.57603);
}

double uuNNEE::synapse0x1e0e5300() {
   return (neuron0x1e0e1400()*0.286574);
}

double uuNNEE::synapse0x1e0e5340() {
   return (neuron0x1e0e1740()*-0.149589);
}

double uuNNEE::synapse0x1e0e5380() {
   return (neuron0x1e0e1a80()*0.0424992);
}

double uuNNEE::synapse0x1e0e5700() {
   return (neuron0x1e0e1ef0()*-0.728981);
}

double uuNNEE::synapse0x1e0e5740() {
   return (neuron0x1e0e2320()*-0.217835);
}

double uuNNEE::synapse0x1e0e5780() {
   return (neuron0x1e0e2860()*2.14288);
}

double uuNNEE::synapse0x1e0e57c0() {
   return (neuron0x1e0e2e30()*0.843815);
}

double uuNNEE::synapse0x1e0e5800() {
   return (neuron0x1e0e3370()*-1.88617);
}

double uuNNEE::synapse0x1e0e5840() {
   return (neuron0x1e0e36f0()*-1.95356);
}

double uuNNEE::synapse0x1e0e5880() {
   return (neuron0x1e0e3c30()*0.139252);
}

double uuNNEE::synapse0x1e0e58c0() {
   return (neuron0x1e0e4170()*1.47315);
}

double uuNNEE::synapse0x1e0e5900() {
   return (neuron0x1e0e46b0()*0.554027);
}

double uuNNEE::synapse0x1e0e5940() {
   return (neuron0x1e0e4f10()*0.17062);
}

double uuNNEE::synapse0x1e0e5cc0() {
   return (neuron0x1e0e1ef0()*0.448036);
}

double uuNNEE::synapse0x1e0e5d00() {
   return (neuron0x1e0e2320()*-0.254115);
}

double uuNNEE::synapse0x1e0e5d40() {
   return (neuron0x1e0e2860()*-0.00580066);
}

double uuNNEE::synapse0x1e0e5d80() {
   return (neuron0x1e0e2e30()*-1.69415);
}

double uuNNEE::synapse0x1e0e5dc0() {
   return (neuron0x1e0e3370()*0.854384);
}

double uuNNEE::synapse0x1e0e5e00() {
   return (neuron0x1e0e36f0()*0.979013);
}

double uuNNEE::synapse0x1e0e5e40() {
   return (neuron0x1e0e3c30()*-0.992856);
}

double uuNNEE::synapse0x1e0e5e80() {
   return (neuron0x1e0e4170()*-0.753223);
}

double uuNNEE::synapse0x1e0e5ec0() {
   return (neuron0x1e0e46b0()*-2.07683);
}

double uuNNEE::synapse0x1e0e5f00() {
   return (neuron0x1e0e4f10()*0.716962);
}

double uuNNEE::synapse0x1e0e6280() {
   return (neuron0x1e0e1ef0()*0.475778);
}

double uuNNEE::synapse0x1e0e62c0() {
   return (neuron0x1e0e2320()*0.272449);
}

double uuNNEE::synapse0x1e0e6300() {
   return (neuron0x1e0e2860()*1.24325);
}

double uuNNEE::synapse0x1e0e6340() {
   return (neuron0x1e0e2e30()*1.14387);
}

double uuNNEE::synapse0x1e0e6380() {
   return (neuron0x1e0e3370()*-1.03597);
}

double uuNNEE::synapse0x1e0e63c0() {
   return (neuron0x1e0e36f0()*-0.808608);
}

double uuNNEE::synapse0x1e0e6400() {
   return (neuron0x1e0e3c30()*-0.67539);
}

double uuNNEE::synapse0x1e0e6440() {
   return (neuron0x1e0e4170()*1.19305);
}

double uuNNEE::synapse0x1e0e6480() {
   return (neuron0x1e0e46b0()*0.303352);
}

double uuNNEE::synapse0x1e0e64c0() {
   return (neuron0x1e0e4f10()*1.16241);
}

double uuNNEE::synapse0x1e0e6840() {
   return (neuron0x1e0e1ef0()*-0.398235);
}

double uuNNEE::synapse0x1e0e6880() {
   return (neuron0x1e0e2320()*-1.07879);
}

double uuNNEE::synapse0x1e0e68c0() {
   return (neuron0x1e0e2860()*-1.88175);
}

double uuNNEE::synapse0x1e0e6900() {
   return (neuron0x1e0e2e30()*-1.7548);
}

double uuNNEE::synapse0x1e0e6940() {
   return (neuron0x1e0e3370()*-0.585679);
}

double uuNNEE::synapse0x1e0e6980() {
   return (neuron0x1e0e36f0()*0.328479);
}

double uuNNEE::synapse0x1e0e69c0() {
   return (neuron0x1e0e3c30()*-1.09065);
}

double uuNNEE::synapse0x1e0e6a00() {
   return (neuron0x1e0e4170()*1.3457);
}

double uuNNEE::synapse0x1e0e6a40() {
   return (neuron0x1e0e46b0()*1.02988);
}

double uuNNEE::synapse0x1e0e6a80() {
   return (neuron0x1e0e4f10()*1.23338);
}

double uuNNEE::synapse0x1e0e6e00() {
   return (neuron0x1e0e1ef0()*1.19672);
}

double uuNNEE::synapse0x1e0e6e40() {
   return (neuron0x1e0e2320()*0.632228);
}

double uuNNEE::synapse0x1e0e6e80() {
   return (neuron0x1e0e2860()*-0.546108);
}

double uuNNEE::synapse0x1e0e6ec0() {
   return (neuron0x1e0e2e30()*-0.471203);
}

double uuNNEE::synapse0x1e0e6f00() {
   return (neuron0x1e0e3370()*-1.06115);
}

double uuNNEE::synapse0x1e0e6f40() {
   return (neuron0x1e0e36f0()*-0.790267);
}

double uuNNEE::synapse0x1e0e6f80() {
   return (neuron0x1e0e3c30()*2.03872);
}

double uuNNEE::synapse0x1e0e6fc0() {
   return (neuron0x1e0e4170()*0.339001);
}

double uuNNEE::synapse0x1e0e7000() {
   return (neuron0x1e0e46b0()*2.30874);
}

double uuNNEE::synapse0x1e0e4b00() {
   return (neuron0x1e0e4f10()*-2.70062);
}

double uuNNEE::synapse0x1e0e4e80() {
   return (neuron0x1e0e53c0()*3.61326);
}

double uuNNEE::synapse0x1e0e4ec0() {
   return (neuron0x1e0e5980()*-2.48462);
}

double uuNNEE::synapse0x1e0e1dc0() {
   return (neuron0x1e0e5f40()*3.14274);
}

double uuNNEE::synapse0x1e0e1e00() {
   return (neuron0x1e0e6500()*-4.10291);
}

double uuNNEE::synapse0x1e0e1e40() {
   return (neuron0x1e0e6ac0()*3.40931);
}

