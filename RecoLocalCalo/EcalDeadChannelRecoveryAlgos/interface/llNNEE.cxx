#include "llNNEE.h"
#include <cmath>

double llNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.40353)/1.40003;
   input2 = (in2 - 2.42657)/1.40629;
   input3 = (in3 - 2.42619)/1.40466;
   input4 = (in4 - 1.33856)/1.28698;
   input5 = (in5 - 1.33177)/1.28879;
   input6 = (in6 - 1.33367)/1.29347;
   input7 = (in7 - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x13016bc0();
     default:
         return 0.;
   }
}

double llNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.40353)/1.40003;
   input2 = (input[2] - 2.42657)/1.40629;
   input3 = (input[3] - 2.42619)/1.40466;
   input4 = (input[4] - 1.33856)/1.28698;
   input5 = (input[5] - 1.33177)/1.28879;
   input6 = (input[6] - 1.33367)/1.29347;
   input7 = (input[7] - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x13016bc0();
     default:
         return 0.;
   }
}

double llNNEE::neuron0x13012440() {
   return input0;
}

double llNNEE::neuron0x13012780() {
   return input1;
}

double llNNEE::neuron0x13012ac0() {
   return input2;
}

double llNNEE::neuron0x13012e00() {
   return input3;
}

double llNNEE::neuron0x13013140() {
   return input4;
}

double llNNEE::neuron0x13013480() {
   return input5;
}

double llNNEE::neuron0x130137c0() {
   return input6;
}

double llNNEE::neuron0x13013b00() {
   return input7;
}

double llNNEE::input0x13013f70() {
   double input = 0.900227;
   input += synapse0x12f72780();
   input += synapse0x1301b040();
   input += synapse0x13014220();
   input += synapse0x13014260();
   input += synapse0x130142a0();
   input += synapse0x130142e0();
   input += synapse0x13014320();
   input += synapse0x13014360();
   return input;
}

double llNNEE::neuron0x13013f70() {
   double input = input0x13013f70();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x130143a0() {
   double input = -0.566678;
   input += synapse0x130146e0();
   input += synapse0x13014720();
   input += synapse0x13014760();
   input += synapse0x130147a0();
   input += synapse0x130147e0();
   input += synapse0x13014820();
   input += synapse0x13014860();
   input += synapse0x130148a0();
   return input;
}

double llNNEE::neuron0x130143a0() {
   double input = input0x130143a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x130148e0() {
   double input = 0.272189;
   input += synapse0x13014c20();
   input += synapse0x12f40d20();
   input += synapse0x12f40d60();
   input += synapse0x13014d70();
   input += synapse0x13014db0();
   input += synapse0x13014df0();
   input += synapse0x13014e30();
   input += synapse0x13014e70();
   return input;
}

double llNNEE::neuron0x130148e0() {
   double input = input0x130148e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13014eb0() {
   double input = -0.907915;
   input += synapse0x130151f0();
   input += synapse0x13015230();
   input += synapse0x13015270();
   input += synapse0x130152b0();
   input += synapse0x130152f0();
   input += synapse0x13015330();
   input += synapse0x13015370();
   input += synapse0x130153b0();
   return input;
}

double llNNEE::neuron0x13014eb0() {
   double input = input0x13014eb0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x130153f0() {
   double input = -1.00744;
   input += synapse0x13015730();
   input += synapse0x13012370();
   input += synapse0x1301b080();
   input += synapse0x12f5d090();
   input += synapse0x13014c60();
   input += synapse0x13014ca0();
   input += synapse0x13014ce0();
   input += synapse0x13014d20();
   return input;
}

double llNNEE::neuron0x130153f0() {
   double input = input0x130153f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13015770() {
   double input = 0.396889;
   input += synapse0x13015ab0();
   input += synapse0x13015af0();
   input += synapse0x13015b30();
   input += synapse0x13015b70();
   input += synapse0x13015bb0();
   input += synapse0x13015bf0();
   input += synapse0x13015c30();
   input += synapse0x13015c70();
   return input;
}

double llNNEE::neuron0x13015770() {
   double input = input0x13015770();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13015cb0() {
   double input = -1.18242;
   input += synapse0x13015ff0();
   input += synapse0x13016030();
   input += synapse0x13016070();
   input += synapse0x130160b0();
   input += synapse0x130160f0();
   input += synapse0x13016130();
   input += synapse0x13016170();
   input += synapse0x130161b0();
   return input;
}

double llNNEE::neuron0x13015cb0() {
   double input = input0x13015cb0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x130161f0() {
   double input = 0.882358;
   input += synapse0x13016530();
   input += synapse0x13016570();
   input += synapse0x130165b0();
   input += synapse0x130165f0();
   input += synapse0x13016630();
   input += synapse0x13016670();
   input += synapse0x130166b0();
   input += synapse0x130166f0();
   return input;
}

double llNNEE::neuron0x130161f0() {
   double input = input0x130161f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13016730() {
   double input = 1.01806;
   input += synapse0x12f3e980();
   input += synapse0x12f3e9c0();
   input += synapse0x12f59880();
   input += synapse0x12f598c0();
   input += synapse0x12f59900();
   input += synapse0x12f59940();
   input += synapse0x12f59980();
   input += synapse0x12f599c0();
   return input;
}

double llNNEE::neuron0x13016730() {
   double input = input0x13016730();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13016f90() {
   double input = 0.855648;
   input += synapse0x13017240();
   input += synapse0x13017280();
   input += synapse0x130172c0();
   input += synapse0x13017300();
   input += synapse0x13017340();
   input += synapse0x13017380();
   input += synapse0x130173c0();
   input += synapse0x13017400();
   return input;
}

double llNNEE::neuron0x13016f90() {
   double input = input0x13016f90();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13017440() {
   double input = -1.64745;
   input += synapse0x13017780();
   input += synapse0x130177c0();
   input += synapse0x13017800();
   input += synapse0x13017840();
   input += synapse0x13017880();
   input += synapse0x130178c0();
   input += synapse0x13017900();
   input += synapse0x13017940();
   input += synapse0x13017980();
   input += synapse0x130179c0();
   return input;
}

double llNNEE::neuron0x13017440() {
   double input = input0x13017440();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13017a00() {
   double input = 0.502623;
   input += synapse0x13017d40();
   input += synapse0x13017d80();
   input += synapse0x13017dc0();
   input += synapse0x13017e00();
   input += synapse0x13017e40();
   input += synapse0x13017e80();
   input += synapse0x13017ec0();
   input += synapse0x13017f00();
   input += synapse0x13017f40();
   input += synapse0x13017f80();
   return input;
}

double llNNEE::neuron0x13017a00() {
   double input = input0x13017a00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13017fc0() {
   double input = 1.05585;
   input += synapse0x13018300();
   input += synapse0x13018340();
   input += synapse0x13018380();
   input += synapse0x130183c0();
   input += synapse0x13018400();
   input += synapse0x13018440();
   input += synapse0x13018480();
   input += synapse0x130184c0();
   input += synapse0x13018500();
   input += synapse0x13018540();
   return input;
}

double llNNEE::neuron0x13017fc0() {
   double input = input0x13017fc0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13018580() {
   double input = 0.709421;
   input += synapse0x130188c0();
   input += synapse0x13018900();
   input += synapse0x13018940();
   input += synapse0x13018980();
   input += synapse0x130189c0();
   input += synapse0x13018a00();
   input += synapse0x13018a40();
   input += synapse0x13018a80();
   input += synapse0x13018ac0();
   input += synapse0x13018b00();
   return input;
}

double llNNEE::neuron0x13018580() {
   double input = input0x13018580();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13018b40() {
   double input = -0.51613;
   input += synapse0x13018e80();
   input += synapse0x13018ec0();
   input += synapse0x13018f00();
   input += synapse0x13018f40();
   input += synapse0x13018f80();
   input += synapse0x13018fc0();
   input += synapse0x13019000();
   input += synapse0x13019040();
   input += synapse0x13019080();
   input += synapse0x13016b80();
   return input;
}

double llNNEE::neuron0x13018b40() {
   double input = input0x13018b40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double llNNEE::input0x13016bc0() {
   double input = -1.55835;
   input += synapse0x13016f00();
   input += synapse0x13016f40();
   input += synapse0x13013e40();
   input += synapse0x13013e80();
   input += synapse0x13013ec0();
   return input;
}

double llNNEE::neuron0x13016bc0() {
   double input = input0x13016bc0();
   return (input * 1)+0;
}

double llNNEE::synapse0x12f72780() {
   return (neuron0x13012440()*0.972565);
}

double llNNEE::synapse0x1301b040() {
   return (neuron0x13012780()*-0.186841);
}

double llNNEE::synapse0x13014220() {
   return (neuron0x13012ac0()*-0.675938);
}

double llNNEE::synapse0x13014260() {
   return (neuron0x13012e00()*0.921684);
}

double llNNEE::synapse0x130142a0() {
   return (neuron0x13013140()*-0.0250039);
}

double llNNEE::synapse0x130142e0() {
   return (neuron0x13013480()*-0.0829526);
}

double llNNEE::synapse0x13014320() {
   return (neuron0x130137c0()*0.641632);
}

double llNNEE::synapse0x13014360() {
   return (neuron0x13013b00()*-1.18715);
}

double llNNEE::synapse0x130146e0() {
   return (neuron0x13012440()*1.73331);
}

double llNNEE::synapse0x13014720() {
   return (neuron0x13012780()*-0.556065);
}

double llNNEE::synapse0x13014760() {
   return (neuron0x13012ac0()*0.501218);
}

double llNNEE::synapse0x130147a0() {
   return (neuron0x13012e00()*1.02699);
}

double llNNEE::synapse0x130147e0() {
   return (neuron0x13013140()*-0.523322);
}

double llNNEE::synapse0x13014820() {
   return (neuron0x13013480()*0.0387328);
}

double llNNEE::synapse0x13014860() {
   return (neuron0x130137c0()*0.270953);
}

double llNNEE::synapse0x130148a0() {
   return (neuron0x13013b00()*-3.38283);
}

double llNNEE::synapse0x13014c20() {
   return (neuron0x13012440()*0.429829);
}

double llNNEE::synapse0x12f40d20() {
   return (neuron0x13012780()*-0.215613);
}

double llNNEE::synapse0x12f40d60() {
   return (neuron0x13012ac0()*-2.95853);
}

double llNNEE::synapse0x13014d70() {
   return (neuron0x13012e00()*-0.0254754);
}

double llNNEE::synapse0x13014db0() {
   return (neuron0x13013140()*0.408803);
}

double llNNEE::synapse0x13014df0() {
   return (neuron0x13013480()*-0.229918);
}

double llNNEE::synapse0x13014e30() {
   return (neuron0x130137c0()*2.08506);
}

double llNNEE::synapse0x13014e70() {
   return (neuron0x13013b00()*-0.0131537);
}

double llNNEE::synapse0x130151f0() {
   return (neuron0x13012440()*0.746666);
}

double llNNEE::synapse0x13015230() {
   return (neuron0x13012780()*-1.13844);
}

double llNNEE::synapse0x13015270() {
   return (neuron0x13012ac0()*-0.20628);
}

double llNNEE::synapse0x130152b0() {
   return (neuron0x13012e00()*0.689589);
}

double llNNEE::synapse0x130152f0() {
   return (neuron0x13013140()*0.0139899);
}

double llNNEE::synapse0x13015330() {
   return (neuron0x13013480()*-1.07008);
}

double llNNEE::synapse0x13015370() {
   return (neuron0x130137c0()*-0.451243);
}

double llNNEE::synapse0x130153b0() {
   return (neuron0x13013b00()*1.71738);
}

double llNNEE::synapse0x13015730() {
   return (neuron0x13012440()*-1.69868);
}

double llNNEE::synapse0x13012370() {
   return (neuron0x13012780()*-0.76067);
}

double llNNEE::synapse0x1301b080() {
   return (neuron0x13012ac0()*-0.269277);
}

double llNNEE::synapse0x12f5d090() {
   return (neuron0x13012e00()*-0.77228);
}

double llNNEE::synapse0x13014c60() {
   return (neuron0x13013140()*0.13292);
}

double llNNEE::synapse0x13014ca0() {
   return (neuron0x13013480()*-0.0801225);
}

double llNNEE::synapse0x13014ce0() {
   return (neuron0x130137c0()*1.50998);
}

double llNNEE::synapse0x13014d20() {
   return (neuron0x13013b00()*1.75937);
}

double llNNEE::synapse0x13015ab0() {
   return (neuron0x13012440()*0.632788);
}

double llNNEE::synapse0x13015af0() {
   return (neuron0x13012780()*0.144782);
}

double llNNEE::synapse0x13015b30() {
   return (neuron0x13012ac0()*0.558561);
}

double llNNEE::synapse0x13015b70() {
   return (neuron0x13012e00()*-1.76916);
}

double llNNEE::synapse0x13015bb0() {
   return (neuron0x13013140()*-0.409269);
}

double llNNEE::synapse0x13015bf0() {
   return (neuron0x13013480()*0.00840235);
}

double llNNEE::synapse0x13015c30() {
   return (neuron0x130137c0()*-0.796708);
}

double llNNEE::synapse0x13015c70() {
   return (neuron0x13013b00()*1.4515);
}

double llNNEE::synapse0x13015ff0() {
   return (neuron0x13012440()*1.22116);
}

double llNNEE::synapse0x13016030() {
   return (neuron0x13012780()*-0.489883);
}

double llNNEE::synapse0x13016070() {
   return (neuron0x13012ac0()*-0.730029);
}

double llNNEE::synapse0x130160b0() {
   return (neuron0x13012e00()*-1.37951);
}

double llNNEE::synapse0x130160f0() {
   return (neuron0x13013140()*1.18109);
}

double llNNEE::synapse0x13016130() {
   return (neuron0x13013480()*0.71902);
}

double llNNEE::synapse0x13016170() {
   return (neuron0x130137c0()*-0.0593575);
}

double llNNEE::synapse0x130161b0() {
   return (neuron0x13013b00()*-0.127015);
}

double llNNEE::synapse0x13016530() {
   return (neuron0x13012440()*-0.531685);
}

double llNNEE::synapse0x13016570() {
   return (neuron0x13012780()*0.458726);
}

double llNNEE::synapse0x130165b0() {
   return (neuron0x13012ac0()*-0.00391746);
}

double llNNEE::synapse0x130165f0() {
   return (neuron0x13012e00()*0.116965);
}

double llNNEE::synapse0x13016630() {
   return (neuron0x13013140()*-0.903644);
}

double llNNEE::synapse0x13016670() {
   return (neuron0x13013480()*0.129238);
}

double llNNEE::synapse0x130166b0() {
   return (neuron0x130137c0()*0.0490041);
}

double llNNEE::synapse0x130166f0() {
   return (neuron0x13013b00()*0.105892);
}

double llNNEE::synapse0x12f3e980() {
   return (neuron0x13012440()*-1.65312);
}

double llNNEE::synapse0x12f3e9c0() {
   return (neuron0x13012780()*0.376034);
}

double llNNEE::synapse0x12f59880() {
   return (neuron0x13012ac0()*-1.08633);
}

double llNNEE::synapse0x12f598c0() {
   return (neuron0x13012e00()*-0.43458);
}

double llNNEE::synapse0x12f59900() {
   return (neuron0x13013140()*-0.339381);
}

double llNNEE::synapse0x12f59940() {
   return (neuron0x13013480()*0.326537);
}

double llNNEE::synapse0x12f59980() {
   return (neuron0x130137c0()*3.81688);
}

double llNNEE::synapse0x12f599c0() {
   return (neuron0x13013b00()*-0.15933);
}

double llNNEE::synapse0x13017240() {
   return (neuron0x13012440()*-0.303486);
}

double llNNEE::synapse0x13017280() {
   return (neuron0x13012780()*0.176229);
}

double llNNEE::synapse0x130172c0() {
   return (neuron0x13012ac0()*-0.651793);
}

double llNNEE::synapse0x13017300() {
   return (neuron0x13012e00()*-0.109911);
}

double llNNEE::synapse0x13017340() {
   return (neuron0x13013140()*0.204654);
}

double llNNEE::synapse0x13017380() {
   return (neuron0x13013480()*-0.444645);
}

double llNNEE::synapse0x130173c0() {
   return (neuron0x130137c0()*0.323009);
}

double llNNEE::synapse0x13017400() {
   return (neuron0x13013b00()*-0.32932);
}

double llNNEE::synapse0x13017780() {
   return (neuron0x13013f70()*0.307842);
}

double llNNEE::synapse0x130177c0() {
   return (neuron0x130143a0()*-2.10435);
}

double llNNEE::synapse0x13017800() {
   return (neuron0x130148e0()*1.29426);
}

double llNNEE::synapse0x13017840() {
   return (neuron0x13014eb0()*-1.9263);
}

double llNNEE::synapse0x13017880() {
   return (neuron0x130153f0()*0.498768);
}

double llNNEE::synapse0x130178c0() {
   return (neuron0x13015770()*1.94302);
}

double llNNEE::synapse0x13017900() {
   return (neuron0x13015cb0()*-1.51854);
}

double llNNEE::synapse0x13017940() {
   return (neuron0x130161f0()*-1.5205);
}

double llNNEE::synapse0x13017980() {
   return (neuron0x13016730()*1.4293);
}

double llNNEE::synapse0x130179c0() {
   return (neuron0x13016f90()*-0.0628727);
}

double llNNEE::synapse0x13017d40() {
   return (neuron0x13013f70()*-0.936375);
}

double llNNEE::synapse0x13017d80() {
   return (neuron0x130143a0()*-1.01656);
}

double llNNEE::synapse0x13017dc0() {
   return (neuron0x130148e0()*-0.94244);
}

double llNNEE::synapse0x13017e00() {
   return (neuron0x13014eb0()*0.648178);
}

double llNNEE::synapse0x13017e40() {
   return (neuron0x130153f0()*-0.215294);
}

double llNNEE::synapse0x13017e80() {
   return (neuron0x13015770()*-0.0498091);
}

double llNNEE::synapse0x13017ec0() {
   return (neuron0x13015cb0()*-0.346044);
}

double llNNEE::synapse0x13017f00() {
   return (neuron0x130161f0()*2.10931);
}

double llNNEE::synapse0x13017f40() {
   return (neuron0x13016730()*-0.534627);
}

double llNNEE::synapse0x13017f80() {
   return (neuron0x13016f90()*0.378392);
}

double llNNEE::synapse0x13018300() {
   return (neuron0x13013f70()*2.48796);
}

double llNNEE::synapse0x13018340() {
   return (neuron0x130143a0()*-1.0624);
}

double llNNEE::synapse0x13018380() {
   return (neuron0x130148e0()*-1.35455);
}

double llNNEE::synapse0x130183c0() {
   return (neuron0x13014eb0()*1.03804);
}

double llNNEE::synapse0x13018400() {
   return (neuron0x130153f0()*-0.328661);
}

double llNNEE::synapse0x13018440() {
   return (neuron0x13015770()*1.41886);
}

double llNNEE::synapse0x13018480() {
   return (neuron0x13015cb0()*-0.705526);
}

double llNNEE::synapse0x130184c0() {
   return (neuron0x130161f0()*-0.536636);
}

double llNNEE::synapse0x13018500() {
   return (neuron0x13016730()*0.702937);
}

double llNNEE::synapse0x13018540() {
   return (neuron0x13016f90()*-1.11388);
}

double llNNEE::synapse0x130188c0() {
   return (neuron0x13013f70()*0.970495);
}

double llNNEE::synapse0x13018900() {
   return (neuron0x130143a0()*-1.02329);
}

double llNNEE::synapse0x13018940() {
   return (neuron0x130148e0()*-0.535743);
}

double llNNEE::synapse0x13018980() {
   return (neuron0x13014eb0()*0.184999);
}

double llNNEE::synapse0x130189c0() {
   return (neuron0x130153f0()*-0.0849754);
}

double llNNEE::synapse0x13018a00() {
   return (neuron0x13015770()*1.382);
}

double llNNEE::synapse0x13018a40() {
   return (neuron0x13015cb0()*-0.210058);
}

double llNNEE::synapse0x13018a80() {
   return (neuron0x130161f0()*-0.341471);
}

double llNNEE::synapse0x13018ac0() {
   return (neuron0x13016730()*-0.142899);
}

double llNNEE::synapse0x13018b00() {
   return (neuron0x13016f90()*-1.38485);
}

double llNNEE::synapse0x13018e80() {
   return (neuron0x13013f70()*0.831022);
}

double llNNEE::synapse0x13018ec0() {
   return (neuron0x130143a0()*0.340431);
}

double llNNEE::synapse0x13018f00() {
   return (neuron0x130148e0()*0.894314);
}

double llNNEE::synapse0x13018f40() {
   return (neuron0x13014eb0()*0.644547);
}

double llNNEE::synapse0x13018f80() {
   return (neuron0x130153f0()*1.17724);
}

double llNNEE::synapse0x13018fc0() {
   return (neuron0x13015770()*-0.0303256);
}

double llNNEE::synapse0x13019000() {
   return (neuron0x13015cb0()*-1.17792);
}

double llNNEE::synapse0x13019040() {
   return (neuron0x130161f0()*-0.662309);
}

double llNNEE::synapse0x13019080() {
   return (neuron0x13016730()*-0.295636);
}

double llNNEE::synapse0x13016b80() {
   return (neuron0x13016f90()*-2.16503);
}

double llNNEE::synapse0x13016f00() {
   return (neuron0x13017440()*3.71847);
}

double llNNEE::synapse0x13016f40() {
   return (neuron0x13017a00()*-2.82565);
}

double llNNEE::synapse0x13013e40() {
   return (neuron0x13017fc0()*3.13229);
}

double llNNEE::synapse0x13013e80() {
   return (neuron0x13018580()*2.1016);
}

double llNNEE::synapse0x13013ec0() {
   return (neuron0x13018b40()*4.08793);
}

