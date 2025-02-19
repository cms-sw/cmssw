#include "TestPos_100.h"
#include <cmath>

double TestPos_100::value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7,double in8) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   input4 = (in4 - 0)/1;
   input5 = (in5 - 0)/1;
   input6 = (in6 - 0)/1;
   input7 = (in7 - 0)/1;
   input8 = (in8 - 0)/1;
   switch(index) {
     case 0:
         return ((neuron0xa5d6850()*1)+0);
     default:
         return 0.;
   }
}

double TestPos_100::neuron0xa5c5530() {
   return input0;
}

double TestPos_100::neuron0xa5d3b10() {
   return input1;
}

double TestPos_100::neuron0xa5d3c10() {
   return input2;
}

double TestPos_100::neuron0xa5d3d58() {
   return input3;
}

double TestPos_100::neuron0xa5d3f30() {
   return input4;
}

double TestPos_100::neuron0xa5d4108() {
   return input5;
}

double TestPos_100::neuron0xa5d42e0() {
   return input6;
}

double TestPos_100::neuron0xa5d44b8() {
   return input7;
}

double TestPos_100::neuron0xa5d46b0() {
   return input8;
}

double TestPos_100::input0xa5d49c8() {
   double input = 0.531303;
   input += synapse0xa5c18e8();
   input += synapse0xa5d4b58();
   input += synapse0xa5d4b80();
   input += synapse0xa5d4ba8();
   input += synapse0xa5d4bd0();
   input += synapse0xa5d4bf8();
   input += synapse0xa5d4c20();
   input += synapse0xa5d4c48();
   input += synapse0xa5d4c70();
   return input;
}

double TestPos_100::neuron0xa5d49c8() {
   double input = input0xa5d49c8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d4c98() {
   double input = 0.00858638;
   input += synapse0xa5d4e70();
   input += synapse0xa5d4e98();
   input += synapse0xa5d4ec0();
   input += synapse0xa5d4ee8();
   input += synapse0xa5d4f10();
   input += synapse0xa5d4f38();
   input += synapse0xa5d4f60();
   input += synapse0xa5d4f88();
   input += synapse0xa5d5038();
   return input;
}

double TestPos_100::neuron0xa5d4c98() {
   double input = input0xa5d4c98();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d5060() {
   double input = 0.540891;
   input += synapse0xa5d51f0();
   input += synapse0xa5d5218();
   input += synapse0xa5d5240();
   input += synapse0xa5d5268();
   input += synapse0xa5d5290();
   input += synapse0xa5d52b8();
   input += synapse0xa5d52e0();
   input += synapse0xa5d5308();
   input += synapse0xa5d5330();
   return input;
}

double TestPos_100::neuron0xa5d5060() {
   double input = input0xa5d5060();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d5358() {
   double input = 0.31223;
   input += synapse0xa5d5530();
   input += synapse0xa5d5558();
   input += synapse0xa5d5580();
   input += synapse0xa5d55a8();
   input += synapse0xa5d55d0();
   input += synapse0xa5d55f8();
   input += synapse0xa5d4fb0();
   input += synapse0xa5d4fd8();
   input += synapse0xa5d5000();
   return input;
}

double TestPos_100::neuron0xa5d5358() {
   double input = input0xa5d5358();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d5728() {
   double input = -0.709237;
   input += synapse0xa5d5900();
   input += synapse0xa5d5928();
   input += synapse0xa5d5950();
   input += synapse0xa5d5978();
   input += synapse0xa5d59a0();
   input += synapse0xa5d59c8();
   input += synapse0xa5d59f0();
   input += synapse0xa5d5a18();
   input += synapse0xa5d5a40();
   return input;
}

double TestPos_100::neuron0xa5d5728() {
   double input = input0xa5d5728();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d5a68() {
   double input = -1.13743;
   input += synapse0xa5d5c40();
   input += synapse0xa5d5c68();
   input += synapse0xa5d5c90();
   input += synapse0xa5d5cb8();
   input += synapse0xa5d5ce0();
   input += synapse0xa5d5d08();
   input += synapse0xa5d5d30();
   input += synapse0xa5d5d58();
   input += synapse0xa5d5d80();
   return input;
}

double TestPos_100::neuron0xa5d5a68() {
   double input = input0xa5d5a68();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d5da8() {
   double input = -0.396006;
   input += synapse0xa5d5f80();
   input += synapse0xa5d5fa8();
   input += synapse0xa5d5fd0();
   input += synapse0xa5d5ff8();
   input += synapse0xa5d6020();
   input += synapse0xa5d6048();
   input += synapse0xa5d6070();
   input += synapse0xa5d6098();
   input += synapse0xa5d60c0();
   return input;
}

double TestPos_100::neuron0xa5d5da8() {
   double input = input0xa5d5da8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d60e8() {
   double input = -1.31206;
   input += synapse0xa5d6348();
   input += synapse0xa5d6370();
   input += synapse0xa521c20();
   input += synapse0xa521ba8();
   input += synapse0xa521dc0();
   input += synapse0xa34c098();
   input += synapse0xa5d5620();
   input += synapse0xa5d5648();
   input += synapse0xa5d5670();
   return input;
}

double TestPos_100::neuron0xa5d60e8() {
   double input = input0xa5d60e8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d65a0() {
   double input = 0.768949;
   input += synapse0xa5d66e8();
   input += synapse0xa5d6710();
   input += synapse0xa5d6738();
   input += synapse0xa5d6760();
   input += synapse0xa5d6788();
   input += synapse0xa5d67b0();
   input += synapse0xa5d67d8();
   input += synapse0xa5d6800();
   input += synapse0xa5d6828();
   return input;
}

double TestPos_100::neuron0xa5d65a0() {
   double input = input0xa5d65a0();
   return ((1/(1+exp(-input))) * 1)+0;
}

double TestPos_100::input0xa5d6850() {
   double input = 1.4245;
   input += synapse0xa5d6950();
   input += synapse0xa5d6978();
   input += synapse0xa5d69a0();
   input += synapse0xa5d69c8();
   input += synapse0xa5d69f0();
   input += synapse0xa5d6a18();
   input += synapse0xa5d6a40();
   input += synapse0xa5d6a68();
   input += synapse0xa5d6a90();
   return input;
}

double TestPos_100::neuron0xa5d6850() {
   double input = input0xa5d6850();
   return (input * 1)+0;
}

double TestPos_100::synapse0xa5c18e8() {
   return (neuron0xa5c5530()*-1.4618);
}

double TestPos_100::synapse0xa5d4b58() {
   return (neuron0xa5d3b10()*0.087325);
}

double TestPos_100::synapse0xa5d4b80() {
   return (neuron0xa5d3c10()*-0.728406);
}

double TestPos_100::synapse0xa5d4ba8() {
   return (neuron0xa5d3d58()*0.339684);
}

double TestPos_100::synapse0xa5d4bd0() {
   return (neuron0xa5d3f30()*-0.450042);
}

double TestPos_100::synapse0xa5d4bf8() {
   return (neuron0xa5d4108()*0.589395);
}

double TestPos_100::synapse0xa5d4c20() {
   return (neuron0xa5d42e0()*-0.405287);
}

double TestPos_100::synapse0xa5d4c48() {
   return (neuron0xa5d44b8()*0.0336332);
}

double TestPos_100::synapse0xa5d4c70() {
   return (neuron0xa5d46b0()*-1.21001);
}

double TestPos_100::synapse0xa5d4e70() {
   return (neuron0xa5c5530()*-0.17238);
}

double TestPos_100::synapse0xa5d4e98() {
   return (neuron0xa5d3b10()*0.263866);
}

double TestPos_100::synapse0xa5d4ec0() {
   return (neuron0xa5d3c10()*-0.231923);
}

double TestPos_100::synapse0xa5d4ee8() {
   return (neuron0xa5d3d58()*-0.534989);
}

double TestPos_100::synapse0xa5d4f10() {
   return (neuron0xa5d3f30()*-0.632176);
}

double TestPos_100::synapse0xa5d4f38() {
   return (neuron0xa5d4108()*0.128704);
}

double TestPos_100::synapse0xa5d4f60() {
   return (neuron0xa5d42e0()*-0.280789);
}

double TestPos_100::synapse0xa5d4f88() {
   return (neuron0xa5d44b8()*-0.792244);
}

double TestPos_100::synapse0xa5d5038() {
   return (neuron0xa5d46b0()*-0.927055);
}

double TestPos_100::synapse0xa5d51f0() {
   return (neuron0xa5c5530()*0.630397);
}

double TestPos_100::synapse0xa5d5218() {
   return (neuron0xa5d3b10()*0.0402529);
}

double TestPos_100::synapse0xa5d5240() {
   return (neuron0xa5d3c10()*-0.802947);
}

double TestPos_100::synapse0xa5d5268() {
   return (neuron0xa5d3d58()*0.463431);
}

double TestPos_100::synapse0xa5d5290() {
   return (neuron0xa5d3f30()*0.0615201);
}

double TestPos_100::synapse0xa5d52b8() {
   return (neuron0xa5d4108()*-1.43815);
}

double TestPos_100::synapse0xa5d52e0() {
   return (neuron0xa5d42e0()*-1.01121);
}

double TestPos_100::synapse0xa5d5308() {
   return (neuron0xa5d44b8()*-1.62751);
}

double TestPos_100::synapse0xa5d5330() {
   return (neuron0xa5d46b0()*-2.93597);
}

double TestPos_100::synapse0xa5d5530() {
   return (neuron0xa5c5530()*0.809439);
}

double TestPos_100::synapse0xa5d5558() {
   return (neuron0xa5d3b10()*0.201994);
}

double TestPos_100::synapse0xa5d5580() {
   return (neuron0xa5d3c10()*-0.0247191);
}

double TestPos_100::synapse0xa5d55a8() {
   return (neuron0xa5d3d58()*1.62191);
}

double TestPos_100::synapse0xa5d55d0() {
   return (neuron0xa5d3f30()*0.340893);
}

double TestPos_100::synapse0xa5d55f8() {
   return (neuron0xa5d4108()*-0.126776);
}

double TestPos_100::synapse0xa5d4fb0() {
   return (neuron0xa5d42e0()*0.848585);
}

double TestPos_100::synapse0xa5d4fd8() {
   return (neuron0xa5d44b8()*1.17485);
}

double TestPos_100::synapse0xa5d5000() {
   return (neuron0xa5d46b0()*3.12906);
}

double TestPos_100::synapse0xa5d5900() {
   return (neuron0xa5c5530()*1.39657);
}

double TestPos_100::synapse0xa5d5928() {
   return (neuron0xa5d3b10()*1.60718);
}

double TestPos_100::synapse0xa5d5950() {
   return (neuron0xa5d3c10()*-0.630117);
}

double TestPos_100::synapse0xa5d5978() {
   return (neuron0xa5d3d58()*-0.595902);
}

double TestPos_100::synapse0xa5d59a0() {
   return (neuron0xa5d3f30()*-1.4292);
}

double TestPos_100::synapse0xa5d59c8() {
   return (neuron0xa5d4108()*-2.37143);
}

double TestPos_100::synapse0xa5d59f0() {
   return (neuron0xa5d42e0()*-1.04322);
}

double TestPos_100::synapse0xa5d5a18() {
   return (neuron0xa5d44b8()*-2.12608);
}

double TestPos_100::synapse0xa5d5a40() {
   return (neuron0xa5d46b0()*-0.833842);
}

double TestPos_100::synapse0xa5d5c40() {
   return (neuron0xa5c5530()*1.54503);
}

double TestPos_100::synapse0xa5d5c68() {
   return (neuron0xa5d3b10()*1.37067);
}

double TestPos_100::synapse0xa5d5c90() {
   return (neuron0xa5d3c10()*-1.19252);
}

double TestPos_100::synapse0xa5d5cb8() {
   return (neuron0xa5d3d58()*-0.769953);
}

double TestPos_100::synapse0xa5d5ce0() {
   return (neuron0xa5d3f30()*-0.993863);
}

double TestPos_100::synapse0xa5d5d08() {
   return (neuron0xa5d4108()*1.03303);
}

double TestPos_100::synapse0xa5d5d30() {
   return (neuron0xa5d42e0()*-0.0630523);
}

double TestPos_100::synapse0xa5d5d58() {
   return (neuron0xa5d44b8()*0.216776);
}

double TestPos_100::synapse0xa5d5d80() {
   return (neuron0xa5d46b0()*0.255905);
}

double TestPos_100::synapse0xa5d5f80() {
   return (neuron0xa5c5530()*-0.361551);
}

double TestPos_100::synapse0xa5d5fa8() {
   return (neuron0xa5d3b10()*-0.207123);
}

double TestPos_100::synapse0xa5d5fd0() {
   return (neuron0xa5d3c10()*-1.4357);
}

double TestPos_100::synapse0xa5d5ff8() {
   return (neuron0xa5d3d58()*0.872323);
}

double TestPos_100::synapse0xa5d6020() {
   return (neuron0xa5d3f30()*-0.168812);
}

double TestPos_100::synapse0xa5d6048() {
   return (neuron0xa5d4108()*1.20287);
}

double TestPos_100::synapse0xa5d6070() {
   return (neuron0xa5d42e0()*0.654947);
}

double TestPos_100::synapse0xa5d6098() {
   return (neuron0xa5d44b8()*1.24797);
}

double TestPos_100::synapse0xa5d60c0() {
   return (neuron0xa5d46b0()*1.35056);
}

double TestPos_100::synapse0xa5d6348() {
   return (neuron0xa5c5530()*1.42169);
}

double TestPos_100::synapse0xa5d6370() {
   return (neuron0xa5d3b10()*0.782071);
}

double TestPos_100::synapse0xa521c20() {
   return (neuron0xa5d3c10()*-1.5144);
}

double TestPos_100::synapse0xa521ba8() {
   return (neuron0xa5d3d58()*-0.21864);
}

double TestPos_100::synapse0xa521dc0() {
   return (neuron0xa5d3f30()*-1.20186);
}

double TestPos_100::synapse0xa34c098() {
   return (neuron0xa5d4108()*-1.79633);
}

double TestPos_100::synapse0xa5d5620() {
   return (neuron0xa5d42e0()*-0.529529);
}

double TestPos_100::synapse0xa5d5648() {
   return (neuron0xa5d44b8()*-1.67783);
}

double TestPos_100::synapse0xa5d5670() {
   return (neuron0xa5d46b0()*-0.595233);
}

double TestPos_100::synapse0xa5d66e8() {
   return (neuron0xa5c5530()*-1.18159);
}

double TestPos_100::synapse0xa5d6710() {
   return (neuron0xa5d3b10()*0.143654);
}

double TestPos_100::synapse0xa5d6738() {
   return (neuron0xa5d3c10()*1.14203);
}

double TestPos_100::synapse0xa5d6760() {
   return (neuron0xa5d3d58()*1.1597);
}

double TestPos_100::synapse0xa5d6788() {
   return (neuron0xa5d3f30()*-0.68027);
}

double TestPos_100::synapse0xa5d67b0() {
   return (neuron0xa5d4108()*-0.704502);
}

double TestPos_100::synapse0xa5d67d8() {
   return (neuron0xa5d42e0()*-1.44991);
}

double TestPos_100::synapse0xa5d6800() {
   return (neuron0xa5d44b8()*-0.00746011);
}

double TestPos_100::synapse0xa5d6828() {
   return (neuron0xa5d46b0()*-2.01778);
}

double TestPos_100::synapse0xa5d6950() {
   return (neuron0xa5d49c8()*0.515968);
}

double TestPos_100::synapse0xa5d6978() {
   return (neuron0xa5d4c98()*-0.439706);
}

double TestPos_100::synapse0xa5d69a0() {
   return (neuron0xa5d5060()*3.41467);
}

double TestPos_100::synapse0xa5d69c8() {
   return (neuron0xa5d5358()*-1.85878);
}

double TestPos_100::synapse0xa5d69f0() {
   return (neuron0xa5d5728()*1.15939);
}

double TestPos_100::synapse0xa5d6a18() {
   return (neuron0xa5d5a68()*-1.46626);
}

double TestPos_100::synapse0xa5d6a40() {
   return (neuron0xa5d5da8()*-1.70389);
}

double TestPos_100::synapse0xa5d6a68() {
   return (neuron0xa5d60e8()*0.802298);
}

double TestPos_100::synapse0xa5d6a90() {
   return (neuron0xa5d65a0()*1.49076);
}

