#include "ccNNEE.h"
#include <cmath>

double ccNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 2.40353)/1.40003;
   input1 = (in1 - 2.41121)/1.41004;
   input2 = (in2 - 2.42657)/1.40629;
   input3 = (in3 - 2.42619)/1.40466;
   input4 = (in4 - 1.33856)/1.28698;
   input5 = (in5 - 1.33177)/1.28879;
   input6 = (in6 - 1.33367)/1.29347;
   input7 = (in7 - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x19a4cc00();
     default:
         return 0.;
   }
}

double ccNNEE::Value(int index, double* input) {
   input0 = (input[0] - 2.40353)/1.40003;
   input1 = (input[1] - 2.41121)/1.41004;
   input2 = (input[2] - 2.42657)/1.40629;
   input3 = (input[3] - 2.42619)/1.40466;
   input4 = (input[4] - 1.33856)/1.28698;
   input5 = (input[5] - 1.33177)/1.28879;
   input6 = (input[6] - 1.33367)/1.29347;
   input7 = (input[7] - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x19a4cc00();
     default:
         return 0.;
   }
}

double ccNNEE::neuron0x19a48480() {
   return input0;
}

double ccNNEE::neuron0x19a487c0() {
   return input1;
}

double ccNNEE::neuron0x19a48b00() {
   return input2;
}

double ccNNEE::neuron0x19a48e40() {
   return input3;
}

double ccNNEE::neuron0x19a49180() {
   return input4;
}

double ccNNEE::neuron0x19a494c0() {
   return input5;
}

double ccNNEE::neuron0x19a49800() {
   return input6;
}

double ccNNEE::neuron0x19a49b40() {
   return input7;
}

double ccNNEE::input0x19a49fb0() {
   double input = 1.45018;
   input += synapse0x199a87c0();
   input += synapse0x19a51080();
   input += synapse0x19a4a260();
   input += synapse0x19a4a2a0();
   input += synapse0x19a4a2e0();
   input += synapse0x19a4a320();
   input += synapse0x19a4a360();
   input += synapse0x19a4a3a0();
   return input;
}

double ccNNEE::neuron0x19a49fb0() {
   double input = input0x19a49fb0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4a3e0() {
   double input = 0.513263;
   input += synapse0x19a4a720();
   input += synapse0x19a4a760();
   input += synapse0x19a4a7a0();
   input += synapse0x19a4a7e0();
   input += synapse0x19a4a820();
   input += synapse0x19a4a860();
   input += synapse0x19a4a8a0();
   input += synapse0x19a4a8e0();
   return input;
}

double ccNNEE::neuron0x19a4a3e0() {
   double input = input0x19a4a3e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4a920() {
   double input = -0.424002;
   input += synapse0x19a4ac60();
   input += synapse0x19976d60();
   input += synapse0x19976da0();
   input += synapse0x19a4adb0();
   input += synapse0x19a4adf0();
   input += synapse0x19a4ae30();
   input += synapse0x19a4ae70();
   input += synapse0x19a4aeb0();
   return input;
}

double ccNNEE::neuron0x19a4a920() {
   double input = input0x19a4a920();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4aef0() {
   double input = -3.93277;
   input += synapse0x19a4b230();
   input += synapse0x19a4b270();
   input += synapse0x19a4b2b0();
   input += synapse0x19a4b2f0();
   input += synapse0x19a4b330();
   input += synapse0x19a4b370();
   input += synapse0x19a4b3b0();
   input += synapse0x19a4b3f0();
   return input;
}

double ccNNEE::neuron0x19a4aef0() {
   double input = input0x19a4aef0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4b430() {
   double input = -0.770492;
   input += synapse0x19a4b770();
   input += synapse0x19a483b0();
   input += synapse0x19a510c0();
   input += synapse0x199930d0();
   input += synapse0x19a4aca0();
   input += synapse0x19a4ace0();
   input += synapse0x19a4ad20();
   input += synapse0x19a4ad60();
   return input;
}

double ccNNEE::neuron0x19a4b430() {
   double input = input0x19a4b430();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4b7b0() {
   double input = 0.50899;
   input += synapse0x19a4baf0();
   input += synapse0x19a4bb30();
   input += synapse0x19a4bb70();
   input += synapse0x19a4bbb0();
   input += synapse0x19a4bbf0();
   input += synapse0x19a4bc30();
   input += synapse0x19a4bc70();
   input += synapse0x19a4bcb0();
   return input;
}

double ccNNEE::neuron0x19a4b7b0() {
   double input = input0x19a4b7b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4bcf0() {
   double input = -2.4899;
   input += synapse0x19a4c030();
   input += synapse0x19a4c070();
   input += synapse0x19a4c0b0();
   input += synapse0x19a4c0f0();
   input += synapse0x19a4c130();
   input += synapse0x19a4c170();
   input += synapse0x19a4c1b0();
   input += synapse0x19a4c1f0();
   return input;
}

double ccNNEE::neuron0x19a4bcf0() {
   double input = input0x19a4bcf0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4c230() {
   double input = -1.0127;
   input += synapse0x19a4c570();
   input += synapse0x19a4c5b0();
   input += synapse0x19a4c5f0();
   input += synapse0x19a4c630();
   input += synapse0x19a4c670();
   input += synapse0x19a4c6b0();
   input += synapse0x19a4c6f0();
   input += synapse0x19a4c730();
   return input;
}

double ccNNEE::neuron0x19a4c230() {
   double input = input0x19a4c230();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4c770() {
   double input = -1.4571;
   input += synapse0x199749c0();
   input += synapse0x19974a00();
   input += synapse0x1998f8c0();
   input += synapse0x1998f900();
   input += synapse0x1998f940();
   input += synapse0x1998f980();
   input += synapse0x1998f9c0();
   input += synapse0x1998fa00();
   return input;
}

double ccNNEE::neuron0x19a4c770() {
   double input = input0x19a4c770();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4cfd0() {
   double input = 0.71886;
   input += synapse0x19a4d280();
   input += synapse0x19a4d2c0();
   input += synapse0x19a4d300();
   input += synapse0x19a4d340();
   input += synapse0x19a4d380();
   input += synapse0x19a4d3c0();
   input += synapse0x19a4d400();
   input += synapse0x19a4d440();
   return input;
}

double ccNNEE::neuron0x19a4cfd0() {
   double input = input0x19a4cfd0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4d480() {
   double input = -4.01118;
   input += synapse0x19a4d7c0();
   input += synapse0x19a4d800();
   input += synapse0x19a4d840();
   input += synapse0x19a4d880();
   input += synapse0x19a4d8c0();
   input += synapse0x19a4d900();
   input += synapse0x19a4d940();
   input += synapse0x19a4d980();
   input += synapse0x19a4d9c0();
   input += synapse0x19a4da00();
   return input;
}

double ccNNEE::neuron0x19a4d480() {
   double input = input0x19a4d480();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4da40() {
   double input = -2.05647;
   input += synapse0x19a4dd80();
   input += synapse0x19a4ddc0();
   input += synapse0x19a4de00();
   input += synapse0x19a4de40();
   input += synapse0x19a4de80();
   input += synapse0x19a4dec0();
   input += synapse0x19a4df00();
   input += synapse0x19a4df40();
   input += synapse0x19a4df80();
   input += synapse0x19a4dfc0();
   return input;
}

double ccNNEE::neuron0x19a4da40() {
   double input = input0x19a4da40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4e000() {
   double input = -1.37802;
   input += synapse0x19a4e340();
   input += synapse0x19a4e380();
   input += synapse0x19a4e3c0();
   input += synapse0x19a4e400();
   input += synapse0x19a4e440();
   input += synapse0x19a4e480();
   input += synapse0x19a4e4c0();
   input += synapse0x19a4e500();
   input += synapse0x19a4e540();
   input += synapse0x19a4e580();
   return input;
}

double ccNNEE::neuron0x19a4e000() {
   double input = input0x19a4e000();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4e5c0() {
   double input = 0.267395;
   input += synapse0x19a4e900();
   input += synapse0x19a4e940();
   input += synapse0x19a4e980();
   input += synapse0x19a4e9c0();
   input += synapse0x19a4ea00();
   input += synapse0x19a4ea40();
   input += synapse0x19a4ea80();
   input += synapse0x19a4eac0();
   input += synapse0x19a4eb00();
   input += synapse0x19a4eb40();
   return input;
}

double ccNNEE::neuron0x19a4e5c0() {
   double input = input0x19a4e5c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4eb80() {
   double input = -1.92227;
   input += synapse0x19a4eec0();
   input += synapse0x19a4ef00();
   input += synapse0x19a4ef40();
   input += synapse0x19a4ef80();
   input += synapse0x19a4efc0();
   input += synapse0x19a4f000();
   input += synapse0x19a4f040();
   input += synapse0x19a4f080();
   input += synapse0x19a4f0c0();
   input += synapse0x19a4cbc0();
   return input;
}

double ccNNEE::neuron0x19a4eb80() {
   double input = input0x19a4eb80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ccNNEE::input0x19a4cc00() {
   double input = 1.45162;
   input += synapse0x19a4cf40();
   input += synapse0x19a4cf80();
   input += synapse0x19a49e80();
   input += synapse0x19a49ec0();
   input += synapse0x19a49f00();
   return input;
}

double ccNNEE::neuron0x19a4cc00() {
   double input = input0x19a4cc00();
   return (input * 1)+0;
}

double ccNNEE::synapse0x199a87c0() {
   return (neuron0x19a48480()*1.71165);
}

double ccNNEE::synapse0x19a51080() {
   return (neuron0x19a487c0()*1.54118);
}

double ccNNEE::synapse0x19a4a260() {
   return (neuron0x19a48b00()*-0.726711);
}

double ccNNEE::synapse0x19a4a2a0() {
   return (neuron0x19a48e40()*-1.5899);
}

double ccNNEE::synapse0x19a4a2e0() {
   return (neuron0x19a49180()*-0.704007);
}

double ccNNEE::synapse0x19a4a320() {
   return (neuron0x19a494c0()*0.154605);
}

double ccNNEE::synapse0x19a4a360() {
   return (neuron0x19a49800()*0.189555);
}

double ccNNEE::synapse0x19a4a3a0() {
   return (neuron0x19a49b40()*-0.282059);
}

double ccNNEE::synapse0x19a4a720() {
   return (neuron0x19a48480()*-1.80849);
}

double ccNNEE::synapse0x19a4a760() {
   return (neuron0x19a487c0()*-0.573659);
}

double ccNNEE::synapse0x19a4a7a0() {
   return (neuron0x19a48b00()*0.604474);
}

double ccNNEE::synapse0x19a4a7e0() {
   return (neuron0x19a48e40()*0.964384);
}

double ccNNEE::synapse0x19a4a820() {
   return (neuron0x19a49180()*0.0577964);
}

double ccNNEE::synapse0x19a4a860() {
   return (neuron0x19a494c0()*0.0857454);
}

double ccNNEE::synapse0x19a4a8a0() {
   return (neuron0x19a49800()*0.243587);
}

double ccNNEE::synapse0x19a4a8e0() {
   return (neuron0x19a49b40()*0.162777);
}

double ccNNEE::synapse0x19a4ac60() {
   return (neuron0x19a48480()*0.268058);
}

double ccNNEE::synapse0x19976d60() {
   return (neuron0x19a487c0()*2.06646);
}

double ccNNEE::synapse0x19976da0() {
   return (neuron0x19a48b00()*-0.372554);
}

double ccNNEE::synapse0x19a4adb0() {
   return (neuron0x19a48e40()*-0.861342);
}

double ccNNEE::synapse0x19a4adf0() {
   return (neuron0x19a49180()*-0.755119);
}

double ccNNEE::synapse0x19a4ae30() {
   return (neuron0x19a494c0()*0.322542);
}

double ccNNEE::synapse0x19a4ae70() {
   return (neuron0x19a49800()*0.13772);
}

double ccNNEE::synapse0x19a4aeb0() {
   return (neuron0x19a49b40()*-0.366595);
}

double ccNNEE::synapse0x19a4b230() {
   return (neuron0x19a48480()*0.979019);
}

double ccNNEE::synapse0x19a4b270() {
   return (neuron0x19a487c0()*0.670766);
}

double ccNNEE::synapse0x19a4b2b0() {
   return (neuron0x19a48b00()*-3.34411);
}

double ccNNEE::synapse0x19a4b2f0() {
   return (neuron0x19a48e40()*-0.264608);
}

double ccNNEE::synapse0x19a4b330() {
   return (neuron0x19a49180()*0.650668);
}

double ccNNEE::synapse0x19a4b370() {
   return (neuron0x19a494c0()*-0.233652);
}

double ccNNEE::synapse0x19a4b3b0() {
   return (neuron0x19a49800()*0.312077);
}

double ccNNEE::synapse0x19a4b3f0() {
   return (neuron0x19a49b40()*0.27134);
}

double ccNNEE::synapse0x19a4b770() {
   return (neuron0x19a48480()*0.777705);
}

double ccNNEE::synapse0x19a483b0() {
   return (neuron0x19a487c0()*0.0741502);
}

double ccNNEE::synapse0x19a510c0() {
   return (neuron0x19a48b00()*-0.83739);
}

double ccNNEE::synapse0x199930d0() {
   return (neuron0x19a48e40()*-0.055425);
}

double ccNNEE::synapse0x19a4aca0() {
   return (neuron0x19a49180()*0.170293);
}

double ccNNEE::synapse0x19a4ace0() {
   return (neuron0x19a494c0()*0.0117798);
}

double ccNNEE::synapse0x19a4ad20() {
   return (neuron0x19a49800()*0.308031);
}

double ccNNEE::synapse0x19a4ad60() {
   return (neuron0x19a49b40()*0.403181);
}

double ccNNEE::synapse0x19a4baf0() {
   return (neuron0x19a48480()*-0.264386);
}

double ccNNEE::synapse0x19a4bb30() {
   return (neuron0x19a487c0()*1.32561);
}

double ccNNEE::synapse0x19a4bb70() {
   return (neuron0x19a48b00()*-0.489181);
}

double ccNNEE::synapse0x19a4bbb0() {
   return (neuron0x19a48e40()*0.655156);
}

double ccNNEE::synapse0x19a4bbf0() {
   return (neuron0x19a49180()*-0.377223);
}

double ccNNEE::synapse0x19a4bc30() {
   return (neuron0x19a494c0()*0.505918);
}

double ccNNEE::synapse0x19a4bc70() {
   return (neuron0x19a49800()*0.256939);
}

double ccNNEE::synapse0x19a4bcb0() {
   return (neuron0x19a49b40()*-1.1697);
}

double ccNNEE::synapse0x19a4c030() {
   return (neuron0x19a48480()*-0.961581);
}

double ccNNEE::synapse0x19a4c070() {
   return (neuron0x19a487c0()*0.941952);
}

double ccNNEE::synapse0x19a4c0b0() {
   return (neuron0x19a48b00()*0.0325193);
}

double ccNNEE::synapse0x19a4c0f0() {
   return (neuron0x19a48e40()*-0.569454);
}

double ccNNEE::synapse0x19a4c130() {
   return (neuron0x19a49180()*0.887025);
}

double ccNNEE::synapse0x19a4c170() {
   return (neuron0x19a494c0()*-0.53192);
}

double ccNNEE::synapse0x19a4c1b0() {
   return (neuron0x19a49800()*-0.612533);
}

double ccNNEE::synapse0x19a4c1f0() {
   return (neuron0x19a49b40()*0.515826);
}

double ccNNEE::synapse0x19a4c570() {
   return (neuron0x19a48480()*-0.228631);
}

double ccNNEE::synapse0x19a4c5b0() {
   return (neuron0x19a487c0()*0.117628);
}

double ccNNEE::synapse0x19a4c5f0() {
   return (neuron0x19a48b00()*0.767413);
}

double ccNNEE::synapse0x19a4c630() {
   return (neuron0x19a48e40()*0.172603);
}

double ccNNEE::synapse0x19a4c670() {
   return (neuron0x19a49180()*0.0917384);
}

double ccNNEE::synapse0x19a4c6b0() {
   return (neuron0x19a494c0()*0.245461);
}

double ccNNEE::synapse0x19a4c6f0() {
   return (neuron0x19a49800()*0.0541929);
}

double ccNNEE::synapse0x19a4c730() {
   return (neuron0x19a49b40()*0.163008);
}

double ccNNEE::synapse0x199749c0() {
   return (neuron0x19a48480()*-0.564364);
}

double ccNNEE::synapse0x19974a00() {
   return (neuron0x19a487c0()*-0.162665);
}

double ccNNEE::synapse0x1998f8c0() {
   return (neuron0x19a48b00()*-0.110124);
}

double ccNNEE::synapse0x1998f900() {
   return (neuron0x19a48e40()*-0.00196696);
}

double ccNNEE::synapse0x1998f940() {
   return (neuron0x19a49180()*-0.459567);
}

double ccNNEE::synapse0x1998f980() {
   return (neuron0x19a494c0()*0.86846);
}

double ccNNEE::synapse0x1998f9c0() {
   return (neuron0x19a49800()*0.989158);
}

double ccNNEE::synapse0x1998fa00() {
   return (neuron0x19a49b40()*-0.979802);
}

double ccNNEE::synapse0x19a4d280() {
   return (neuron0x19a48480()*0.936438);
}

double ccNNEE::synapse0x19a4d2c0() {
   return (neuron0x19a487c0()*0.580903);
}

double ccNNEE::synapse0x19a4d300() {
   return (neuron0x19a48b00()*-0.522228);
}

double ccNNEE::synapse0x19a4d340() {
   return (neuron0x19a48e40()*-1.61368);
}

double ccNNEE::synapse0x19a4d380() {
   return (neuron0x19a49180()*0.6777);
}

double ccNNEE::synapse0x19a4d3c0() {
   return (neuron0x19a494c0()*-0.385456);
}

double ccNNEE::synapse0x19a4d400() {
   return (neuron0x19a49800()*-0.0587413);
}

double ccNNEE::synapse0x19a4d440() {
   return (neuron0x19a49b40()*0.117251);
}

double ccNNEE::synapse0x19a4d7c0() {
   return (neuron0x19a49fb0()*2.17389);
}

double ccNNEE::synapse0x19a4d800() {
   return (neuron0x19a4a3e0()*2.18535);
}

double ccNNEE::synapse0x19a4d840() {
   return (neuron0x19a4a920()*-1.15272);
}

double ccNNEE::synapse0x19a4d880() {
   return (neuron0x19a4aef0()*-3.18652);
}

double ccNNEE::synapse0x19a4d8c0() {
   return (neuron0x19a4b430()*-0.668958);
}

double ccNNEE::synapse0x19a4d900() {
   return (neuron0x19a4b7b0()*1.1318);
}

double ccNNEE::synapse0x19a4d940() {
   return (neuron0x19a4bcf0()*-3.22956);
}

double ccNNEE::synapse0x19a4d980() {
   return (neuron0x19a4c230()*0.639365);
}

double ccNNEE::synapse0x19a4d9c0() {
   return (neuron0x19a4c770()*-2.57762);
}

double ccNNEE::synapse0x19a4da00() {
   return (neuron0x19a4cfd0()*1.2187);
}

double ccNNEE::synapse0x19a4dd80() {
   return (neuron0x19a49fb0()*0.924918);
}

double ccNNEE::synapse0x19a4ddc0() {
   return (neuron0x19a4a3e0()*-0.235074);
}

double ccNNEE::synapse0x19a4de00() {
   return (neuron0x19a4a920()*-0.554771);
}

double ccNNEE::synapse0x19a4de40() {
   return (neuron0x19a4aef0()*-1.45049);
}

double ccNNEE::synapse0x19a4de80() {
   return (neuron0x19a4b430()*0.670977);
}

double ccNNEE::synapse0x19a4dec0() {
   return (neuron0x19a4b7b0()*1.09232);
}

double ccNNEE::synapse0x19a4df00() {
   return (neuron0x19a4bcf0()*-0.882513);
}

double ccNNEE::synapse0x19a4df40() {
   return (neuron0x19a4c230()*0.933392);
}

double ccNNEE::synapse0x19a4df80() {
   return (neuron0x19a4c770()*-1.07881);
}

double ccNNEE::synapse0x19a4dfc0() {
   return (neuron0x19a4cfd0()*0.231102);
}

double ccNNEE::synapse0x19a4e340() {
   return (neuron0x19a49fb0()*0.278011);
}

double ccNNEE::synapse0x19a4e380() {
   return (neuron0x19a4a3e0()*0.611001);
}

double ccNNEE::synapse0x19a4e3c0() {
   return (neuron0x19a4a920()*-0.288964);
}

double ccNNEE::synapse0x19a4e400() {
   return (neuron0x19a4aef0()*0.543014);
}

double ccNNEE::synapse0x19a4e440() {
   return (neuron0x19a4b430()*1.78693);
}

double ccNNEE::synapse0x19a4e480() {
   return (neuron0x19a4b7b0()*0.724617);
}

double ccNNEE::synapse0x19a4e4c0() {
   return (neuron0x19a4bcf0()*-0.163241);
}

double ccNNEE::synapse0x19a4e500() {
   return (neuron0x19a4c230()*1.36038);
}

double ccNNEE::synapse0x19a4e540() {
   return (neuron0x19a4c770()*-0.541271);
}

double ccNNEE::synapse0x19a4e580() {
   return (neuron0x19a4cfd0()*0.250387);
}

double ccNNEE::synapse0x19a4e900() {
   return (neuron0x19a49fb0()*0.333114);
}

double ccNNEE::synapse0x19a4e940() {
   return (neuron0x19a4a3e0()*-0.379421);
}

double ccNNEE::synapse0x19a4e980() {
   return (neuron0x19a4a920()*0.229684);
}

double ccNNEE::synapse0x19a4e9c0() {
   return (neuron0x19a4aef0()*-0.232456);
}

double ccNNEE::synapse0x19a4ea00() {
   return (neuron0x19a4b430()*-0.583428);
}

double ccNNEE::synapse0x19a4ea40() {
   return (neuron0x19a4b7b0()*-0.377764);
}

double ccNNEE::synapse0x19a4ea80() {
   return (neuron0x19a4bcf0()*-1.1476);
}

double ccNNEE::synapse0x19a4eac0() {
   return (neuron0x19a4c230()*-0.576983);
}

double ccNNEE::synapse0x19a4eb00() {
   return (neuron0x19a4c770()*-0.375297);
}

double ccNNEE::synapse0x19a4eb40() {
   return (neuron0x19a4cfd0()*0.0778199);
}

double ccNNEE::synapse0x19a4eec0() {
   return (neuron0x19a49fb0()*-0.177098);
}

double ccNNEE::synapse0x19a4ef00() {
   return (neuron0x19a4a3e0()*0.208895);
}

double ccNNEE::synapse0x19a4ef40() {
   return (neuron0x19a4a920()*-0.539107);
}

double ccNNEE::synapse0x19a4ef80() {
   return (neuron0x19a4aef0()*-0.844427);
}

double ccNNEE::synapse0x19a4efc0() {
   return (neuron0x19a4b430()*1.73929);
}

double ccNNEE::synapse0x19a4f000() {
   return (neuron0x19a4b7b0()*1.19979);
}

double ccNNEE::synapse0x19a4f040() {
   return (neuron0x19a4bcf0()*0.350089);
}

double ccNNEE::synapse0x19a4f080() {
   return (neuron0x19a4c230()*1.25415);
}

double ccNNEE::synapse0x19a4f0c0() {
   return (neuron0x19a4c770()*-0.387113);
}

double ccNNEE::synapse0x19a4cbc0() {
   return (neuron0x19a4cfd0()*0.61157);
}

double ccNNEE::synapse0x19a4cf40() {
   return (neuron0x19a4d480()*4.82065);
}

double ccNNEE::synapse0x19a4cf80() {
   return (neuron0x19a4da40()*2.0913);
}

double ccNNEE::synapse0x19a49e80() {
   return (neuron0x19a4e000()*2.28334);
}

double ccNNEE::synapse0x19a49ec0() {
   return (neuron0x19a4e5c0()*-1.85272);
}

double ccNNEE::synapse0x19a49f00() {
   return (neuron0x19a4eb80()*2.63883);
}

