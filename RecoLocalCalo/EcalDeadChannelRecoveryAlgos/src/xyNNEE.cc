#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/xyNNEE.h"
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

double ldNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.40353)/1.40003;
   input2 = (in2 - 2.41121)/1.41004;
   input3 = (in3 - 2.42657)/1.40629;
   input4 = (in4 - 2.42619)/1.40466;
   input5 = (in5 - 1.33856)/1.28698;
   input6 = (in6 - 1.33177)/1.28879;
   input7 = (in7 - 1.33367)/1.29347;
   switch(index) {
     case 0:
         return neuron0x1e95bcc0();
     default:
         return 0.;
   }
}

double ldNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.40353)/1.40003;
   input2 = (input[2] - 2.41121)/1.41004;
   input3 = (input[3] - 2.42657)/1.40629;
   input4 = (input[4] - 2.42619)/1.40466;
   input5 = (input[5] - 1.33856)/1.28698;
   input6 = (input[6] - 1.33177)/1.28879;
   input7 = (input[7] - 1.33367)/1.29347;
   switch(index) {
     case 0:
         return neuron0x1e95bcc0();
     default:
         return 0.;
   }
}

double ldNNEE::neuron0x1e957540() {
   return input0;
}

double ldNNEE::neuron0x1e957880() {
   return input1;
}

double ldNNEE::neuron0x1e957bc0() {
   return input2;
}

double ldNNEE::neuron0x1e957f00() {
   return input3;
}

double ldNNEE::neuron0x1e958240() {
   return input4;
}

double ldNNEE::neuron0x1e958580() {
   return input5;
}

double ldNNEE::neuron0x1e9588c0() {
   return input6;
}

double ldNNEE::neuron0x1e958c00() {
   return input7;
}

double ldNNEE::input0x1e959070() {
   double input = -1.56243;
   input += synapse0x1e8b7880();
   input += synapse0x1e960140();
   input += synapse0x1e959320();
   input += synapse0x1e959360();
   input += synapse0x1e9593a0();
   input += synapse0x1e9593e0();
   input += synapse0x1e959420();
   input += synapse0x1e959460();
   return input;
}

double ldNNEE::neuron0x1e959070() {
   double input = input0x1e959070();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e9594a0() {
   double input = 0.277395;
   input += synapse0x1e9597e0();
   input += synapse0x1e959820();
   input += synapse0x1e959860();
   input += synapse0x1e9598a0();
   input += synapse0x1e9598e0();
   input += synapse0x1e959920();
   input += synapse0x1e959960();
   input += synapse0x1e9599a0();
   return input;
}

double ldNNEE::neuron0x1e9594a0() {
   double input = input0x1e9594a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e9599e0() {
   double input = -2.09409;
   input += synapse0x1e959d20();
   input += synapse0x1e885e20();
   input += synapse0x1e885e60();
   input += synapse0x1e959e70();
   input += synapse0x1e959eb0();
   input += synapse0x1e959ef0();
   input += synapse0x1e959f30();
   input += synapse0x1e959f70();
   return input;
}

double ldNNEE::neuron0x1e9599e0() {
   double input = input0x1e9599e0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e959fb0() {
   double input = -0.665299;
   input += synapse0x1e95a2f0();
   input += synapse0x1e95a330();
   input += synapse0x1e95a370();
   input += synapse0x1e95a3b0();
   input += synapse0x1e95a3f0();
   input += synapse0x1e95a430();
   input += synapse0x1e95a470();
   input += synapse0x1e95a4b0();
   return input;
}

double ldNNEE::neuron0x1e959fb0() {
   double input = input0x1e959fb0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95a4f0() {
   double input = -0.112128;
   input += synapse0x1e95a830();
   input += synapse0x1e957470();
   input += synapse0x1e960180();
   input += synapse0x1e8a2190();
   input += synapse0x1e959d60();
   input += synapse0x1e959da0();
   input += synapse0x1e959de0();
   input += synapse0x1e959e20();
   return input;
}

double ldNNEE::neuron0x1e95a4f0() {
   double input = input0x1e95a4f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95a870() {
   double input = 0.0424555;
   input += synapse0x1e95abb0();
   input += synapse0x1e95abf0();
   input += synapse0x1e95ac30();
   input += synapse0x1e95ac70();
   input += synapse0x1e95acb0();
   input += synapse0x1e95acf0();
   input += synapse0x1e95ad30();
   input += synapse0x1e95ad70();
   return input;
}

double ldNNEE::neuron0x1e95a870() {
   double input = input0x1e95a870();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95adb0() {
   double input = 0.783657;
   input += synapse0x1e95b0f0();
   input += synapse0x1e95b130();
   input += synapse0x1e95b170();
   input += synapse0x1e95b1b0();
   input += synapse0x1e95b1f0();
   input += synapse0x1e95b230();
   input += synapse0x1e95b270();
   input += synapse0x1e95b2b0();
   return input;
}

double ldNNEE::neuron0x1e95adb0() {
   double input = input0x1e95adb0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95b2f0() {
   double input = -0.327327;
   input += synapse0x1e95b630();
   input += synapse0x1e95b670();
   input += synapse0x1e95b6b0();
   input += synapse0x1e95b6f0();
   input += synapse0x1e95b730();
   input += synapse0x1e95b770();
   input += synapse0x1e95b7b0();
   input += synapse0x1e95b7f0();
   return input;
}

double ldNNEE::neuron0x1e95b2f0() {
   double input = input0x1e95b2f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95b830() {
   double input = 2.23142;
   input += synapse0x1e883a80();
   input += synapse0x1e883ac0();
   input += synapse0x1e89e980();
   input += synapse0x1e89e9c0();
   input += synapse0x1e89ea00();
   input += synapse0x1e89ea40();
   input += synapse0x1e89ea80();
   input += synapse0x1e89eac0();
   return input;
}

double ldNNEE::neuron0x1e95b830() {
   double input = input0x1e95b830();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95c090() {
   double input = 2.37996;
   input += synapse0x1e95c340();
   input += synapse0x1e95c380();
   input += synapse0x1e95c3c0();
   input += synapse0x1e95c400();
   input += synapse0x1e95c440();
   input += synapse0x1e95c480();
   input += synapse0x1e95c4c0();
   input += synapse0x1e95c500();
   return input;
}

double ldNNEE::neuron0x1e95c090() {
   double input = input0x1e95c090();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95c540() {
   double input = -0.959941;
   input += synapse0x1e95c880();
   input += synapse0x1e95c8c0();
   input += synapse0x1e95c900();
   input += synapse0x1e95c940();
   input += synapse0x1e95c980();
   input += synapse0x1e95c9c0();
   input += synapse0x1e95ca00();
   input += synapse0x1e95ca40();
   input += synapse0x1e95ca80();
   input += synapse0x1e95cac0();
   return input;
}

double ldNNEE::neuron0x1e95c540() {
   double input = input0x1e95c540();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95cb00() {
   double input = -0.0545535;
   input += synapse0x1e95ce40();
   input += synapse0x1e95ce80();
   input += synapse0x1e95cec0();
   input += synapse0x1e95cf00();
   input += synapse0x1e95cf40();
   input += synapse0x1e95cf80();
   input += synapse0x1e95cfc0();
   input += synapse0x1e95d000();
   input += synapse0x1e95d040();
   input += synapse0x1e95d080();
   return input;
}

double ldNNEE::neuron0x1e95cb00() {
   double input = input0x1e95cb00();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95d0c0() {
   double input = 0.108088;
   input += synapse0x1e95d400();
   input += synapse0x1e95d440();
   input += synapse0x1e95d480();
   input += synapse0x1e95d4c0();
   input += synapse0x1e95d500();
   input += synapse0x1e95d540();
   input += synapse0x1e95d580();
   input += synapse0x1e95d5c0();
   input += synapse0x1e95d600();
   input += synapse0x1e95d640();
   return input;
}

double ldNNEE::neuron0x1e95d0c0() {
   double input = input0x1e95d0c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95d680() {
   double input = -0.0100983;
   input += synapse0x1e95d9c0();
   input += synapse0x1e95da00();
   input += synapse0x1e95da40();
   input += synapse0x1e95da80();
   input += synapse0x1e95dac0();
   input += synapse0x1e95db00();
   input += synapse0x1e95db40();
   input += synapse0x1e95db80();
   input += synapse0x1e95dbc0();
   input += synapse0x1e95dc00();
   return input;
}

double ldNNEE::neuron0x1e95d680() {
   double input = input0x1e95d680();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95dc40() {
   double input = -0.534084;
   input += synapse0x1e95df80();
   input += synapse0x1e95dfc0();
   input += synapse0x1e95e000();
   input += synapse0x1e95e040();
   input += synapse0x1e95e080();
   input += synapse0x1e95e0c0();
   input += synapse0x1e95e100();
   input += synapse0x1e95e140();
   input += synapse0x1e95e180();
   input += synapse0x1e95bc80();
   return input;
}

double ldNNEE::neuron0x1e95dc40() {
   double input = input0x1e95dc40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double ldNNEE::input0x1e95bcc0() {
   double input = -1.46932;
   input += synapse0x1e95c000();
   input += synapse0x1e95c040();
   input += synapse0x1e958f40();
   input += synapse0x1e958f80();
   input += synapse0x1e958fc0();
   return input;
}

double ldNNEE::neuron0x1e95bcc0() {
   double input = input0x1e95bcc0();
   return (input * 1)+0;
}

double ldNNEE::synapse0x1e8b7880() {
   return (neuron0x1e957540()*1.46564);
}

double ldNNEE::synapse0x1e960140() {
   return (neuron0x1e957880()*0.428584);
}

double ldNNEE::synapse0x1e959320() {
   return (neuron0x1e957bc0()*0.582542);
}

double ldNNEE::synapse0x1e959360() {
   return (neuron0x1e957f00()*-0.787778);
}

double ldNNEE::synapse0x1e9593a0() {
   return (neuron0x1e958240()*-1.76293);
}

double ldNNEE::synapse0x1e9593e0() {
   return (neuron0x1e958580()*0.173838);
}

double ldNNEE::synapse0x1e959420() {
   return (neuron0x1e9588c0()*-0.902033);
}

double ldNNEE::synapse0x1e959460() {
   return (neuron0x1e958c00()*-0.089348);
}

double ldNNEE::synapse0x1e9597e0() {
   return (neuron0x1e957540()*1.41247);
}

double ldNNEE::synapse0x1e959820() {
   return (neuron0x1e957880()*-0.042349);
}

double ldNNEE::synapse0x1e959860() {
   return (neuron0x1e957bc0()*-0.7785);
}

double ldNNEE::synapse0x1e9598a0() {
   return (neuron0x1e957f00()*-0.210301);
}

double ldNNEE::synapse0x1e9598e0() {
   return (neuron0x1e958240()*-1.17315);
}

double ldNNEE::synapse0x1e959920() {
   return (neuron0x1e958580()*-0.195878);
}

double ldNNEE::synapse0x1e959960() {
   return (neuron0x1e9588c0()*1.3596);
}

double ldNNEE::synapse0x1e9599a0() {
   return (neuron0x1e958c00()*1.28309);
}

double ldNNEE::synapse0x1e959d20() {
   return (neuron0x1e957540()*1.15124);
}

double ldNNEE::synapse0x1e885e20() {
   return (neuron0x1e957880()*-0.0668577);
}

double ldNNEE::synapse0x1e885e60() {
   return (neuron0x1e957bc0()*0.408767);
}

double ldNNEE::synapse0x1e959e70() {
   return (neuron0x1e957f00()*-0.383822);
}

double ldNNEE::synapse0x1e959eb0() {
   return (neuron0x1e958240()*-0.350844);
}

double ldNNEE::synapse0x1e959ef0() {
   return (neuron0x1e958580()*-0.658633);
}

double ldNNEE::synapse0x1e959f30() {
   return (neuron0x1e9588c0()*0.253153);
}

double ldNNEE::synapse0x1e959f70() {
   return (neuron0x1e958c00()*0.450355);
}

double ldNNEE::synapse0x1e95a2f0() {
   return (neuron0x1e957540()*0.533386);
}

double ldNNEE::synapse0x1e95a330() {
   return (neuron0x1e957880()*-0.0825323);
}

double ldNNEE::synapse0x1e95a370() {
   return (neuron0x1e957bc0()*-0.690911);
}

double ldNNEE::synapse0x1e95a3b0() {
   return (neuron0x1e957f00()*0.216972);
}

double ldNNEE::synapse0x1e95a3f0() {
   return (neuron0x1e958240()*0.753513);
}

double ldNNEE::synapse0x1e95a430() {
   return (neuron0x1e958580()*0.0971258);
}

double ldNNEE::synapse0x1e95a470() {
   return (neuron0x1e9588c0()*0.355891);
}

double ldNNEE::synapse0x1e95a4b0() {
   return (neuron0x1e958c00()*0.62749);
}

double ldNNEE::synapse0x1e95a830() {
   return (neuron0x1e957540()*2.40699);
}

double ldNNEE::synapse0x1e957470() {
   return (neuron0x1e957880()*-1.59664);
}

double ldNNEE::synapse0x1e960180() {
   return (neuron0x1e957bc0()*-0.865839);
}

double ldNNEE::synapse0x1e8a2190() {
   return (neuron0x1e957f00()*-1.03403);
}

double ldNNEE::synapse0x1e959d60() {
   return (neuron0x1e958240()*-0.0327801);
}

double ldNNEE::synapse0x1e959da0() {
   return (neuron0x1e958580()*-0.883803);
}

double ldNNEE::synapse0x1e959de0() {
   return (neuron0x1e9588c0()*0.919519);
}

double ldNNEE::synapse0x1e959e20() {
   return (neuron0x1e958c00()*1.463);
}

double ldNNEE::synapse0x1e95abb0() {
   return (neuron0x1e957540()*-0.0746568);
}

double ldNNEE::synapse0x1e95abf0() {
   return (neuron0x1e957880()*-0.332818);
}

double ldNNEE::synapse0x1e95ac30() {
   return (neuron0x1e957bc0()*-1.35063);
}

double ldNNEE::synapse0x1e95ac70() {
   return (neuron0x1e957f00()*0.183133);
}

double ldNNEE::synapse0x1e95acb0() {
   return (neuron0x1e958240()*0.134358);
}

double ldNNEE::synapse0x1e95acf0() {
   return (neuron0x1e958580()*-0.491049);
}

double ldNNEE::synapse0x1e95ad30() {
   return (neuron0x1e9588c0()*1.64453);
}

double ldNNEE::synapse0x1e95ad70() {
   return (neuron0x1e958c00()*0.67541);
}

double ldNNEE::synapse0x1e95b0f0() {
   return (neuron0x1e957540()*0.434697);
}

double ldNNEE::synapse0x1e95b130() {
   return (neuron0x1e957880()*-0.360692);
}

double ldNNEE::synapse0x1e95b170() {
   return (neuron0x1e957bc0()*0.217958);
}

double ldNNEE::synapse0x1e95b1b0() {
   return (neuron0x1e957f00()*0.0690149);
}

double ldNNEE::synapse0x1e95b1f0() {
   return (neuron0x1e958240()*1.07911);
}

double ldNNEE::synapse0x1e95b230() {
   return (neuron0x1e958580()*0.461768);
}

double ldNNEE::synapse0x1e95b270() {
   return (neuron0x1e9588c0()*-0.778333);
}

double ldNNEE::synapse0x1e95b2b0() {
   return (neuron0x1e958c00()*-1.37221);
}

double ldNNEE::synapse0x1e95b630() {
   return (neuron0x1e957540()*-0.885185);
}

double ldNNEE::synapse0x1e95b670() {
   return (neuron0x1e957880()*-0.146735);
}

double ldNNEE::synapse0x1e95b6b0() {
   return (neuron0x1e957bc0()*1.0352);
}

double ldNNEE::synapse0x1e95b6f0() {
   return (neuron0x1e957f00()*0.00272632);
}

double ldNNEE::synapse0x1e95b730() {
   return (neuron0x1e958240()*0.761924);
}

double ldNNEE::synapse0x1e95b770() {
   return (neuron0x1e958580()*0.0672013);
}

double ldNNEE::synapse0x1e95b7b0() {
   return (neuron0x1e9588c0()*-0.640816);
}

double ldNNEE::synapse0x1e95b7f0() {
   return (neuron0x1e958c00()*-0.691704);
}

double ldNNEE::synapse0x1e883a80() {
   return (neuron0x1e957540()*-0.384238);
}

double ldNNEE::synapse0x1e883ac0() {
   return (neuron0x1e957880()*0.397007);
}

double ldNNEE::synapse0x1e89e980() {
   return (neuron0x1e957bc0()*-0.39839);
}

double ldNNEE::synapse0x1e89e9c0() {
   return (neuron0x1e957f00()*-0.00745285);
}

double ldNNEE::synapse0x1e89ea00() {
   return (neuron0x1e958240()*-0.413446);
}

double ldNNEE::synapse0x1e89ea40() {
   return (neuron0x1e958580()*-0.243978);
}

double ldNNEE::synapse0x1e89ea80() {
   return (neuron0x1e9588c0()*-0.500086);
}

double ldNNEE::synapse0x1e89eac0() {
   return (neuron0x1e958c00()*-0.178044);
}

double ldNNEE::synapse0x1e95c340() {
   return (neuron0x1e957540()*-0.323586);
}

double ldNNEE::synapse0x1e95c380() {
   return (neuron0x1e957880()*0.501998);
}

double ldNNEE::synapse0x1e95c3c0() {
   return (neuron0x1e957bc0()*1.51388);
}

double ldNNEE::synapse0x1e95c400() {
   return (neuron0x1e957f00()*-0.879529);
}

double ldNNEE::synapse0x1e95c440() {
   return (neuron0x1e958240()*-0.805979);
}

double ldNNEE::synapse0x1e95c480() {
   return (neuron0x1e958580()*-0.461876);
}

double ldNNEE::synapse0x1e95c4c0() {
   return (neuron0x1e9588c0()*0.0814574);
}

double ldNNEE::synapse0x1e95c500() {
   return (neuron0x1e958c00()*0.690482);
}

double ldNNEE::synapse0x1e95c880() {
   return (neuron0x1e959070()*-0.92281);
}

double ldNNEE::synapse0x1e95c8c0() {
   return (neuron0x1e9594a0()*-1.65121);
}

double ldNNEE::synapse0x1e95c900() {
   return (neuron0x1e9599e0()*1.7076);
}

double ldNNEE::synapse0x1e95c940() {
   return (neuron0x1e959fb0()*0.799588);
}

double ldNNEE::synapse0x1e95c980() {
   return (neuron0x1e95a4f0()*-1.21932);
}

double ldNNEE::synapse0x1e95c9c0() {
   return (neuron0x1e95a870()*-0.265466);
}

double ldNNEE::synapse0x1e95ca00() {
   return (neuron0x1e95adb0()*1.20421);
}

double ldNNEE::synapse0x1e95ca40() {
   return (neuron0x1e95b2f0()*0.585932);
}

double ldNNEE::synapse0x1e95ca80() {
   return (neuron0x1e95b830()*-1.8946);
}

double ldNNEE::synapse0x1e95cac0() {
   return (neuron0x1e95c090()*1.08486);
}

double ldNNEE::synapse0x1e95ce40() {
   return (neuron0x1e959070()*0.764949);
}

double ldNNEE::synapse0x1e95ce80() {
   return (neuron0x1e9594a0()*-1.77857);
}

double ldNNEE::synapse0x1e95cec0() {
   return (neuron0x1e9599e0()*-1.44957);
}

double ldNNEE::synapse0x1e95cf00() {
   return (neuron0x1e959fb0()*-1.41676);
}

double ldNNEE::synapse0x1e95cf40() {
   return (neuron0x1e95a4f0()*-0.458544);
}

double ldNNEE::synapse0x1e95cf80() {
   return (neuron0x1e95a870()*0.528898);
}

double ldNNEE::synapse0x1e95cfc0() {
   return (neuron0x1e95adb0()*-0.750037);
}

double ldNNEE::synapse0x1e95d000() {
   return (neuron0x1e95b2f0()*-0.764525);
}

double ldNNEE::synapse0x1e95d040() {
   return (neuron0x1e95b830()*0.371361);
}

double ldNNEE::synapse0x1e95d080() {
   return (neuron0x1e95c090()*-0.399181);
}

double ldNNEE::synapse0x1e95d400() {
   return (neuron0x1e959070()*-0.376096);
}

double ldNNEE::synapse0x1e95d440() {
   return (neuron0x1e9594a0()*0.137658);
}

double ldNNEE::synapse0x1e95d480() {
   return (neuron0x1e9599e0()*0.298105);
}

double ldNNEE::synapse0x1e95d4c0() {
   return (neuron0x1e959fb0()*0.105747);
}

double ldNNEE::synapse0x1e95d500() {
   return (neuron0x1e95a4f0()*-0.738776);
}

double ldNNEE::synapse0x1e95d540() {
   return (neuron0x1e95a870()*-0.216621);
}

double ldNNEE::synapse0x1e95d580() {
   return (neuron0x1e95adb0()*0.522074);
}

double ldNNEE::synapse0x1e95d5c0() {
   return (neuron0x1e95b2f0()*-0.133003);
}

double ldNNEE::synapse0x1e95d600() {
   return (neuron0x1e95b830()*-0.116397);
}

double ldNNEE::synapse0x1e95d640() {
   return (neuron0x1e95c090()*0.643543);
}

double ldNNEE::synapse0x1e95d9c0() {
   return (neuron0x1e959070()*-0.684852);
}

double ldNNEE::synapse0x1e95da00() {
   return (neuron0x1e9594a0()*0.556556);
}

double ldNNEE::synapse0x1e95da40() {
   return (neuron0x1e9599e0()*1.16285);
}

double ldNNEE::synapse0x1e95da80() {
   return (neuron0x1e959fb0()*1.13189);
}

double ldNNEE::synapse0x1e95dac0() {
   return (neuron0x1e95a4f0()*-1.5021);
}

double ldNNEE::synapse0x1e95db00() {
   return (neuron0x1e95a870()*-0.509703);
}

double ldNNEE::synapse0x1e95db40() {
   return (neuron0x1e95adb0()*0.980433);
}

double ldNNEE::synapse0x1e95db80() {
   return (neuron0x1e95b2f0()*0.240291);
}

double ldNNEE::synapse0x1e95dbc0() {
   return (neuron0x1e95b830()*-1.16458);
}

double ldNNEE::synapse0x1e95dc00() {
   return (neuron0x1e95c090()*0.598338);
}

double ldNNEE::synapse0x1e95df80() {
   return (neuron0x1e959070()*-1.05336);
}

double ldNNEE::synapse0x1e95dfc0() {
   return (neuron0x1e9594a0()*0.897677);
}

double ldNNEE::synapse0x1e95e000() {
   return (neuron0x1e9599e0()*2.32913);
}

double ldNNEE::synapse0x1e95e040() {
   return (neuron0x1e959fb0()*0.837162);
}

double ldNNEE::synapse0x1e95e080() {
   return (neuron0x1e95a4f0()*0.0385715);
}

double ldNNEE::synapse0x1e95e0c0() {
   return (neuron0x1e95a870()*-1.22258);
}

double ldNNEE::synapse0x1e95e100() {
   return (neuron0x1e95adb0()*0.946005);
}

double ldNNEE::synapse0x1e95e140() {
   return (neuron0x1e95b2f0()*1.22504);
}

double ldNNEE::synapse0x1e95e180() {
   return (neuron0x1e95b830()*-1.85501);
}

double ldNNEE::synapse0x1e95bc80() {
   return (neuron0x1e95c090()*1.77983);
}

double ldNNEE::synapse0x1e95c000() {
   return (neuron0x1e95c540()*4.10013);
}

double ldNNEE::synapse0x1e95c040() {
   return (neuron0x1e95cb00()*-3.6466);
}

double ldNNEE::synapse0x1e958f40() {
   return (neuron0x1e95d0c0()*-0.333399);
}

double ldNNEE::synapse0x1e958f80() {
   return (neuron0x1e95d680()*1.42134);
}

double ldNNEE::synapse0x1e958fc0() {
   return (neuron0x1e95dc40()*3.14812);
}

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

double rdNNEE::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.99102)/1.1492;
   input1 = (in1 - 2.40353)/1.40003;
   input2 = (in2 - 2.41121)/1.41004;
   input3 = (in3 - 2.42657)/1.40629;
   input4 = (in4 - 2.42619)/1.40466;
   input5 = (in5 - 1.33856)/1.28698;
   input6 = (in6 - 1.33367)/1.29347;
   input7 = (in7 - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x1ff97d40();
     default:
         return 0.;
   }
}

double rdNNEE::Value(int index, double* input) {
   input0 = (input[0] - 4.99102)/1.1492;
   input1 = (input[1] - 2.40353)/1.40003;
   input2 = (input[2] - 2.41121)/1.41004;
   input3 = (input[3] - 2.42657)/1.40629;
   input4 = (input[4] - 2.42619)/1.40466;
   input5 = (input[5] - 1.33856)/1.28698;
   input6 = (input[6] - 1.33367)/1.29347;
   input7 = (input[7] - 1.34123)/1.29317;
   switch(index) {
     case 0:
         return neuron0x1ff97d40();
     default:
         return 0.;
   }
}

double rdNNEE::neuron0x1ff935c0() {
   return input0;
}

double rdNNEE::neuron0x1ff93900() {
   return input1;
}

double rdNNEE::neuron0x1ff93c40() {
   return input2;
}

double rdNNEE::neuron0x1ff93f80() {
   return input3;
}

double rdNNEE::neuron0x1ff942c0() {
   return input4;
}

double rdNNEE::neuron0x1ff94600() {
   return input5;
}

double rdNNEE::neuron0x1ff94940() {
   return input6;
}

double rdNNEE::neuron0x1ff94c80() {
   return input7;
}

double rdNNEE::input0x1ff950f0() {
   double input = 0.825175;
   input += synapse0x1fef3900();
   input += synapse0x1ff9c1c0();
   input += synapse0x1ff953a0();
   input += synapse0x1ff953e0();
   input += synapse0x1ff95420();
   input += synapse0x1ff95460();
   input += synapse0x1ff954a0();
   input += synapse0x1ff954e0();
   return input;
}

double rdNNEE::neuron0x1ff950f0() {
   double input = input0x1ff950f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff95520() {
   double input = 2.54534;
   input += synapse0x1ff95860();
   input += synapse0x1ff958a0();
   input += synapse0x1ff958e0();
   input += synapse0x1ff95920();
   input += synapse0x1ff95960();
   input += synapse0x1ff959a0();
   input += synapse0x1ff959e0();
   input += synapse0x1ff95a20();
   return input;
}

double rdNNEE::neuron0x1ff95520() {
   double input = input0x1ff95520();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff95a60() {
   double input = 2.28817;
   input += synapse0x1ff95da0();
   input += synapse0x1fec1ea0();
   input += synapse0x1fec1ee0();
   input += synapse0x1ff95ef0();
   input += synapse0x1ff95f30();
   input += synapse0x1ff95f70();
   input += synapse0x1ff95fb0();
   input += synapse0x1ff95ff0();
   return input;
}

double rdNNEE::neuron0x1ff95a60() {
   double input = input0x1ff95a60();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff96030() {
   double input = -0.815879;
   input += synapse0x1ff96370();
   input += synapse0x1ff963b0();
   input += synapse0x1ff963f0();
   input += synapse0x1ff96430();
   input += synapse0x1ff96470();
   input += synapse0x1ff964b0();
   input += synapse0x1ff964f0();
   input += synapse0x1ff96530();
   return input;
}

double rdNNEE::neuron0x1ff96030() {
   double input = input0x1ff96030();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff96570() {
   double input = -1.58742;
   input += synapse0x1ff968b0();
   input += synapse0x1ff934f0();
   input += synapse0x1ff9c200();
   input += synapse0x1fede210();
   input += synapse0x1ff95de0();
   input += synapse0x1ff95e20();
   input += synapse0x1ff95e60();
   input += synapse0x1ff95ea0();
   return input;
}

double rdNNEE::neuron0x1ff96570() {
   double input = input0x1ff96570();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff968f0() {
   double input = 1.38619;
   input += synapse0x1ff96c30();
   input += synapse0x1ff96c70();
   input += synapse0x1ff96cb0();
   input += synapse0x1ff96cf0();
   input += synapse0x1ff96d30();
   input += synapse0x1ff96d70();
   input += synapse0x1ff96db0();
   input += synapse0x1ff96df0();
   return input;
}

double rdNNEE::neuron0x1ff968f0() {
   double input = input0x1ff968f0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff96e30() {
   double input = 0.892159;
   input += synapse0x1ff97170();
   input += synapse0x1ff971b0();
   input += synapse0x1ff971f0();
   input += synapse0x1ff97230();
   input += synapse0x1ff97270();
   input += synapse0x1ff972b0();
   input += synapse0x1ff972f0();
   input += synapse0x1ff97330();
   return input;
}

double rdNNEE::neuron0x1ff96e30() {
   double input = input0x1ff96e30();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff97370() {
   double input = 0.996943;
   input += synapse0x1ff976b0();
   input += synapse0x1ff976f0();
   input += synapse0x1ff97730();
   input += synapse0x1ff97770();
   input += synapse0x1ff977b0();
   input += synapse0x1ff977f0();
   input += synapse0x1ff97830();
   input += synapse0x1ff97870();
   return input;
}

double rdNNEE::neuron0x1ff97370() {
   double input = input0x1ff97370();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff978b0() {
   double input = -0.282028;
   input += synapse0x1febfb00();
   input += synapse0x1febfb40();
   input += synapse0x1fedaa00();
   input += synapse0x1fedaa40();
   input += synapse0x1fedaa80();
   input += synapse0x1fedaac0();
   input += synapse0x1fedab00();
   input += synapse0x1fedab40();
   return input;
}

double rdNNEE::neuron0x1ff978b0() {
   double input = input0x1ff978b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff98110() {
   double input = 1.65324;
   input += synapse0x1ff983c0();
   input += synapse0x1ff98400();
   input += synapse0x1ff98440();
   input += synapse0x1ff98480();
   input += synapse0x1ff984c0();
   input += synapse0x1ff98500();
   input += synapse0x1ff98540();
   input += synapse0x1ff98580();
   return input;
}

double rdNNEE::neuron0x1ff98110() {
   double input = input0x1ff98110();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff985c0() {
   double input = 0.239174;
   input += synapse0x1ff98900();
   input += synapse0x1ff98940();
   input += synapse0x1ff98980();
   input += synapse0x1ff989c0();
   input += synapse0x1ff98a00();
   input += synapse0x1ff98a40();
   input += synapse0x1ff98a80();
   input += synapse0x1ff98ac0();
   input += synapse0x1ff98b00();
   input += synapse0x1ff98b40();
   return input;
}

double rdNNEE::neuron0x1ff985c0() {
   double input = input0x1ff985c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff98b80() {
   double input = 0.843606;
   input += synapse0x1ff98ec0();
   input += synapse0x1ff98f00();
   input += synapse0x1ff98f40();
   input += synapse0x1ff98f80();
   input += synapse0x1ff98fc0();
   input += synapse0x1ff99000();
   input += synapse0x1ff99040();
   input += synapse0x1ff99080();
   input += synapse0x1ff990c0();
   input += synapse0x1ff99100();
   return input;
}

double rdNNEE::neuron0x1ff98b80() {
   double input = input0x1ff98b80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff99140() {
   double input = -0.0990648;
   input += synapse0x1ff99480();
   input += synapse0x1ff994c0();
   input += synapse0x1ff99500();
   input += synapse0x1ff99540();
   input += synapse0x1ff99580();
   input += synapse0x1ff995c0();
   input += synapse0x1ff99600();
   input += synapse0x1ff99640();
   input += synapse0x1ff99680();
   input += synapse0x1ff996c0();
   return input;
}

double rdNNEE::neuron0x1ff99140() {
   double input = input0x1ff99140();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff99700() {
   double input = -0.896219;
   input += synapse0x1ff99a40();
   input += synapse0x1ff99a80();
   input += synapse0x1ff99ac0();
   input += synapse0x1ff99b00();
   input += synapse0x1ff99b40();
   input += synapse0x1ff99b80();
   input += synapse0x1ff99bc0();
   input += synapse0x1ff99c00();
   input += synapse0x1ff99c40();
   input += synapse0x1ff99c80();
   return input;
}

double rdNNEE::neuron0x1ff99700() {
   double input = input0x1ff99700();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff99cc0() {
   double input = -0.436962;
   input += synapse0x1ff9a000();
   input += synapse0x1ff9a040();
   input += synapse0x1ff9a080();
   input += synapse0x1ff9a0c0();
   input += synapse0x1ff9a100();
   input += synapse0x1ff9a140();
   input += synapse0x1ff9a180();
   input += synapse0x1ff9a1c0();
   input += synapse0x1ff9a200();
   input += synapse0x1ff97d00();
   return input;
}

double rdNNEE::neuron0x1ff99cc0() {
   double input = input0x1ff99cc0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEE::input0x1ff97d40() {
   double input = -1.43615;
   input += synapse0x1ff98080();
   input += synapse0x1ff980c0();
   input += synapse0x1ff94fc0();
   input += synapse0x1ff95000();
   input += synapse0x1ff95040();
   return input;
}

double rdNNEE::neuron0x1ff97d40() {
   double input = input0x1ff97d40();
   return (input * 1)+0;
}

double rdNNEE::synapse0x1fef3900() {
   return (neuron0x1ff935c0()*1.00654);
}

double rdNNEE::synapse0x1ff9c1c0() {
   return (neuron0x1ff93900()*-0.782609);
}

double rdNNEE::synapse0x1ff953a0() {
   return (neuron0x1ff93c40()*0.0940371);
}

double rdNNEE::synapse0x1ff953e0() {
   return (neuron0x1ff93f80()*0.262207);
}

double rdNNEE::synapse0x1ff95420() {
   return (neuron0x1ff942c0()*-0.589799);
}

double rdNNEE::synapse0x1ff95460() {
   return (neuron0x1ff94600()*0.336342);
}

double rdNNEE::synapse0x1ff954a0() {
   return (neuron0x1ff94940()*-0.00159879);
}

double rdNNEE::synapse0x1ff954e0() {
   return (neuron0x1ff94c80()*0.510516);
}

double rdNNEE::synapse0x1ff95860() {
   return (neuron0x1ff935c0()*-0.839079);
}

double rdNNEE::synapse0x1ff958a0() {
   return (neuron0x1ff93900()*0.113118);
}

double rdNNEE::synapse0x1ff958e0() {
   return (neuron0x1ff93c40()*0.283235);
}

double rdNNEE::synapse0x1ff95920() {
   return (neuron0x1ff93f80()*-0.0827286);
}

double rdNNEE::synapse0x1ff95960() {
   return (neuron0x1ff942c0()*-0.673746);
}

double rdNNEE::synapse0x1ff959a0() {
   return (neuron0x1ff94600()*-0.160767);
}

double rdNNEE::synapse0x1ff959e0() {
   return (neuron0x1ff94940()*1.17374);
}

double rdNNEE::synapse0x1ff95a20() {
   return (neuron0x1ff94c80()*-0.451086);
}

double rdNNEE::synapse0x1ff95da0() {
   return (neuron0x1ff935c0()*-1.78628);
}

double rdNNEE::synapse0x1fec1ea0() {
   return (neuron0x1ff93900()*3.14715);
}

double rdNNEE::synapse0x1fec1ee0() {
   return (neuron0x1ff93c40()*1.2482);
}

double rdNNEE::synapse0x1ff95ef0() {
   return (neuron0x1ff93f80()*-1.06716);
}

double rdNNEE::synapse0x1ff95f30() {
   return (neuron0x1ff942c0()*-0.920541);
}

double rdNNEE::synapse0x1ff95f70() {
   return (neuron0x1ff94600()*0.611627);
}

double rdNNEE::synapse0x1ff95fb0() {
   return (neuron0x1ff94940()*0.0321043);
}

double rdNNEE::synapse0x1ff95ff0() {
   return (neuron0x1ff94c80()*-0.424328);
}

double rdNNEE::synapse0x1ff96370() {
   return (neuron0x1ff935c0()*-0.934525);
}

double rdNNEE::synapse0x1ff963b0() {
   return (neuron0x1ff93900()*0.227701);
}

double rdNNEE::synapse0x1ff963f0() {
   return (neuron0x1ff93c40()*-0.145405);
}

double rdNNEE::synapse0x1ff96430() {
   return (neuron0x1ff93f80()*-0.202652);
}

double rdNNEE::synapse0x1ff96470() {
   return (neuron0x1ff942c0()*0.0565631);
}

double rdNNEE::synapse0x1ff964b0() {
   return (neuron0x1ff94600()*-0.414642);
}

double rdNNEE::synapse0x1ff964f0() {
   return (neuron0x1ff94940()*-0.018005);
}

double rdNNEE::synapse0x1ff96530() {
   return (neuron0x1ff94c80()*-0.513357);
}

double rdNNEE::synapse0x1ff968b0() {
   return (neuron0x1ff935c0()*0.336392);
}

double rdNNEE::synapse0x1ff934f0() {
   return (neuron0x1ff93900()*0.326511);
}

double rdNNEE::synapse0x1ff9c200() {
   return (neuron0x1ff93c40()*0.0746277);
}

double rdNNEE::synapse0x1fede210() {
   return (neuron0x1ff93f80()*0.0145317);
}

double rdNNEE::synapse0x1ff95de0() {
   return (neuron0x1ff942c0()*0.181203);
}

double rdNNEE::synapse0x1ff95e20() {
   return (neuron0x1ff94600()*-0.0232318);
}

double rdNNEE::synapse0x1ff95e60() {
   return (neuron0x1ff94940()*0.401596);
}

double rdNNEE::synapse0x1ff95ea0() {
   return (neuron0x1ff94c80()*0.301376);
}

double rdNNEE::synapse0x1ff96c30() {
   return (neuron0x1ff935c0()*-0.24831);
}

double rdNNEE::synapse0x1ff96c70() {
   return (neuron0x1ff93900()*-0.709084);
}

double rdNNEE::synapse0x1ff96cb0() {
   return (neuron0x1ff93c40()*-0.498667);
}

double rdNNEE::synapse0x1ff96cf0() {
   return (neuron0x1ff93f80()*0.417819);
}

double rdNNEE::synapse0x1ff96d30() {
   return (neuron0x1ff942c0()*0.255772);
}

double rdNNEE::synapse0x1ff96d70() {
   return (neuron0x1ff94600()*0.652771);
}

double rdNNEE::synapse0x1ff96db0() {
   return (neuron0x1ff94940()*-0.791629);
}

double rdNNEE::synapse0x1ff96df0() {
   return (neuron0x1ff94c80()*0.908048);
}

double rdNNEE::synapse0x1ff97170() {
   return (neuron0x1ff935c0()*0.924764);
}

double rdNNEE::synapse0x1ff971b0() {
   return (neuron0x1ff93900()*0.683953);
}

double rdNNEE::synapse0x1ff971f0() {
   return (neuron0x1ff93c40()*-0.0863026);
}

double rdNNEE::synapse0x1ff97230() {
   return (neuron0x1ff93f80()*-0.619365);
}

double rdNNEE::synapse0x1ff97270() {
   return (neuron0x1ff942c0()*0.15908);
}

double rdNNEE::synapse0x1ff972b0() {
   return (neuron0x1ff94600()*-0.532884);
}

double rdNNEE::synapse0x1ff972f0() {
   return (neuron0x1ff94940()*0.658398);
}

double rdNNEE::synapse0x1ff97330() {
   return (neuron0x1ff94c80()*-1.49995);
}

double rdNNEE::synapse0x1ff976b0() {
   return (neuron0x1ff935c0()*-0.174238);
}

double rdNNEE::synapse0x1ff976f0() {
   return (neuron0x1ff93900()*-0.073891);
}

double rdNNEE::synapse0x1ff97730() {
   return (neuron0x1ff93c40()*-0.455685);
}

double rdNNEE::synapse0x1ff97770() {
   return (neuron0x1ff93f80()*0.505203);
}

double rdNNEE::synapse0x1ff977b0() {
   return (neuron0x1ff942c0()*1.64713);
}

double rdNNEE::synapse0x1ff977f0() {
   return (neuron0x1ff94600()*-1.3679);
}

double rdNNEE::synapse0x1ff97830() {
   return (neuron0x1ff94940()*0.281);
}

double rdNNEE::synapse0x1ff97870() {
   return (neuron0x1ff94c80()*-0.179998);
}

double rdNNEE::synapse0x1febfb00() {
   return (neuron0x1ff935c0()*1.21172);
}

double rdNNEE::synapse0x1febfb40() {
   return (neuron0x1ff93900()*-0.153712);
}

double rdNNEE::synapse0x1fedaa00() {
   return (neuron0x1ff93c40()*-0.677341);
}

double rdNNEE::synapse0x1fedaa40() {
   return (neuron0x1ff93f80()*-0.723119);
}

double rdNNEE::synapse0x1fedaa80() {
   return (neuron0x1ff942c0()*-0.0591842);
}

double rdNNEE::synapse0x1fedaac0() {
   return (neuron0x1ff94600()*0.923111);
}

double rdNNEE::synapse0x1fedab00() {
   return (neuron0x1ff94940()*-1.03907);
}

double rdNNEE::synapse0x1fedab40() {
   return (neuron0x1ff94c80()*0.706416);
}

double rdNNEE::synapse0x1ff983c0() {
   return (neuron0x1ff935c0()*-0.228062);
}

double rdNNEE::synapse0x1ff98400() {
   return (neuron0x1ff93900()*-0.864466);
}

double rdNNEE::synapse0x1ff98440() {
   return (neuron0x1ff93c40()*0.0626024);
}

double rdNNEE::synapse0x1ff98480() {
   return (neuron0x1ff93f80()*0.287105);
}

double rdNNEE::synapse0x1ff984c0() {
   return (neuron0x1ff942c0()*0.0330635);
}

double rdNNEE::synapse0x1ff98500() {
   return (neuron0x1ff94600()*-0.305153);
}

double rdNNEE::synapse0x1ff98540() {
   return (neuron0x1ff94940()*0.21622);
}

double rdNNEE::synapse0x1ff98580() {
   return (neuron0x1ff94c80()*-0.19366);
}

double rdNNEE::synapse0x1ff98900() {
   return (neuron0x1ff950f0()*-0.127884);
}

double rdNNEE::synapse0x1ff98940() {
   return (neuron0x1ff95520()*-1.04592);
}

double rdNNEE::synapse0x1ff98980() {
   return (neuron0x1ff95a60()*-0.199728);
}

double rdNNEE::synapse0x1ff989c0() {
   return (neuron0x1ff96030()*-0.00266199);
}

double rdNNEE::synapse0x1ff98a00() {
   return (neuron0x1ff96570()*0.74619);
}

double rdNNEE::synapse0x1ff98a40() {
   return (neuron0x1ff968f0()*0.115496);
}

double rdNNEE::synapse0x1ff98a80() {
   return (neuron0x1ff96e30()*0.582991);
}

double rdNNEE::synapse0x1ff98ac0() {
   return (neuron0x1ff97370()*0.619113);
}

double rdNNEE::synapse0x1ff98b00() {
   return (neuron0x1ff978b0()*-0.627176);
}

double rdNNEE::synapse0x1ff98b40() {
   return (neuron0x1ff98110()*-0.768379);
}

double rdNNEE::synapse0x1ff98ec0() {
   return (neuron0x1ff950f0()*0.266785);
}

double rdNNEE::synapse0x1ff98f00() {
   return (neuron0x1ff95520()*-1.03515);
}

double rdNNEE::synapse0x1ff98f40() {
   return (neuron0x1ff95a60()*-0.364548);
}

double rdNNEE::synapse0x1ff98f80() {
   return (neuron0x1ff96030()*-1.10946);
}

double rdNNEE::synapse0x1ff98fc0() {
   return (neuron0x1ff96570()*0.631055);
}

double rdNNEE::synapse0x1ff99000() {
   return (neuron0x1ff968f0()*0.719271);
}

double rdNNEE::synapse0x1ff99040() {
   return (neuron0x1ff96e30()*1.00781);
}

double rdNNEE::synapse0x1ff99080() {
   return (neuron0x1ff97370()*-0.160315);
}

double rdNNEE::synapse0x1ff990c0() {
   return (neuron0x1ff978b0()*-0.25212);
}

double rdNNEE::synapse0x1ff99100() {
   return (neuron0x1ff98110()*-0.704644);
}

double rdNNEE::synapse0x1ff99480() {
   return (neuron0x1ff950f0()*-1.80907);
}

double rdNNEE::synapse0x1ff994c0() {
   return (neuron0x1ff95520()*-0.751556);
}

double rdNNEE::synapse0x1ff99500() {
   return (neuron0x1ff95a60()*1.10668);
}

double rdNNEE::synapse0x1ff99540() {
   return (neuron0x1ff96030()*-1.35127);
}

double rdNNEE::synapse0x1ff99580() {
   return (neuron0x1ff96570()*0.815284);
}

double rdNNEE::synapse0x1ff995c0() {
   return (neuron0x1ff968f0()*1.85946);
}

double rdNNEE::synapse0x1ff99600() {
   return (neuron0x1ff96e30()*0.0871906);
}

double rdNNEE::synapse0x1ff99640() {
   return (neuron0x1ff97370()*1.30664);
}

double rdNNEE::synapse0x1ff99680() {
   return (neuron0x1ff978b0()*1.14415);
}

double rdNNEE::synapse0x1ff996c0() {
   return (neuron0x1ff98110()*-0.52077);
}

double rdNNEE::synapse0x1ff99a40() {
   return (neuron0x1ff950f0()*-1.95843);
}

double rdNNEE::synapse0x1ff99a80() {
   return (neuron0x1ff95520()*-0.838962);
}

double rdNNEE::synapse0x1ff99ac0() {
   return (neuron0x1ff95a60()*1.49395);
}

double rdNNEE::synapse0x1ff99b00() {
   return (neuron0x1ff96030()*0.022963);
}

double rdNNEE::synapse0x1ff99b40() {
   return (neuron0x1ff96570()*0.038856);
}

double rdNNEE::synapse0x1ff99b80() {
   return (neuron0x1ff968f0()*-0.257157);
}

double rdNNEE::synapse0x1ff99bc0() {
   return (neuron0x1ff96e30()*1.54999);
}

double rdNNEE::synapse0x1ff99c00() {
   return (neuron0x1ff97370()*2.12141);
}

double rdNNEE::synapse0x1ff99c40() {
   return (neuron0x1ff978b0()*-2.59816);
}

double rdNNEE::synapse0x1ff99c80() {
   return (neuron0x1ff98110()*-1.38224);
}

double rdNNEE::synapse0x1ff9a000() {
   return (neuron0x1ff950f0()*0.189974);
}

double rdNNEE::synapse0x1ff9a040() {
   return (neuron0x1ff95520()*0.481018);
}

double rdNNEE::synapse0x1ff9a080() {
   return (neuron0x1ff95a60()*-0.0162486);
}

double rdNNEE::synapse0x1ff9a0c0() {
   return (neuron0x1ff96030()*0.941632);
}

double rdNNEE::synapse0x1ff9a100() {
   return (neuron0x1ff96570()*-1.21962);
}

double rdNNEE::synapse0x1ff9a140() {
   return (neuron0x1ff968f0()*-0.509644);
}

double rdNNEE::synapse0x1ff9a180() {
   return (neuron0x1ff96e30()*-0.308659);
}

double rdNNEE::synapse0x1ff9a1c0() {
   return (neuron0x1ff97370()*0.61483);
}

double rdNNEE::synapse0x1ff9a200() {
   return (neuron0x1ff978b0()*0.715472);
}

double rdNNEE::synapse0x1ff97d00() {
   return (neuron0x1ff98110()*0.422625);
}

double rdNNEE::synapse0x1ff98080() {
   return (neuron0x1ff985c0()*1.95961);
}

double rdNNEE::synapse0x1ff980c0() {
   return (neuron0x1ff98b80()*2.21297);
}

double rdNNEE::synapse0x1ff94fc0() {
   return (neuron0x1ff99140()*3.56681);
}

double rdNNEE::synapse0x1ff95000() {
   return (neuron0x1ff99700()*3.34891);
}

double rdNNEE::synapse0x1ff95040() {
   return (neuron0x1ff99cc0()*-3.23415);
}

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


