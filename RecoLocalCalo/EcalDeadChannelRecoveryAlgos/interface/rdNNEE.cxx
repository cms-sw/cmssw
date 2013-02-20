#include "rdNNEE.h"
#include <cmath>

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

