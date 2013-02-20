#include "rdNNEB.h"
#include <cmath>

double rdNNEB::Value(int index,double in0,double in1,double in2,double in3,double in4,double in5,double in6,double in7) {
   input0 = (in0 - 4.13794)/1.30956;
   input1 = (in1 - 1.15492)/1.68616;
   input2 = (in2 - 1.14956)/1.69316;
   input3 = (in3 - 1.89563)/1.49913;
   input4 = (in4 - 1.90666)/1.49442;
   input5 = (in5 - 0.3065)/1.46726;
   input6 = (in6 - 0.305354)/1.51455;
   input7 = (in7 - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x14ac00d0();
     default:
         return 0.;
   }
}

double rdNNEB::Value(int index, double* input) {
   input0 = (input[0] - 4.13794)/1.30956;
   input1 = (input[1] - 1.15492)/1.68616;
   input2 = (input[2] - 1.14956)/1.69316;
   input3 = (input[3] - 1.89563)/1.49913;
   input4 = (input[4] - 1.90666)/1.49442;
   input5 = (input[5] - 0.3065)/1.46726;
   input6 = (input[6] - 0.305354)/1.51455;
   input7 = (input[7] - 0.31183)/1.46928;
   switch(index) {
     case 0:
         return neuron0x14ac00d0();
     default:
         return 0.;
   }
}

double rdNNEB::neuron0x14abb950() {
   return input0;
}

double rdNNEB::neuron0x14abbc90() {
   return input1;
}

double rdNNEB::neuron0x14abbfd0() {
   return input2;
}

double rdNNEB::neuron0x14abc310() {
   return input3;
}

double rdNNEB::neuron0x14abc650() {
   return input4;
}

double rdNNEB::neuron0x14abc990() {
   return input5;
}

double rdNNEB::neuron0x14abccd0() {
   return input6;
}

double rdNNEB::neuron0x14abd010() {
   return input7;
}

double rdNNEB::input0x14abd480() {
   double input = 2.42798;
   input += synapse0x14a1c350();
   input += synapse0x14ac4550();
   input += synapse0x14abd730();
   input += synapse0x14abd770();
   input += synapse0x14abd7b0();
   input += synapse0x14abd7f0();
   input += synapse0x14abd830();
   input += synapse0x14abd870();
   return input;
}

double rdNNEB::neuron0x14abd480() {
   double input = input0x14abd480();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abd8b0() {
   double input = -0.411164;
   input += synapse0x14abdbf0();
   input += synapse0x14abdc30();
   input += synapse0x14abdc70();
   input += synapse0x14abdcb0();
   input += synapse0x14abdcf0();
   input += synapse0x14abdd30();
   input += synapse0x14abdd70();
   input += synapse0x14abddb0();
   return input;
}

double rdNNEB::neuron0x14abd8b0() {
   double input = input0x14abd8b0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abddf0() {
   double input = 1.80772;
   input += synapse0x14abe130();
   input += synapse0x1431b010();
   input += synapse0x1431b050();
   input += synapse0x14abe280();
   input += synapse0x14abe2c0();
   input += synapse0x14abe300();
   input += synapse0x14abe340();
   input += synapse0x14abe380();
   return input;
}

double rdNNEB::neuron0x14abddf0() {
   double input = input0x14abddf0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abe3c0() {
   double input = 0.642085;
   input += synapse0x14abe700();
   input += synapse0x14abe740();
   input += synapse0x14abe780();
   input += synapse0x14abe7c0();
   input += synapse0x14abe800();
   input += synapse0x14abe840();
   input += synapse0x14abe880();
   input += synapse0x14abe8c0();
   return input;
}

double rdNNEB::neuron0x14abe3c0() {
   double input = input0x14abe3c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abe900() {
   double input = 1.10385;
   input += synapse0x14abec40();
   input += synapse0x14abb880();
   input += synapse0x14ac4590();
   input += synapse0x14319b50();
   input += synapse0x14abe170();
   input += synapse0x14abe1b0();
   input += synapse0x14abe1f0();
   input += synapse0x14abe230();
   return input;
}

double rdNNEB::neuron0x14abe900() {
   double input = input0x14abe900();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abec80() {
   double input = 1.58604;
   input += synapse0x14abefc0();
   input += synapse0x14abf000();
   input += synapse0x14abf040();
   input += synapse0x14abf080();
   input += synapse0x14abf0c0();
   input += synapse0x14abf100();
   input += synapse0x14abf140();
   input += synapse0x14abf180();
   return input;
}

double rdNNEB::neuron0x14abec80() {
   double input = input0x14abec80();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abf1c0() {
   double input = -0.249532;
   input += synapse0x14abf500();
   input += synapse0x14abf540();
   input += synapse0x14abf580();
   input += synapse0x14abf5c0();
   input += synapse0x14abf600();
   input += synapse0x14abf640();
   input += synapse0x14abf680();
   input += synapse0x14abf6c0();
   return input;
}

double rdNNEB::neuron0x14abf1c0() {
   double input = input0x14abf1c0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abf700() {
   double input = 0.976559;
   input += synapse0x14abfa40();
   input += synapse0x14abfa80();
   input += synapse0x14abfac0();
   input += synapse0x14abfb00();
   input += synapse0x14abfb40();
   input += synapse0x14abfb80();
   input += synapse0x14abfbc0();
   input += synapse0x14abfc00();
   return input;
}

double rdNNEB::neuron0x14abf700() {
   double input = input0x14abf700();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14abfc40() {
   double input = -1.43635;
   input += synapse0x14208f70();
   input += synapse0x14208fb0();
   input += synapse0x14334920();
   input += synapse0x14334960();
   input += synapse0x143349a0();
   input += synapse0x143349e0();
   input += synapse0x14334a20();
   input += synapse0x14334a60();
   return input;
}

double rdNNEB::neuron0x14abfc40() {
   double input = input0x14abfc40();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac04a0() {
   double input = -0.127806;
   input += synapse0x14ac0750();
   input += synapse0x14ac0790();
   input += synapse0x14ac07d0();
   input += synapse0x14ac0810();
   input += synapse0x14ac0850();
   input += synapse0x14ac0890();
   input += synapse0x14ac08d0();
   input += synapse0x14ac0910();
   return input;
}

double rdNNEB::neuron0x14ac04a0() {
   double input = input0x14ac04a0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac0950() {
   double input = 0.455212;
   input += synapse0x14ac0c90();
   input += synapse0x14ac0cd0();
   input += synapse0x14ac0d10();
   input += synapse0x14ac0d50();
   input += synapse0x14ac0d90();
   input += synapse0x14ac0dd0();
   input += synapse0x14ac0e10();
   input += synapse0x14ac0e50();
   input += synapse0x14ac0e90();
   input += synapse0x14ac0ed0();
   return input;
}

double rdNNEB::neuron0x14ac0950() {
   double input = input0x14ac0950();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac0f10() {
   double input = -0.695286;
   input += synapse0x14ac1250();
   input += synapse0x14ac1290();
   input += synapse0x14ac12d0();
   input += synapse0x14ac1310();
   input += synapse0x14ac1350();
   input += synapse0x14ac1390();
   input += synapse0x14ac13d0();
   input += synapse0x14ac1410();
   input += synapse0x14ac1450();
   input += synapse0x14ac1490();
   return input;
}

double rdNNEB::neuron0x14ac0f10() {
   double input = input0x14ac0f10();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac14d0() {
   double input = -0.580753;
   input += synapse0x14ac1810();
   input += synapse0x14ac1850();
   input += synapse0x14ac1890();
   input += synapse0x14ac18d0();
   input += synapse0x14ac1910();
   input += synapse0x14ac1950();
   input += synapse0x14ac1990();
   input += synapse0x14ac19d0();
   input += synapse0x14ac1a10();
   input += synapse0x14ac1a50();
   return input;
}

double rdNNEB::neuron0x14ac14d0() {
   double input = input0x14ac14d0();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac1a90() {
   double input = 0.49098;
   input += synapse0x14ac1dd0();
   input += synapse0x14ac1e10();
   input += synapse0x14ac1e50();
   input += synapse0x14ac1e90();
   input += synapse0x14ac1ed0();
   input += synapse0x14ac1f10();
   input += synapse0x14ac1f50();
   input += synapse0x14ac1f90();
   input += synapse0x14ac1fd0();
   input += synapse0x14ac2010();
   return input;
}

double rdNNEB::neuron0x14ac1a90() {
   double input = input0x14ac1a90();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac2050() {
   double input = -0.0961925;
   input += synapse0x14ac2390();
   input += synapse0x14ac23d0();
   input += synapse0x14ac2410();
   input += synapse0x14ac2450();
   input += synapse0x14ac2490();
   input += synapse0x14ac24d0();
   input += synapse0x14ac2510();
   input += synapse0x14ac2550();
   input += synapse0x14ac2590();
   input += synapse0x14ac0090();
   return input;
}

double rdNNEB::neuron0x14ac2050() {
   double input = input0x14ac2050();
   return ((input < -709. ? 0. : (1/(1+exp(-input)))) * 1)+0;
}

double rdNNEB::input0x14ac00d0() {
   double input = 1.63819;
   input += synapse0x14ac0410();
   input += synapse0x14ac0450();
   input += synapse0x14abd350();
   input += synapse0x14abd390();
   input += synapse0x14abd3d0();
   return input;
}

double rdNNEB::neuron0x14ac00d0() {
   double input = input0x14ac00d0();
   return (input * 1)+0;
}

double rdNNEB::synapse0x14a1c350() {
   return (neuron0x14abb950()*-0.216406);
}

double rdNNEB::synapse0x14ac4550() {
   return (neuron0x14abbc90()*-0.420975);
}

double rdNNEB::synapse0x14abd730() {
   return (neuron0x14abbfd0()*-0.56903);
}

double rdNNEB::synapse0x14abd770() {
   return (neuron0x14abc310()*0.170109);
}

double rdNNEB::synapse0x14abd7b0() {
   return (neuron0x14abc650()*1.72569);
}

double rdNNEB::synapse0x14abd7f0() {
   return (neuron0x14abc990()*-0.141621);
}

double rdNNEB::synapse0x14abd830() {
   return (neuron0x14abccd0()*0.258394);
}

double rdNNEB::synapse0x14abd870() {
   return (neuron0x14abd010()*-0.0122614);
}

double rdNNEB::synapse0x14abdbf0() {
   return (neuron0x14abb950()*1.14533);
}

double rdNNEB::synapse0x14abdc30() {
   return (neuron0x14abbc90()*-0.415792);
}

double rdNNEB::synapse0x14abdc70() {
   return (neuron0x14abbfd0()*-0.661272);
}

double rdNNEB::synapse0x14abdcb0() {
   return (neuron0x14abc310()*0.0745773);
}

double rdNNEB::synapse0x14abdcf0() {
   return (neuron0x14abc650()*-0.0324746);
}

double rdNNEB::synapse0x14abdd30() {
   return (neuron0x14abc990()*0.0664283);
}

double rdNNEB::synapse0x14abdd70() {
   return (neuron0x14abccd0()*-1.67815);
}

double rdNNEB::synapse0x14abddb0() {
   return (neuron0x14abd010()*0.343732);
}

double rdNNEB::synapse0x14abe130() {
   return (neuron0x14abb950()*1.70759);
}

double rdNNEB::synapse0x1431b010() {
   return (neuron0x14abbc90()*-0.829353);
}

double rdNNEB::synapse0x1431b050() {
   return (neuron0x14abbfd0()*0.728579);
}

double rdNNEB::synapse0x14abe280() {
   return (neuron0x14abc310()*0.608541);
}

double rdNNEB::synapse0x14abe2c0() {
   return (neuron0x14abc650()*-0.991141);
}

double rdNNEB::synapse0x14abe300() {
   return (neuron0x14abc990()*-0.397236);
}

double rdNNEB::synapse0x14abe340() {
   return (neuron0x14abccd0()*-0.505365);
}

double rdNNEB::synapse0x14abe380() {
   return (neuron0x14abd010()*0.0697281);
}

double rdNNEB::synapse0x14abe700() {
   return (neuron0x14abb950()*-0.606239);
}

double rdNNEB::synapse0x14abe740() {
   return (neuron0x14abbc90()*1.85788);
}

double rdNNEB::synapse0x14abe780() {
   return (neuron0x14abbfd0()*-0.0541781);
}

double rdNNEB::synapse0x14abe7c0() {
   return (neuron0x14abc310()*-0.156056);
}

double rdNNEB::synapse0x14abe800() {
   return (neuron0x14abc650()*-0.511695);
}

double rdNNEB::synapse0x14abe840() {
   return (neuron0x14abc990()*-0.0604903);
}

double rdNNEB::synapse0x14abe880() {
   return (neuron0x14abccd0()*0.296835);
}

double rdNNEB::synapse0x14abe8c0() {
   return (neuron0x14abd010()*0.151318);
}

double rdNNEB::synapse0x14abec40() {
   return (neuron0x14abb950()*0.45892);
}

double rdNNEB::synapse0x14abb880() {
   return (neuron0x14abbc90()*-0.113319);
}

double rdNNEB::synapse0x14ac4590() {
   return (neuron0x14abbfd0()*-0.0432273);
}

double rdNNEB::synapse0x14319b50() {
   return (neuron0x14abc310()*-1.20745);
}

double rdNNEB::synapse0x14abe170() {
   return (neuron0x14abc650()*-0.414526);
}

double rdNNEB::synapse0x14abe1b0() {
   return (neuron0x14abc990()*0.388562);
}

double rdNNEB::synapse0x14abe1f0() {
   return (neuron0x14abccd0()*0.581199);
}

double rdNNEB::synapse0x14abe230() {
   return (neuron0x14abd010()*-0.242221);
}

double rdNNEB::synapse0x14abefc0() {
   return (neuron0x14abb950()*-1.53883);
}

double rdNNEB::synapse0x14abf000() {
   return (neuron0x14abbc90()*-0.0712372);
}

double rdNNEB::synapse0x14abf040() {
   return (neuron0x14abbfd0()*-0.00473586);
}

double rdNNEB::synapse0x14abf080() {
   return (neuron0x14abc310()*0.202715);
}

double rdNNEB::synapse0x14abf0c0() {
   return (neuron0x14abc650()*0.403952);
}

double rdNNEB::synapse0x14abf100() {
   return (neuron0x14abc990()*0.306872);
}

double rdNNEB::synapse0x14abf140() {
   return (neuron0x14abccd0()*0.256241);
}

double rdNNEB::synapse0x14abf180() {
   return (neuron0x14abd010()*0.203546);
}

double rdNNEB::synapse0x14abf500() {
   return (neuron0x14abb950()*0.749902);
}

double rdNNEB::synapse0x14abf540() {
   return (neuron0x14abbc90()*0.186212);
}

double rdNNEB::synapse0x14abf580() {
   return (neuron0x14abbfd0()*0.641923);
}

double rdNNEB::synapse0x14abf5c0() {
   return (neuron0x14abc310()*0.0616216);
}

double rdNNEB::synapse0x14abf600() {
   return (neuron0x14abc650()*0.0547814);
}

double rdNNEB::synapse0x14abf640() {
   return (neuron0x14abc990()*0.150325);
}

double rdNNEB::synapse0x14abf680() {
   return (neuron0x14abccd0()*-0.037531);
}

double rdNNEB::synapse0x14abf6c0() {
   return (neuron0x14abd010()*0.00331658);
}

double rdNNEB::synapse0x14abfa40() {
   return (neuron0x14abb950()*-0.0260585);
}

double rdNNEB::synapse0x14abfa80() {
   return (neuron0x14abbc90()*0.243857);
}

double rdNNEB::synapse0x14abfac0() {
   return (neuron0x14abbfd0()*0.556623);
}

double rdNNEB::synapse0x14abfb00() {
   return (neuron0x14abc310()*0.786342);
}

double rdNNEB::synapse0x14abfb40() {
   return (neuron0x14abc650()*-1.22138);
}

double rdNNEB::synapse0x14abfb80() {
   return (neuron0x14abc990()*-0.180283);
}

double rdNNEB::synapse0x14abfbc0() {
   return (neuron0x14abccd0()*-0.557386);
}

double rdNNEB::synapse0x14abfc00() {
   return (neuron0x14abd010()*0.194623);
}

double rdNNEB::synapse0x14208f70() {
   return (neuron0x14abb950()*-0.560909);
}

double rdNNEB::synapse0x14208fb0() {
   return (neuron0x14abbc90()*0.618483);
}

double rdNNEB::synapse0x14334920() {
   return (neuron0x14abbfd0()*1.56332);
}

double rdNNEB::synapse0x14334960() {
   return (neuron0x14abc310()*-0.543328);
}

double rdNNEB::synapse0x143349a0() {
   return (neuron0x14abc650()*-0.183581);
}

double rdNNEB::synapse0x143349e0() {
   return (neuron0x14abc990()*0.7814);
}

double rdNNEB::synapse0x14334a20() {
   return (neuron0x14abccd0()*-0.104609);
}

double rdNNEB::synapse0x14334a60() {
   return (neuron0x14abd010()*-0.376717);
}

double rdNNEB::synapse0x14ac0750() {
   return (neuron0x14abb950()*-1.3224);
}

double rdNNEB::synapse0x14ac0790() {
   return (neuron0x14abbc90()*0.05359);
}

double rdNNEB::synapse0x14ac07d0() {
   return (neuron0x14abbfd0()*0.682048);
}

double rdNNEB::synapse0x14ac0810() {
   return (neuron0x14abc310()*-0.249621);
}

double rdNNEB::synapse0x14ac0850() {
   return (neuron0x14abc650()*-0.0417721);
}

double rdNNEB::synapse0x14ac0890() {
   return (neuron0x14abc990()*0.099426);
}

double rdNNEB::synapse0x14ac08d0() {
   return (neuron0x14abccd0()*0.417192);
}

double rdNNEB::synapse0x14ac0910() {
   return (neuron0x14abd010()*0.23391);
}

double rdNNEB::synapse0x14ac0c90() {
   return (neuron0x14abd480()*0.0948073);
}

double rdNNEB::synapse0x14ac0cd0() {
   return (neuron0x14abd8b0()*-1.02893);
}

double rdNNEB::synapse0x14ac0d10() {
   return (neuron0x14abddf0()*-0.509918);
}

double rdNNEB::synapse0x14ac0d50() {
   return (neuron0x14abe3c0()*0.867766);
}

double rdNNEB::synapse0x14ac0d90() {
   return (neuron0x14abe900()*0.664667);
}

double rdNNEB::synapse0x14ac0dd0() {
   return (neuron0x14abec80()*0.659037);
}

double rdNNEB::synapse0x14ac0e10() {
   return (neuron0x14abf1c0()*-1.63301);
}

double rdNNEB::synapse0x14ac0e50() {
   return (neuron0x14abf700()*0.626759);
}

double rdNNEB::synapse0x14ac0e90() {
   return (neuron0x14abfc40()*-0.251753);
}

double rdNNEB::synapse0x14ac0ed0() {
   return (neuron0x14ac04a0()*1.72268);
}

double rdNNEB::synapse0x14ac1250() {
   return (neuron0x14abd480()*0.86684);
}

double rdNNEB::synapse0x14ac1290() {
   return (neuron0x14abd8b0()*-0.228328);
}

double rdNNEB::synapse0x14ac12d0() {
   return (neuron0x14abddf0()*-2.27991);
}

double rdNNEB::synapse0x14ac1310() {
   return (neuron0x14abe3c0()*2.19628);
}

double rdNNEB::synapse0x14ac1350() {
   return (neuron0x14abe900()*-1.12465);
}

double rdNNEB::synapse0x14ac1390() {
   return (neuron0x14abec80()*-2.69582);
}

double rdNNEB::synapse0x14ac13d0() {
   return (neuron0x14abf1c0()*0.388994);
}

double rdNNEB::synapse0x14ac1410() {
   return (neuron0x14abf700()*-1.59652);
}

double rdNNEB::synapse0x14ac1450() {
   return (neuron0x14abfc40()*1.86835);
}

double rdNNEB::synapse0x14ac1490() {
   return (neuron0x14ac04a0()*1.49056);
}

double rdNNEB::synapse0x14ac1810() {
   return (neuron0x14abd480()*-0.290069);
}

double rdNNEB::synapse0x14ac1850() {
   return (neuron0x14abd8b0()*-0.803451);
}

double rdNNEB::synapse0x14ac1890() {
   return (neuron0x14abddf0()*0.00307815);
}

double rdNNEB::synapse0x14ac18d0() {
   return (neuron0x14abe3c0()*-0.777897);
}

double rdNNEB::synapse0x14ac1910() {
   return (neuron0x14abe900()*-0.291359);
}

double rdNNEB::synapse0x14ac1950() {
   return (neuron0x14abec80()*0.501869);
}

double rdNNEB::synapse0x14ac1990() {
   return (neuron0x14abf1c0()*0.991883);
}

double rdNNEB::synapse0x14ac19d0() {
   return (neuron0x14abf700()*-0.601744);
}

double rdNNEB::synapse0x14ac1a10() {
   return (neuron0x14abfc40()*-0.705386);
}

double rdNNEB::synapse0x14ac1a50() {
   return (neuron0x14ac04a0()*-1.02517);
}

double rdNNEB::synapse0x14ac1dd0() {
   return (neuron0x14abd480()*1.05875);
}

double rdNNEB::synapse0x14ac1e10() {
   return (neuron0x14abd8b0()*0.0280678);
}

double rdNNEB::synapse0x14ac1e50() {
   return (neuron0x14abddf0()*0.310589);
}

double rdNNEB::synapse0x14ac1e90() {
   return (neuron0x14abe3c0()*0.950605);
}

double rdNNEB::synapse0x14ac1ed0() {
   return (neuron0x14abe900()*-0.141197);
}

double rdNNEB::synapse0x14ac1f10() {
   return (neuron0x14abec80()*-0.781974);
}

double rdNNEB::synapse0x14ac1f50() {
   return (neuron0x14abf1c0()*1.51498);
}

double rdNNEB::synapse0x14ac1f90() {
   return (neuron0x14abf700()*-0.23685);
}

double rdNNEB::synapse0x14ac1fd0() {
   return (neuron0x14abfc40()*0.881958);
}

double rdNNEB::synapse0x14ac2010() {
   return (neuron0x14ac04a0()*-0.707959);
}

double rdNNEB::synapse0x14ac2390() {
   return (neuron0x14abd480()*-1.16238);
}

double rdNNEB::synapse0x14ac23d0() {
   return (neuron0x14abd8b0()*-0.98238);
}

double rdNNEB::synapse0x14ac2410() {
   return (neuron0x14abddf0()*1.1459);
}

double rdNNEB::synapse0x14ac2450() {
   return (neuron0x14abe3c0()*-1.25397);
}

double rdNNEB::synapse0x14ac2490() {
   return (neuron0x14abe900()*1.4953);
}

double rdNNEB::synapse0x14ac24d0() {
   return (neuron0x14abec80()*1.63361);
}

double rdNNEB::synapse0x14ac2510() {
   return (neuron0x14abf1c0()*-0.665746);
}

double rdNNEB::synapse0x14ac2550() {
   return (neuron0x14abf700()*0.569317);
}

double rdNNEB::synapse0x14ac2590() {
   return (neuron0x14abfc40()*-0.375751);
}

double rdNNEB::synapse0x14ac0090() {
   return (neuron0x14ac04a0()*-0.472168);
}

double rdNNEB::synapse0x14ac0410() {
   return (neuron0x14ac0950()*-2.71126);
}

double rdNNEB::synapse0x14ac0450() {
   return (neuron0x14ac0f10()*5.31703);
}

double rdNNEB::synapse0x14abd350() {
   return (neuron0x14ac14d0()*0.943595);
}

double rdNNEB::synapse0x14abd390() {
   return (neuron0x14ac1a90()*4.36333);
}

double rdNNEB::synapse0x14abd3d0() {
   return (neuron0x14ac2050()*-4.62187);
}

