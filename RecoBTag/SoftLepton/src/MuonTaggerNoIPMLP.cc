#include "RecoBTag/SoftLepton/interface/MuonTaggerNoIPMLP.h"
#include <cmath>

// no normalization
const double prior0 = 1.0;      // b->l
const double prior1 = 1.0;      // b->c->l
const double prior2 = 1.0;      // c->l
const double prior3 = 1.0;      // x->l

// normalization to tt + qcd from lepton info
//const double prior0 = 1.757;    // b->l
//const double prior1 = 1.766;    // b->c->l
//const double prior2 = 0.548;    // c->l
//const double prior3 = 0.701;    // x->l

double MuonTaggerNoIPMLP::value(int index,double in0,double in1,double in2,double in3,double in4) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   input4 = (in4 - 0)/1;

   double out0 = neuron0xb510310() * prior0;
   double out1 = neuron0xb510650() * prior1;
   double out2 = neuron0xb510a58() * prior2;
   double out3 = neuron0xb510f10() * prior3;
   double normalization = out0 + out1 + out2 + out3;

   switch(index) {
     case 0:
         return (out0 / normalization);
     case 1:
         return (out1 / normalization);
     case 2:
         return (out2 / normalization);
     case 3:
         return (out3 / normalization);
     default:
         return 0.;
   }
}

double MuonTaggerNoIPMLP::neuron0xb50b898() {
   return input0;
}

double MuonTaggerNoIPMLP::neuron0xb50ba48() {
   return input1;
}

double MuonTaggerNoIPMLP::neuron0xb50bc20() {
   return input2;
}

double MuonTaggerNoIPMLP::neuron0xb50ce60() {
   return input3;
}

double MuonTaggerNoIPMLP::neuron0xb50d038() {
   return input4;
}

double MuonTaggerNoIPMLP::input0xb50d328() {
   double input = -0.404291;
   input += synapse0xb50bdb0();
   input += synapse0xb027700();
   input += synapse0xb027930();
   input += synapse0xb50d500();
   input += synapse0xb50d528();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50d328() {
   double input = input0xb50d328();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50d550() {
   double input = 0.381854;
   input += synapse0xb50d748();
   input += synapse0xb50d770();
   input += synapse0xb50d798();
   input += synapse0xb50d7c0();
   input += synapse0xb50d7e8();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50d550() {
   double input = input0xb50d550();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50d810() {
   double input = -5.12361;
   input += synapse0xb50da08();
   input += synapse0xb50da30();
   input += synapse0xb50da58();
   input += synapse0xb50da80();
   input += synapse0xb50daa8();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50d810() {
   double input = input0xb50d810();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50dad0() {
   double input = -0.438227;
   input += synapse0xb50dcc8();
   input += synapse0xb50dcf0();
   input += synapse0xb50dda0();
   input += synapse0xb50ddc8();
   input += synapse0xb50ddf0();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50dad0() {
   double input = input0xb50dad0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50de18() {
   double input = -1.86778;
   input += synapse0xb50dfc8();
   input += synapse0xb50dff0();
   input += synapse0xb50e018();
   input += synapse0xb50e040();
   input += synapse0xb50e068();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50de18() {
   double input = input0xb50de18();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50e090() {
   double input = -0.490283;
   input += synapse0xb50e288();
   input += synapse0xb50e2b0();
   input += synapse0xb50e2d8();
   input += synapse0xb50e300();
   input += synapse0xb50e328();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50e090() {
   double input = input0xb50e090();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50e350() {
   double input = 0.551554;
   input += synapse0xb50e548();
   input += synapse0xb50e570();
   input += synapse0xb50e598();
   input += synapse0xb50dd18();
   input += synapse0xb50dd40();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50e350() {
   double input = input0xb50e350();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50e6c8() {
   double input = -0.0553462;
   input += synapse0xb50e8a0();
   input += synapse0xb50e8c8();
   input += synapse0xb50e8f0();
   input += synapse0xb50e918();
   input += synapse0xb50e940();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50e6c8() {
   double input = input0xb50e6c8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50e968() {
   double input = -2.15563;
   input += synapse0xb50eb60();
   input += synapse0xb50eb88();
   input += synapse0xb50ebb0();
   input += synapse0xb50ebd8();
   input += synapse0xb50ec00();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50e968() {
   double input = input0xb50e968();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50ec28() {
   double input = -1.66201;
   input += synapse0xb50ee20();
   input += synapse0xb50ee48();
   input += synapse0xb50ee70();
   input += synapse0xb50ee98();
   input += synapse0xb50eec0();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50ec28() {
   double input = input0xb50ec28();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50eee8() {
   double input = 4.32255;
   input += synapse0xb50f0e0();
   input += synapse0xb50f108();
   input += synapse0xb50f130();
   input += synapse0xb50f158();
   input += synapse0xb50f180();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50eee8() {
   double input = input0xb50eee8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50f1a8() {
   double input = -0.610279;
   input += synapse0xb50f428();
   input += synapse0xb50f450();
   input += synapse0xb50f478();
   input += synapse0xb50f4a0();
   input += synapse0xb50f4c8();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50f1a8() {
   double input = input0xb50f1a8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50f4f0() {
   double input = 0.0956179;
   input += synapse0xb50f6a0();
   input += synapse0xb50f6c8();
   input += synapse0xb50f6f0();
   input += synapse0xb50f718();
   input += synapse0xb50f740();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50f4f0() {
   double input = input0xb50f4f0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50e5c0() {
   double input = -0.430126;
   input += synapse0xb50b7e8();
   input += synapse0xb50fa48();
   input += synapse0xb50fa70();
   input += synapse0xb50fa98();
   input += synapse0xb50fac0();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50e5c0() {
   double input = input0xb50e5c0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50fae8() {
   double input = -5.52182;
   input += synapse0xb50fce0();
   input += synapse0xb50fd08();
   input += synapse0xb50fd30();
   input += synapse0xb50fd58();
   input += synapse0xb50fd80();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50fae8() {
   double input = input0xb50fae8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb50fda8() {
   double input = -1.41181;
   input += synapse0xb50ffa0();
   input += synapse0xb510050();
   input += synapse0xb510100();
   input += synapse0xb5101b0();
   input += synapse0xb510260();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb50fda8() {
   double input = input0xb50fda8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerNoIPMLP::input0xb510310() {
   double input = 0.333426;
   input += synapse0xb50b850();
   input += synapse0xb50d280();
   input += synapse0xb50d2a8();
   input += synapse0xb50d2d0();
   input += synapse0xb50d2f8();
   input += synapse0xb510410();
   input += synapse0xb510438();
   input += synapse0xb510460();
   input += synapse0xb510488();
   input += synapse0xb5104b0();
   input += synapse0xb5104d8();
   input += synapse0xb510500();
   input += synapse0xb510528();
   input += synapse0xb510550();
   input += synapse0xb510578();
   input += synapse0xb5105a0();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb510310() {
   double input = input0xb510310();
   return (exp(input) / (exp(input0xb510310()) + exp(input0xb510650()) + exp(input0xb510a58()) + exp(input0xb510f10())) * 1)+0;
}

double MuonTaggerNoIPMLP::input0xb510650() {
   double input = -0.585024;
   input += synapse0xb510750();
   input += synapse0xb510778();
   input += synapse0xb5107a0();
   input += synapse0xb5107c8();
   input += synapse0xb5107f0();
   input += synapse0xb510818();
   input += synapse0xb510840();
   input += synapse0xb510868();
   input += synapse0xb510890();
   input += synapse0xb5108b8();
   input += synapse0xb5108e0();
   input += synapse0xb510908();
   input += synapse0xb510930();
   input += synapse0xb510958();
   input += synapse0xb510980();
   input += synapse0xb5109a8();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb510650() {
   double input = input0xb510650();
   return (exp(input) / (exp(input0xb510310()) + exp(input0xb510650()) + exp(input0xb510a58()) + exp(input0xb510f10())) * 1)+0;
}

double MuonTaggerNoIPMLP::input0xb510a58() {
   double input = -0.223686;
   input += synapse0xb510c08();
   input += synapse0xb510c30();
   input += synapse0xb510c58();
   input += synapse0xb510c80();
   input += synapse0xb510ca8();
   input += synapse0xb510cd0();
   input += synapse0xb510cf8();
   input += synapse0xb510d20();
   input += synapse0xb510d48();
   input += synapse0xb510d70();
   input += synapse0xb510d98();
   input += synapse0xb510dc0();
   input += synapse0xb510de8();
   input += synapse0xb510e10();
   input += synapse0xb510e38();
   input += synapse0xb510e60();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb510a58() {
   double input = input0xb510a58();
   return (exp(input) / (exp(input0xb510310()) + exp(input0xb510650()) + exp(input0xb510a58()) + exp(input0xb510f10())) * 1)+0;
}

double MuonTaggerNoIPMLP::input0xb510f10() {
   double input = -0.0410047;
   input += synapse0xb5110c0();
   input += synapse0xb50f768();
   input += synapse0xb50f790();
   input += synapse0xb50f7b8();
   input += synapse0xb50f7e0();
   input += synapse0xb50f808();
   input += synapse0xb50f830();
   input += synapse0xb50f858();
   input += synapse0xb50f880();
   input += synapse0xb50f8a8();
   input += synapse0xb50f8d0();
   input += synapse0xb50f8f8();
   input += synapse0xb50f920();
   input += synapse0xb50f948();
   input += synapse0xb5114f0();
   input += synapse0xb511518();
   return input;
}

double MuonTaggerNoIPMLP::neuron0xb510f10() {
   double input = input0xb510f10();
   return (exp(input) / (exp(input0xb510310()) + exp(input0xb510650()) + exp(input0xb510a58()) + exp(input0xb510f10())) * 1)+0;
}

double MuonTaggerNoIPMLP::synapse0xb50bdb0() {
   return (neuron0xb50b898()*0.0266127);
}

double MuonTaggerNoIPMLP::synapse0xb027700() {
   return (neuron0xb50ba48()*-0.787503);
}

double MuonTaggerNoIPMLP::synapse0xb027930() {
   return (neuron0xb50bc20()*0.3814);
}

double MuonTaggerNoIPMLP::synapse0xb50d500() {
   return (neuron0xb50ce60()*0.0202667);
}

double MuonTaggerNoIPMLP::synapse0xb50d528() {
   return (neuron0xb50d038()*-0.121732);
}

double MuonTaggerNoIPMLP::synapse0xb50d748() {
   return (neuron0xb50b898()*-1.38199);
}

double MuonTaggerNoIPMLP::synapse0xb50d770() {
   return (neuron0xb50ba48()*1.00737);
}

double MuonTaggerNoIPMLP::synapse0xb50d798() {
   return (neuron0xb50bc20()*-0.819069);
}

double MuonTaggerNoIPMLP::synapse0xb50d7c0() {
   return (neuron0xb50ce60()*0.877898);
}

double MuonTaggerNoIPMLP::synapse0xb50d7e8() {
   return (neuron0xb50d038()*-0.0207687);
}

double MuonTaggerNoIPMLP::synapse0xb50da08() {
   return (neuron0xb50b898()*4.55447);
}

double MuonTaggerNoIPMLP::synapse0xb50da30() {
   return (neuron0xb50ba48()*4.49461);
}

double MuonTaggerNoIPMLP::synapse0xb50da58() {
   return (neuron0xb50bc20()*-3.83426);
}

double MuonTaggerNoIPMLP::synapse0xb50da80() {
   return (neuron0xb50ce60()*0.00970572);
}

double MuonTaggerNoIPMLP::synapse0xb50daa8() {
   return (neuron0xb50d038()*0.0988528);
}

double MuonTaggerNoIPMLP::synapse0xb50dcc8() {
   return (neuron0xb50b898()*-1.37133);
}

double MuonTaggerNoIPMLP::synapse0xb50dcf0() {
   return (neuron0xb50ba48()*-0.887154);
}

double MuonTaggerNoIPMLP::synapse0xb50dda0() {
   return (neuron0xb50bc20()*1.03883);
}

double MuonTaggerNoIPMLP::synapse0xb50ddc8() {
   return (neuron0xb50ce60()*0.000187213);
}

double MuonTaggerNoIPMLP::synapse0xb50ddf0() {
   return (neuron0xb50d038()*-0.0336801);
}

double MuonTaggerNoIPMLP::synapse0xb50dfc8() {
   return (neuron0xb50b898()*0.0901434);
}

double MuonTaggerNoIPMLP::synapse0xb50dff0() {
   return (neuron0xb50ba48()*1.22334);
}

double MuonTaggerNoIPMLP::synapse0xb50e018() {
   return (neuron0xb50bc20()*0.121724);
}

double MuonTaggerNoIPMLP::synapse0xb50e040() {
   return (neuron0xb50ce60()*-0.0974919);
}

double MuonTaggerNoIPMLP::synapse0xb50e068() {
   return (neuron0xb50d038()*0.426064);
}

double MuonTaggerNoIPMLP::synapse0xb50e288() {
   return (neuron0xb50b898()*0.240952);
}

double MuonTaggerNoIPMLP::synapse0xb50e2b0() {
   return (neuron0xb50ba48()*-0.008872);
}

double MuonTaggerNoIPMLP::synapse0xb50e2d8() {
   return (neuron0xb50bc20()*0.488225);
}

double MuonTaggerNoIPMLP::synapse0xb50e300() {
   return (neuron0xb50ce60()*-5.32553);
}

double MuonTaggerNoIPMLP::synapse0xb50e328() {
   return (neuron0xb50d038()*-0.0941807);
}

double MuonTaggerNoIPMLP::synapse0xb50e548() {
   return (neuron0xb50b898()*0.0831391);
}

double MuonTaggerNoIPMLP::synapse0xb50e570() {
   return (neuron0xb50ba48()*0.267279);
}

double MuonTaggerNoIPMLP::synapse0xb50e598() {
   return (neuron0xb50bc20()*0.499692);
}

double MuonTaggerNoIPMLP::synapse0xb50dd18() {
   return (neuron0xb50ce60()*0.795639);
}

double MuonTaggerNoIPMLP::synapse0xb50dd40() {
   return (neuron0xb50d038()*0.155882);
}

double MuonTaggerNoIPMLP::synapse0xb50e8a0() {
   return (neuron0xb50b898()*0.164697);
}

double MuonTaggerNoIPMLP::synapse0xb50e8c8() {
   return (neuron0xb50ba48()*-0.431761);
}

double MuonTaggerNoIPMLP::synapse0xb50e8f0() {
   return (neuron0xb50bc20()*-0.223045);
}

double MuonTaggerNoIPMLP::synapse0xb50e918() {
   return (neuron0xb50ce60()*-0.60887);
}

double MuonTaggerNoIPMLP::synapse0xb50e940() {
   return (neuron0xb50d038()*0.259537);
}

double MuonTaggerNoIPMLP::synapse0xb50eb60() {
   return (neuron0xb50b898()*3.41847);
}

double MuonTaggerNoIPMLP::synapse0xb50eb88() {
   return (neuron0xb50ba48()*-0.476877);
}

double MuonTaggerNoIPMLP::synapse0xb50ebb0() {
   return (neuron0xb50bc20()*-1.02075);
}

double MuonTaggerNoIPMLP::synapse0xb50ebd8() {
   return (neuron0xb50ce60()*0.00209324);
}

double MuonTaggerNoIPMLP::synapse0xb50ec00() {
   return (neuron0xb50d038()*0.0273101);
}

double MuonTaggerNoIPMLP::synapse0xb50ee20() {
   return (neuron0xb50b898()*2.78842);
}

double MuonTaggerNoIPMLP::synapse0xb50ee48() {
   return (neuron0xb50ba48()*-2.83771);
}

double MuonTaggerNoIPMLP::synapse0xb50ee70() {
   return (neuron0xb50bc20()*-8.65371);
}

double MuonTaggerNoIPMLP::synapse0xb50ee98() {
   return (neuron0xb50ce60()*0.00647572);
}

double MuonTaggerNoIPMLP::synapse0xb50eec0() {
   return (neuron0xb50d038()*0.126806);
}

double MuonTaggerNoIPMLP::synapse0xb50f0e0() {
   return (neuron0xb50b898()*-3.96164);
}

double MuonTaggerNoIPMLP::synapse0xb50f108() {
   return (neuron0xb50ba48()*-3.76339);
}

double MuonTaggerNoIPMLP::synapse0xb50f130() {
   return (neuron0xb50bc20()*-4.15894);
}

double MuonTaggerNoIPMLP::synapse0xb50f158() {
   return (neuron0xb50ce60()*-0.00764395);
}

double MuonTaggerNoIPMLP::synapse0xb50f180() {
   return (neuron0xb50d038()*-0.138208);
}

double MuonTaggerNoIPMLP::synapse0xb50f428() {
   return (neuron0xb50b898()*1.30645);
}

double MuonTaggerNoIPMLP::synapse0xb50f450() {
   return (neuron0xb50ba48()*-1.0135);
}

double MuonTaggerNoIPMLP::synapse0xb50f478() {
   return (neuron0xb50bc20()*0.674552);
}

double MuonTaggerNoIPMLP::synapse0xb50f4a0() {
   return (neuron0xb50ce60()*-0.797396);
}

double MuonTaggerNoIPMLP::synapse0xb50f4c8() {
   return (neuron0xb50d038()*0.336093);
}

double MuonTaggerNoIPMLP::synapse0xb50f6a0() {
   return (neuron0xb50b898()*0.288096);
}

double MuonTaggerNoIPMLP::synapse0xb50f6c8() {
   return (neuron0xb50ba48()*0.503032);
}

double MuonTaggerNoIPMLP::synapse0xb50f6f0() {
   return (neuron0xb50bc20()*-0.473197);
}

double MuonTaggerNoIPMLP::synapse0xb50f718() {
   return (neuron0xb50ce60()*0.739854);
}

double MuonTaggerNoIPMLP::synapse0xb50f740() {
   return (neuron0xb50d038()*0.470193);
}

double MuonTaggerNoIPMLP::synapse0xb50b7e8() {
   return (neuron0xb50b898()*-0.134717);
}

double MuonTaggerNoIPMLP::synapse0xb50fa48() {
   return (neuron0xb50ba48()*-0.471165);
}

double MuonTaggerNoIPMLP::synapse0xb50fa70() {
   return (neuron0xb50bc20()*0.427153);
}

double MuonTaggerNoIPMLP::synapse0xb50fa98() {
   return (neuron0xb50ce60()*-0.54334);
}

double MuonTaggerNoIPMLP::synapse0xb50fac0() {
   return (neuron0xb50d038()*0.462077);
}

double MuonTaggerNoIPMLP::synapse0xb50fce0() {
   return (neuron0xb50b898()*1.03136);
}

double MuonTaggerNoIPMLP::synapse0xb50fd08() {
   return (neuron0xb50ba48()*-1.06388);
}

double MuonTaggerNoIPMLP::synapse0xb50fd30() {
   return (neuron0xb50bc20()*1.72108);
}

double MuonTaggerNoIPMLP::synapse0xb50fd58() {
   return (neuron0xb50ce60()*-0.214569);
}

double MuonTaggerNoIPMLP::synapse0xb50fd80() {
   return (neuron0xb50d038()*8.26042);
}

double MuonTaggerNoIPMLP::synapse0xb50ffa0() {
   return (neuron0xb50b898()*-3.84874);
}

double MuonTaggerNoIPMLP::synapse0xb510050() {
   return (neuron0xb50ba48()*1.25653);
}

double MuonTaggerNoIPMLP::synapse0xb510100() {
   return (neuron0xb50bc20()*1.27311);
}

double MuonTaggerNoIPMLP::synapse0xb5101b0() {
   return (neuron0xb50ce60()*-0.00565679);
}

double MuonTaggerNoIPMLP::synapse0xb510260() {
   return (neuron0xb50d038()*-0.076792);
}

double MuonTaggerNoIPMLP::synapse0xb50b850() {
   return (neuron0xb50d328()*-0.742609);
}

double MuonTaggerNoIPMLP::synapse0xb50d280() {
   return (neuron0xb50d550()*-1.54936);
}

double MuonTaggerNoIPMLP::synapse0xb50d2a8() {
   return (neuron0xb50d810()*-0.616462);
}

double MuonTaggerNoIPMLP::synapse0xb50d2d0() {
   return (neuron0xb50dad0()*-2.50878);
}

double MuonTaggerNoIPMLP::synapse0xb50d2f8() {
   return (neuron0xb50de18()*0.901498);
}

double MuonTaggerNoIPMLP::synapse0xb510410() {
   return (neuron0xb50e090()*0.124071);
}

double MuonTaggerNoIPMLP::synapse0xb510438() {
   return (neuron0xb50e350()*0.268812);
}

double MuonTaggerNoIPMLP::synapse0xb510460() {
   return (neuron0xb50e6c8()*0.268676);
}

double MuonTaggerNoIPMLP::synapse0xb510488() {
   return (neuron0xb50e968()*2.53128);
}

double MuonTaggerNoIPMLP::synapse0xb5104b0() {
   return (neuron0xb50ec28()*-0.915796);
}

double MuonTaggerNoIPMLP::synapse0xb5104d8() {
   return (neuron0xb50eee8()*1.66172);
}

double MuonTaggerNoIPMLP::synapse0xb510500() {
   return (neuron0xb50f1a8()*1.42875);
}

double MuonTaggerNoIPMLP::synapse0xb510528() {
   return (neuron0xb50f4f0()*0.363917);
}

double MuonTaggerNoIPMLP::synapse0xb510550() {
   return (neuron0xb50e5c0()*-0.205041);
}

double MuonTaggerNoIPMLP::synapse0xb510578() {
   return (neuron0xb50fae8()*-0.286926);
}

double MuonTaggerNoIPMLP::synapse0xb5105a0() {
   return (neuron0xb50fda8()*1.52037);
}

double MuonTaggerNoIPMLP::synapse0xb510750() {
   return (neuron0xb50d328()*0.567602);
}

double MuonTaggerNoIPMLP::synapse0xb510778() {
   return (neuron0xb50d550()*0.10175);
}

double MuonTaggerNoIPMLP::synapse0xb5107a0() {
   return (neuron0xb50d810()*2.37359);
}

double MuonTaggerNoIPMLP::synapse0xb5107c8() {
   return (neuron0xb50dad0()*1.27293);
}

double MuonTaggerNoIPMLP::synapse0xb5107f0() {
   return (neuron0xb50de18()*-0.060524);
}

double MuonTaggerNoIPMLP::synapse0xb510818() {
   return (neuron0xb50e090()*-0.181962);
}

double MuonTaggerNoIPMLP::synapse0xb510840() {
   return (neuron0xb50e350()*-0.878628);
}

double MuonTaggerNoIPMLP::synapse0xb510868() {
   return (neuron0xb50e6c8()*0.268003);
}

double MuonTaggerNoIPMLP::synapse0xb510890() {
   return (neuron0xb50e968()*-1.3776);
}

double MuonTaggerNoIPMLP::synapse0xb5108b8() {
   return (neuron0xb50ec28()*-0.619587);
}

double MuonTaggerNoIPMLP::synapse0xb5108e0() {
   return (neuron0xb50eee8()*1.55994);
}

double MuonTaggerNoIPMLP::synapse0xb510908() {
   return (neuron0xb50f1a8()*0.307386);
}

double MuonTaggerNoIPMLP::synapse0xb510930() {
   return (neuron0xb50f4f0()*-0.746327);
}

double MuonTaggerNoIPMLP::synapse0xb510958() {
   return (neuron0xb50e5c0()*0.190604);
}

double MuonTaggerNoIPMLP::synapse0xb510980() {
   return (neuron0xb50fae8()*-3.27889);
}

double MuonTaggerNoIPMLP::synapse0xb5109a8() {
   return (neuron0xb50fda8()*-1.49498);
}

double MuonTaggerNoIPMLP::synapse0xb510c08() {
   return (neuron0xb50d328()*-0.278047);
}

double MuonTaggerNoIPMLP::synapse0xb510c30() {
   return (neuron0xb50d550()*0.465304);
}

double MuonTaggerNoIPMLP::synapse0xb510c58() {
   return (neuron0xb50d810()*1.5647);
}

double MuonTaggerNoIPMLP::synapse0xb510c80() {
   return (neuron0xb50dad0()*-5.23866);
}

double MuonTaggerNoIPMLP::synapse0xb510ca8() {
   return (neuron0xb50de18()*-1.06729);
}

double MuonTaggerNoIPMLP::synapse0xb510cd0() {
   return (neuron0xb50e090()*0.39036);
}

double MuonTaggerNoIPMLP::synapse0xb510cf8() {
   return (neuron0xb50e350()*-0.27431);
}

double MuonTaggerNoIPMLP::synapse0xb510d20() {
   return (neuron0xb50e6c8()*0.338673);
}

double MuonTaggerNoIPMLP::synapse0xb510d48() {
   return (neuron0xb50e968()*-2.56297);
}

double MuonTaggerNoIPMLP::synapse0xb510d70() {
   return (neuron0xb50ec28()*0.0921012);
}

double MuonTaggerNoIPMLP::synapse0xb510d98() {
   return (neuron0xb50eee8()*1.95351);
}

double MuonTaggerNoIPMLP::synapse0xb510dc0() {
   return (neuron0xb50f1a8()*-0.170827);
}

double MuonTaggerNoIPMLP::synapse0xb510de8() {
   return (neuron0xb50f4f0()*0.0994031);
}

double MuonTaggerNoIPMLP::synapse0xb510e10() {
   return (neuron0xb50e5c0()*-0.337334);
}

double MuonTaggerNoIPMLP::synapse0xb510e38() {
   return (neuron0xb50fae8()*1.06742);
}

double MuonTaggerNoIPMLP::synapse0xb510e60() {
   return (neuron0xb50fda8()*-0.99141);
}

double MuonTaggerNoIPMLP::synapse0xb5110c0() {
   return (neuron0xb50d328()*0.573292);
}

double MuonTaggerNoIPMLP::synapse0xb50f768() {
   return (neuron0xb50d550()*2.05643);
}

double MuonTaggerNoIPMLP::synapse0xb50f790() {
   return (neuron0xb50d810()*-3.21657);
}

double MuonTaggerNoIPMLP::synapse0xb50f7b8() {
   return (neuron0xb50dad0()*6.13726);
}

double MuonTaggerNoIPMLP::synapse0xb50f7e0() {
   return (neuron0xb50de18()*0.222816);
}

double MuonTaggerNoIPMLP::synapse0xb50f808() {
   return (neuron0xb50e090()*-0.405157);
}

double MuonTaggerNoIPMLP::synapse0xb50f830() {
   return (neuron0xb50e350()*0.630832);
}

double MuonTaggerNoIPMLP::synapse0xb50f858() {
   return (neuron0xb50e6c8()*0.286139);
}

double MuonTaggerNoIPMLP::synapse0xb50f880() {
   return (neuron0xb50e968()*0.740421);
}

double MuonTaggerNoIPMLP::synapse0xb50f8a8() {
   return (neuron0xb50ec28()*0.61012);
}

double MuonTaggerNoIPMLP::synapse0xb50f8d0() {
   return (neuron0xb50eee8()*-4.67691);
}

double MuonTaggerNoIPMLP::synapse0xb50f8f8() {
   return (neuron0xb50f1a8()*-1.94479);
}

double MuonTaggerNoIPMLP::synapse0xb50f920() {
   return (neuron0xb50f4f0()*0.243948);
}

double MuonTaggerNoIPMLP::synapse0xb50f948() {
   return (neuron0xb50e5c0()*-0.0813435);
}

double MuonTaggerNoIPMLP::synapse0xb5114f0() {
   return (neuron0xb50fae8()*1.58896);
}

double MuonTaggerNoIPMLP::synapse0xb511518() {
   return (neuron0xb50fda8()*0.725111);
}

