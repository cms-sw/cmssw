#include "MuonTaggerMLP.h"
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

double MuonTaggerMLP::value(int index,double in0,double in1,double in2,double in3,double in4,double in5) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   input3 = (in3 - 0)/1;
   input4 = (in4 - 0)/1;
   input5 = (in5 - 0)/1;

   double out0 = neuron0xc44a968() * prior0;
   double out1 = neuron0xc44aca8() * prior1;
   double out2 = neuron0xc44b0b0() * prior2;
   double out3 = neuron0xc44b720() * prior3;
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

double MuonTaggerMLP::neuron0xc4459c0() {
   return input0;
}

double MuonTaggerMLP::neuron0xc445b70() {
   return input1;
}

double MuonTaggerMLP::neuron0xc445d48() {
   return input2;
}

double MuonTaggerMLP::neuron0xc446fe0() {
   return input3;
}

double MuonTaggerMLP::neuron0xc447170() {
   return input4;
}

double MuonTaggerMLP::neuron0xc447348() {
   return input5;
}

double MuonTaggerMLP::input0xc447638() {
   double input = -3.71752;
   input += synapse0xbf59610();
   input += synapse0xbf59840();
   input += synapse0xc447810();
   input += synapse0xc447838();
   input += synapse0xc447860();
   input += synapse0xc447888();
   return input;
}

double MuonTaggerMLP::neuron0xc447638() {
   double input = input0xc447638();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc4478b0() {
   double input = -0.0176005;
   input += synapse0xc447aa8();
   input += synapse0xc447ad0();
   input += synapse0xc447af8();
   input += synapse0xc447b20();
   input += synapse0xc447b48();
   input += synapse0xc447b70();
   return input;
}

double MuonTaggerMLP::neuron0xc4478b0() {
   double input = input0xc4478b0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc447b98() {
   double input = -0.273148;
   input += synapse0xc447d90();
   input += synapse0xc447db8();
   input += synapse0xc447de0();
   input += synapse0xc447e08();
   input += synapse0xc447e30();
   input += synapse0xc447ee0();
   return input;
}

double MuonTaggerMLP::neuron0xc447b98() {
   double input = input0xc447b98();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc447f08() {
   double input = -1.21887;
   input += synapse0xc4480b8();
   input += synapse0xc4480e0();
   input += synapse0xc448108();
   input += synapse0xc448130();
   input += synapse0xc448158();
   input += synapse0xc448180();
   return input;
}

double MuonTaggerMLP::neuron0xc447f08() {
   double input = input0xc447f08();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc4481a8() {
   double input = 0.0758877;
   input += synapse0xc4483a0();
   input += synapse0xc4483c8();
   input += synapse0xc4483f0();
   input += synapse0xc448418();
   input += synapse0xc448440();
   input += synapse0xc448468();
   return input;
}

double MuonTaggerMLP::neuron0xc4481a8() {
   double input = input0xc4481a8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc448490() {
   double input = 0.0737572;
   input += synapse0xc448688();
   input += synapse0xc4486b0();
   input += synapse0xc4486d8();
   input += synapse0xc447e58();
   input += synapse0xc447e80();
   input += synapse0xc447ea8();
   return input;
}

double MuonTaggerMLP::neuron0xc448490() {
   double input = input0xc448490();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc448808() {
   double input = 1.08264;
   input += synapse0xc448a00();
   input += synapse0xc448a28();
   input += synapse0xc448a50();
   input += synapse0xc448a78();
   input += synapse0xc448aa0();
   input += synapse0xc448ac8();
   return input;
}

double MuonTaggerMLP::neuron0xc448808() {
   double input = input0xc448808();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc448af0() {
   double input = -2.50121;
   input += synapse0xc448ce8();
   input += synapse0xc448d10();
   input += synapse0xc448d38();
   input += synapse0xc448d60();
   input += synapse0xc448d88();
   input += synapse0xc448db0();
   return input;
}

double MuonTaggerMLP::neuron0xc448af0() {
   double input = input0xc448af0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc448dd8() {
   double input = -1.47117;
   input += synapse0xc448fd0();
   input += synapse0xc448ff8();
   input += synapse0xc449020();
   input += synapse0xc449048();
   input += synapse0xc449070();
   input += synapse0xc449098();
   return input;
}

double MuonTaggerMLP::neuron0xc448dd8() {
   double input = input0xc448dd8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc4490c0() {
   double input = -5.23482;
   input += synapse0xc4492b8();
   input += synapse0xc4492e0();
   input += synapse0xc449308();
   input += synapse0xc449330();
   input += synapse0xc449358();
   input += synapse0xc449380();
   return input;
}

double MuonTaggerMLP::neuron0xc4490c0() {
   double input = input0xc4490c0();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc4493a8() {
   double input = -0.717508;
   input += synapse0xc449628();
   input += synapse0xc449650();
   input += synapse0xc449678();
   input += synapse0xc4496a0();
   input += synapse0xc4496c8();
   input += synapse0xc445910();
   return input;
}

double MuonTaggerMLP::neuron0xc4493a8() {
   double input = input0xc4493a8();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc448700() {
   double input = -3.58367;
   input += synapse0xbf591e0();
   input += synapse0xc4499d0();
   input += synapse0xc4499f8();
   input += synapse0xc449a20();
   input += synapse0xc449a48();
   input += synapse0xc449a70();
   return input;
}

double MuonTaggerMLP::neuron0xc448700() {
   double input = input0xc448700();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc449a98() {
   double input = 0.148087;
   input += synapse0xc449c90();
   input += synapse0xc449cb8();
   input += synapse0xc449ce0();
   input += synapse0xc449d08();
   input += synapse0xc449d30();
   input += synapse0xc449d58();
   return input;
}

double MuonTaggerMLP::neuron0xc449a98() {
   double input = input0xc449a98();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc449d80() {
   double input = -1.64762;
   input += synapse0xc449f78();
   input += synapse0xc449fa0();
   input += synapse0xc449fc8();
   input += synapse0xc449ff0();
   input += synapse0xc44a018();
   input += synapse0xc44a040();
   return input;
}

double MuonTaggerMLP::neuron0xc449d80() {
   double input = input0xc449d80();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc44a068() {
   double input = 0.299243;
   input += synapse0xc44a260();
   input += synapse0xc44a288();
   input += synapse0xc44a2b0();
   input += synapse0xc44a2d8();
   input += synapse0xc44a300();
   input += synapse0xc44a328();
   return input;
}

double MuonTaggerMLP::neuron0xc44a068() {
   double input = input0xc44a068();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc44a350() {
   double input = 1.64807;
   input += synapse0xc44a548();
   input += synapse0xc44a5f8();
   input += synapse0xc44a6a8();
   input += synapse0xc44a758();
   input += synapse0xc44a808();
   input += synapse0xc44a8b8();
   return input;
}

double MuonTaggerMLP::neuron0xc44a350() {
   double input = input0xc44a350();
   return (input < -100) ? 0 : (input > 100) ? 1 : (1/(1+exp(-input)));
}

double MuonTaggerMLP::input0xc44a968() {
   double input = -0.160732;
   input += synapse0xc445978();
   input += synapse0xc447590();
   input += synapse0xc4475b8();
   input += synapse0xc4475e0();
   input += synapse0xc447608();
   input += synapse0xc44aa68();
   input += synapse0xc44aa90();
   input += synapse0xc44aab8();
   input += synapse0xc44aae0();
   input += synapse0xc44ab08();
   input += synapse0xc44ab30();
   input += synapse0xc44ab58();
   input += synapse0xc44ab80();
   input += synapse0xc44aba8();
   input += synapse0xc44abd0();
   input += synapse0xc44abf8();
   return input;
}

double MuonTaggerMLP::neuron0xc44a968() {
   double input = input0xc44a968();
   return (exp(input) / (exp(input0xc44a968()) + exp(input0xc44aca8()) + exp(input0xc44b0b0()) + exp(input0xc44b720())) * 1)+0;
}

double MuonTaggerMLP::input0xc44aca8() {
   double input = -2.49703;
   input += synapse0xc44ada8();
   input += synapse0xc44add0();
   input += synapse0xc44adf8();
   input += synapse0xc44ae20();
   input += synapse0xc44ae48();
   input += synapse0xc44ae70();
   input += synapse0xc44ae98();
   input += synapse0xc44aec0();
   input += synapse0xc44aee8();
   input += synapse0xc44af10();
   input += synapse0xc44af38();
   input += synapse0xc44af60();
   input += synapse0xc44af88();
   input += synapse0xc44afb0();
   input += synapse0xc44afd8();
   input += synapse0xc44b000();
   return input;
}

double MuonTaggerMLP::neuron0xc44aca8() {
   double input = input0xc44aca8();
   return (exp(input) / (exp(input0xc44a968()) + exp(input0xc44aca8()) + exp(input0xc44b0b0()) + exp(input0xc44b720())) * 1)+0;
}

double MuonTaggerMLP::input0xc44b0b0() {
   double input = 0.400375;
   input += synapse0xc44b218();
   input += synapse0xc4496f0();
   input += synapse0xc449718();
   input += synapse0xc449740();
   input += synapse0xc449768();
   input += synapse0xc449790();
   input += synapse0xc4497b8();
   input += synapse0xc4497e0();
   input += synapse0xc449808();
   input += synapse0xc449830();
   input += synapse0xc449858();
   input += synapse0xc449880();
   input += synapse0xc4498a8();
   input += synapse0xc4498d0();
   input += synapse0xc44b648();
   input += synapse0xc44b670();
   return input;
}

double MuonTaggerMLP::neuron0xc44b0b0() {
   double input = input0xc44b0b0();
   return (exp(input) / (exp(input0xc44a968()) + exp(input0xc44aca8()) + exp(input0xc44b0b0()) + exp(input0xc44b720())) * 1)+0;
}

double MuonTaggerMLP::input0xc44b720() {
   double input = 1.54635;
   input += synapse0xc44b8d0();
   input += synapse0xc44b8f8();
   input += synapse0xc44b920();
   input += synapse0xc44b948();
   input += synapse0xc44b970();
   input += synapse0xc44b998();
   input += synapse0xc44b9c0();
   input += synapse0xc44b9e8();
   input += synapse0xc44ba10();
   input += synapse0xc44ba38();
   input += synapse0xc44ba60();
   input += synapse0xc44ba88();
   input += synapse0xc44bab0();
   input += synapse0xc44bad8();
   input += synapse0xc44bb00();
   input += synapse0xc44bb28();
   return input;
}

double MuonTaggerMLP::neuron0xc44b720() {
   double input = input0xc44b720();
   return (exp(input) / (exp(input0xc44a968()) + exp(input0xc44aca8()) + exp(input0xc44b0b0()) + exp(input0xc44b720())) * 1)+0;
}

double MuonTaggerMLP::synapse0xbf59610() {
   return (neuron0xc4459c0()*0.613753);
}

double MuonTaggerMLP::synapse0xbf59840() {
   return (neuron0xc445b70()*0.70502);
}

double MuonTaggerMLP::synapse0xc447810() {
   return (neuron0xc445d48()*-0.708544);
}

double MuonTaggerMLP::synapse0xc447838() {
   return (neuron0xc446fe0()*-0.0047014);
}

double MuonTaggerMLP::synapse0xc447860() {
   return (neuron0xc447170()*-0.132441);
}

double MuonTaggerMLP::synapse0xc447888() {
   return (neuron0xc447348()*0.0775376);
}

double MuonTaggerMLP::synapse0xc447aa8() {
   return (neuron0xc4459c0()*-1.2834);
}

double MuonTaggerMLP::synapse0xc447ad0() {
   return (neuron0xc445b70()*0.163381);
}

double MuonTaggerMLP::synapse0xc447af8() {
   return (neuron0xc445d48()*1.97762);
}

double MuonTaggerMLP::synapse0xc447b20() {
   return (neuron0xc446fe0()*-0.018039);
}

double MuonTaggerMLP::synapse0xc447b48() {
   return (neuron0xc447170()*-0.190593);
}

double MuonTaggerMLP::synapse0xc447b70() {
   return (neuron0xc447348()*-0.0622746);
}

double MuonTaggerMLP::synapse0xc447d90() {
   return (neuron0xc4459c0()*-0.000461921);
}

double MuonTaggerMLP::synapse0xc447db8() {
   return (neuron0xc445b70()*0.333507);
}

double MuonTaggerMLP::synapse0xc447de0() {
   return (neuron0xc445d48()*0.415889);
}

double MuonTaggerMLP::synapse0xc447e08() {
   return (neuron0xc446fe0()*-1.85686);
}

double MuonTaggerMLP::synapse0xc447e30() {
   return (neuron0xc447170()*0.539882);
}

double MuonTaggerMLP::synapse0xc447ee0() {
   return (neuron0xc447348()*-0.633015);
}

double MuonTaggerMLP::synapse0xc4480b8() {
   return (neuron0xc4459c0()*-0.22985);
}

double MuonTaggerMLP::synapse0xc4480e0() {
   return (neuron0xc445b70()*0.851132);
}

double MuonTaggerMLP::synapse0xc448108() {
   return (neuron0xc445d48()*-0.25402);
}

double MuonTaggerMLP::synapse0xc448130() {
   return (neuron0xc446fe0()*-0.0771693);
}

double MuonTaggerMLP::synapse0xc448158() {
   return (neuron0xc447170()*-2.69593);
}

double MuonTaggerMLP::synapse0xc448180() {
   return (neuron0xc447348()*0.484327);
}

double MuonTaggerMLP::synapse0xc4483a0() {
   return (neuron0xc4459c0()*0.0695208);
}

double MuonTaggerMLP::synapse0xc4483c8() {
   return (neuron0xc445b70()*-0.963389);
}

double MuonTaggerMLP::synapse0xc4483f0() {
   return (neuron0xc445d48()*0.393606);
}

double MuonTaggerMLP::synapse0xc448418() {
   return (neuron0xc446fe0()*1.62142);
}

double MuonTaggerMLP::synapse0xc448440() {
   return (neuron0xc447170()*-0.264814);
}

double MuonTaggerMLP::synapse0xc448468() {
   return (neuron0xc447348()*0.0975774);
}

double MuonTaggerMLP::synapse0xc448688() {
   return (neuron0xc4459c0()*-0.322286);
}

double MuonTaggerMLP::synapse0xc4486b0() {
   return (neuron0xc445b70()*-0.875564);
}

double MuonTaggerMLP::synapse0xc4486d8() {
   return (neuron0xc445d48()*0.398738);
}

double MuonTaggerMLP::synapse0xc447e58() {
   return (neuron0xc446fe0()*1.11604);
}

double MuonTaggerMLP::synapse0xc447e80() {
   return (neuron0xc447170()*0.264483);
}

double MuonTaggerMLP::synapse0xc447ea8() {
   return (neuron0xc447348()*-0.183035);
}

double MuonTaggerMLP::synapse0xc448a00() {
   return (neuron0xc4459c0()*-0.873237);
}

double MuonTaggerMLP::synapse0xc448a28() {
   return (neuron0xc445b70()*0.600683);
}

double MuonTaggerMLP::synapse0xc448a50() {
   return (neuron0xc445d48()*3.10378);
}

double MuonTaggerMLP::synapse0xc448a78() {
   return (neuron0xc446fe0()*-0.00327025);
}

double MuonTaggerMLP::synapse0xc448aa0() {
   return (neuron0xc447170()*-0.0243287);
}

double MuonTaggerMLP::synapse0xc448ac8() {
   return (neuron0xc447348()*-0.0005311);
}

double MuonTaggerMLP::synapse0xc448ce8() {
   return (neuron0xc4459c0()*0.415053);
}

double MuonTaggerMLP::synapse0xc448d10() {
   return (neuron0xc445b70()*-0.16301);
}

double MuonTaggerMLP::synapse0xc448d38() {
   return (neuron0xc445d48()*2.85804);
}

double MuonTaggerMLP::synapse0xc448d60() {
   return (neuron0xc446fe0()*5.93533e-05);
}

double MuonTaggerMLP::synapse0xc448d88() {
   return (neuron0xc447170()*-0.0949804);
}

double MuonTaggerMLP::synapse0xc448db0() {
   return (neuron0xc447348()*0.286721);
}

double MuonTaggerMLP::synapse0xc448fd0() {
   return (neuron0xc4459c0()*-0.341524);
}

double MuonTaggerMLP::synapse0xc448ff8() {
   return (neuron0xc445b70()*-1.5349);
}

double MuonTaggerMLP::synapse0xc449020() {
   return (neuron0xc445d48()*6.40865);
}

double MuonTaggerMLP::synapse0xc449048() {
   return (neuron0xc446fe0()*0.000506998);
}

double MuonTaggerMLP::synapse0xc449070() {
   return (neuron0xc447170()*0.0100732);
}

double MuonTaggerMLP::synapse0xc449098() {
   return (neuron0xc447348()*-0.000924829);
}

double MuonTaggerMLP::synapse0xc4492b8() {
   return (neuron0xc4459c0()*5.03569);
}

double MuonTaggerMLP::synapse0xc4492e0() {
   return (neuron0xc445b70()*-0.155476);
}

double MuonTaggerMLP::synapse0xc449308() {
   return (neuron0xc445d48()*0.560566);
}

double MuonTaggerMLP::synapse0xc449330() {
   return (neuron0xc446fe0()*0.00374005);
}

double MuonTaggerMLP::synapse0xc449358() {
   return (neuron0xc447170()*0.127987);
}

double MuonTaggerMLP::synapse0xc449380() {
   return (neuron0xc447348()*0.00148012);
}

double MuonTaggerMLP::synapse0xc449628() {
   return (neuron0xc4459c0()*0.0508729);
}

double MuonTaggerMLP::synapse0xc449650() {
   return (neuron0xc445b70()*0.206212);
}

double MuonTaggerMLP::synapse0xc449678() {
   return (neuron0xc445d48()*1.71108);
}

double MuonTaggerMLP::synapse0xc4496a0() {
   return (neuron0xc446fe0()*0.000291626);
}

double MuonTaggerMLP::synapse0xc4496c8() {
   return (neuron0xc447170()*-0.00678763);
}

double MuonTaggerMLP::synapse0xc445910() {
   return (neuron0xc447348()*0.286167);
}

double MuonTaggerMLP::synapse0xbf591e0() {
   return (neuron0xc4459c0()*-1.33311);
}

double MuonTaggerMLP::synapse0xc4499d0() {
   return (neuron0xc445b70()*0.549506);
}

double MuonTaggerMLP::synapse0xc4499f8() {
   return (neuron0xc445d48()*-0.0984853);
}

double MuonTaggerMLP::synapse0xc449a20() {
   return (neuron0xc446fe0()*-0.00267527);
}

double MuonTaggerMLP::synapse0xc449a48() {
   return (neuron0xc447170()*0.0929378);
}

double MuonTaggerMLP::synapse0xc449a70() {
   return (neuron0xc447348()*1.7219);
}

double MuonTaggerMLP::synapse0xc449c90() {
   return (neuron0xc4459c0()*-0.88777);
}

double MuonTaggerMLP::synapse0xc449cb8() {
   return (neuron0xc445b70()*0.82276);
}

double MuonTaggerMLP::synapse0xc449ce0() {
   return (neuron0xc445d48()*-0.204261);
}

double MuonTaggerMLP::synapse0xc449d08() {
   return (neuron0xc446fe0()*-0.099686);
}

double MuonTaggerMLP::synapse0xc449d30() {
   return (neuron0xc447170()*-0.0676644);
}

double MuonTaggerMLP::synapse0xc449d58() {
   return (neuron0xc447348()*-0.169774);
}

double MuonTaggerMLP::synapse0xc449f78() {
   return (neuron0xc4459c0()*-0.0398536);
}

double MuonTaggerMLP::synapse0xc449fa0() {
   return (neuron0xc445b70()*-0.138676);
}

double MuonTaggerMLP::synapse0xc449fc8() {
   return (neuron0xc445d48()*2.15938);
}

double MuonTaggerMLP::synapse0xc449ff0() {
   return (neuron0xc446fe0()*0.023284);
}

double MuonTaggerMLP::synapse0xc44a018() {
   return (neuron0xc447170()*0.214466);
}

double MuonTaggerMLP::synapse0xc44a040() {
   return (neuron0xc447348()*0.0114212);
}

double MuonTaggerMLP::synapse0xc44a260() {
   return (neuron0xc4459c0()*0.288223);
}

double MuonTaggerMLP::synapse0xc44a288() {
   return (neuron0xc445b70()*0.0381795);
}

double MuonTaggerMLP::synapse0xc44a2b0() {
   return (neuron0xc445d48()*-0.292561);
}

double MuonTaggerMLP::synapse0xc44a2d8() {
   return (neuron0xc446fe0()*-0.615402);
}

double MuonTaggerMLP::synapse0xc44a300() {
   return (neuron0xc447170()*-0.514053);
}

double MuonTaggerMLP::synapse0xc44a328() {
   return (neuron0xc447348()*0.490844);
}

double MuonTaggerMLP::synapse0xc44a548() {
   return (neuron0xc4459c0()*-2.34555);
}

double MuonTaggerMLP::synapse0xc44a5f8() {
   return (neuron0xc445b70()*0.135258);
}

double MuonTaggerMLP::synapse0xc44a6a8() {
   return (neuron0xc445d48()*-3.36648);
}

double MuonTaggerMLP::synapse0xc44a758() {
   return (neuron0xc446fe0()*-0.00840276);
}

double MuonTaggerMLP::synapse0xc44a808() {
   return (neuron0xc447170()*-0.0721345);
}

double MuonTaggerMLP::synapse0xc44a8b8() {
   return (neuron0xc447348()*-0.0927585);
}

double MuonTaggerMLP::synapse0xc445978() {
   return (neuron0xc447638()*0.956711);
}

double MuonTaggerMLP::synapse0xc447590() {
   return (neuron0xc4478b0()*-2.91171);
}

double MuonTaggerMLP::synapse0xc4475b8() {
   return (neuron0xc447b98()*-0.624696);
}

double MuonTaggerMLP::synapse0xc4475e0() {
   return (neuron0xc447f08()*-0.524643);
}

double MuonTaggerMLP::synapse0xc447608() {
   return (neuron0xc4481a8()*-1.18143);
}

double MuonTaggerMLP::synapse0xc44aa68() {
   return (neuron0xc448490()*-0.358965);
}

double MuonTaggerMLP::synapse0xc44aa90() {
   return (neuron0xc448808()*1.20703);
}

double MuonTaggerMLP::synapse0xc44aab8() {
   return (neuron0xc448af0()*0.776583);
}

double MuonTaggerMLP::synapse0xc44aae0() {
   return (neuron0xc448dd8()*2.17228);
}

double MuonTaggerMLP::synapse0xc44ab08() {
   return (neuron0xc4490c0()*1.93271);
}

double MuonTaggerMLP::synapse0xc44ab30() {
   return (neuron0xc4493a8()*0.516479);
}

double MuonTaggerMLP::synapse0xc44ab58() {
   return (neuron0xc448700()*-0.0915335);
}

double MuonTaggerMLP::synapse0xc44ab80() {
   return (neuron0xc449a98()*1.19571);
}

double MuonTaggerMLP::synapse0xc44aba8() {
   return (neuron0xc449d80()*-1.49047);
}

double MuonTaggerMLP::synapse0xc44abd0() {
   return (neuron0xc44a068()*-0.00757379);
}

double MuonTaggerMLP::synapse0xc44abf8() {
   return (neuron0xc44a350()*2.07263);
}

double MuonTaggerMLP::synapse0xc44ada8() {
   return (neuron0xc447638()*-1.19103);
}

double MuonTaggerMLP::synapse0xc44add0() {
   return (neuron0xc4478b0()*-0.288857);
}

double MuonTaggerMLP::synapse0xc44adf8() {
   return (neuron0xc447b98()*0.633768);
}

double MuonTaggerMLP::synapse0xc44ae20() {
   return (neuron0xc447f08()*0.172425);
}

double MuonTaggerMLP::synapse0xc44ae48() {
   return (neuron0xc4481a8()*2.45962);
}

double MuonTaggerMLP::synapse0xc44ae70() {
   return (neuron0xc448490()*-0.00395761);
}

double MuonTaggerMLP::synapse0xc44ae98() {
   return (neuron0xc448808()*0.463354);
}

double MuonTaggerMLP::synapse0xc44aec0() {
   return (neuron0xc448af0()*1.17911);
}

double MuonTaggerMLP::synapse0xc44aee8() {
   return (neuron0xc448dd8()*-1.53556);
}

double MuonTaggerMLP::synapse0xc44af10() {
   return (neuron0xc4490c0()*-0.567438);
}

double MuonTaggerMLP::synapse0xc44af38() {
   return (neuron0xc4493a8()*-2.25013);
}

double MuonTaggerMLP::synapse0xc44af60() {
   return (neuron0xc448700()*1.38231);
}

double MuonTaggerMLP::synapse0xc44af88() {
   return (neuron0xc449a98()*0.212396);
}

double MuonTaggerMLP::synapse0xc44afb0() {
   return (neuron0xc449d80()*0.616213);
}

double MuonTaggerMLP::synapse0xc44afd8() {
   return (neuron0xc44a068()*0.106496);
}

double MuonTaggerMLP::synapse0xc44b000() {
   return (neuron0xc44a350()*-0.585778);
}

double MuonTaggerMLP::synapse0xc44b218() {
   return (neuron0xc447638()*-0.239202);
}

double MuonTaggerMLP::synapse0xc4496f0() {
   return (neuron0xc4478b0()*1.87372);
}

double MuonTaggerMLP::synapse0xc449718() {
   return (neuron0xc447b98()*0.1833);
}

double MuonTaggerMLP::synapse0xc449740() {
   return (neuron0xc447f08()*-0.316417);
}

double MuonTaggerMLP::synapse0xc449768() {
   return (neuron0xc4481a8()*0.0166233);
}

double MuonTaggerMLP::synapse0xc449790() {
   return (neuron0xc448490()*0.0610624);
}

double MuonTaggerMLP::synapse0xc4497b8() {
   return (neuron0xc448808()*0.00435751);
}

double MuonTaggerMLP::synapse0xc4497e0() {
   return (neuron0xc448af0()*-1.4395);
}

double MuonTaggerMLP::synapse0xc449808() {
   return (neuron0xc448dd8()*-5.58704);
}

double MuonTaggerMLP::synapse0xc449830() {
   return (neuron0xc4490c0()*-0.962024);
}

double MuonTaggerMLP::synapse0xc449858() {
   return (neuron0xc4493a8()*1.55258);
}

double MuonTaggerMLP::synapse0xc449880() {
   return (neuron0xc448700()*-0.643586);
}

double MuonTaggerMLP::synapse0xc4498a8() {
   return (neuron0xc449a98()*-1.71565);
}

double MuonTaggerMLP::synapse0xc4498d0() {
   return (neuron0xc449d80()*-0.013779);
}

double MuonTaggerMLP::synapse0xc44b648() {
   return (neuron0xc44a068()*0.196737);
}

double MuonTaggerMLP::synapse0xc44b670() {
   return (neuron0xc44a350()*-0.910842);
}

double MuonTaggerMLP::synapse0xc44b8d0() {
   return (neuron0xc447638()*0.328149);
}

double MuonTaggerMLP::synapse0xc44b8f8() {
   return (neuron0xc4478b0()*0.827109);
}

double MuonTaggerMLP::synapse0xc44b920() {
   return (neuron0xc447b98()*-0.957688);
}

double MuonTaggerMLP::synapse0xc44b948() {
   return (neuron0xc447f08()*0.0765428);
}

double MuonTaggerMLP::synapse0xc44b970() {
   return (neuron0xc4481a8()*-0.213007);
}

double MuonTaggerMLP::synapse0xc44b998() {
   return (neuron0xc448490()*0.628048);
}

double MuonTaggerMLP::synapse0xc44b9c0() {
   return (neuron0xc448808()*-1.29968);
}

double MuonTaggerMLP::synapse0xc44b9e8() {
   return (neuron0xc448af0()*-0.401453);
}

double MuonTaggerMLP::synapse0xc44ba10() {
   return (neuron0xc448dd8()*4.78859);
}

double MuonTaggerMLP::synapse0xc44ba38() {
   return (neuron0xc4490c0()*-0.0636783);
}

double MuonTaggerMLP::synapse0xc44ba60() {
   return (neuron0xc4493a8()*-0.091912);
}

double MuonTaggerMLP::synapse0xc44ba88() {
   return (neuron0xc448700()*-0.407086);
}

double MuonTaggerMLP::synapse0xc44bab0() {
   return (neuron0xc449a98()*-0.0835674);
}

double MuonTaggerMLP::synapse0xc44bad8() {
   return (neuron0xc449d80()*0.57047);
}

double MuonTaggerMLP::synapse0xc44bb00() {
   return (neuron0xc44a068()*0.303081);
}

double MuonTaggerMLP::synapse0xc44bb28() {
   return (neuron0xc44a350()*-0.843993);
}

