#include "Test_Side_1500.h"
#include <cmath>

double Test_Side_1500::value(int index,double in0,double in1,double in2) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   switch(index) {
     case 0:
         return ((neuron0xa9de248()*1)+0);
     default:
         return 0.;
   }
}

double Test_Side_1500::neuron0xa9ddce8() {
   return input0;
}

double Test_Side_1500::neuron0xa9dde78() {
   return input1;
}

double Test_Side_1500::neuron0xa9de050() {
   return input2;
}

double Test_Side_1500::input0xa9de368() {
   double input = -0.395752;
   input += synapse0xa9e2b18();
   input += synapse0xa9de4f8();
   input += synapse0xa9de520();
   return input;
}

double Test_Side_1500::neuron0xa9de368() {
   double input = input0xa9de368();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9de548() {
   double input = 0.123288;
   input += synapse0xa9de720();
   input += synapse0xa9de748();
   input += synapse0xa9de770();
   return input;
}

double Test_Side_1500::neuron0xa9de548() {
   double input = input0xa9de548();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9de798() {
   double input = -0.0932859;
   input += synapse0xa9de970();
   input += synapse0xa9de998();
   input += synapse0xa9de9c0();
   return input;
}

double Test_Side_1500::neuron0xa9de798() {
   double input = input0xa9de798();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9de9e8() {
   double input = 4.13841;
   input += synapse0xa9debe0();
   input += synapse0xa9dec08();
   input += synapse0xa9dec30();
   return input;
}

double Test_Side_1500::neuron0xa9de9e8() {
   double input = input0xa9de9e8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9dec58() {
   double input = 10.0738;
   input += synapse0xa9dee50();
   input += synapse0xa9dee78();
   input += synapse0xa9deea0();
   return input;
}

double Test_Side_1500::neuron0xa9dec58() {
   double input = input0xa9dec58();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9deec8() {
   double input = 5.63011;
   input += synapse0xa9df0c0();
   input += synapse0xa9df0e8();
   input += synapse0xa9df198();
   return input;
}

double Test_Side_1500::neuron0xa9deec8() {
   double input = input0xa9deec8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9df1c0() {
   double input = 5.97351;
   input += synapse0xa9df370();
   input += synapse0xa9df398();
   input += synapse0xa9df3c0();
   return input;
}

double Test_Side_1500::neuron0xa9df1c0() {
   double input = input0xa9df1c0();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9df3e8() {
   double input = -0.708511;
   input += synapse0xa9df5e0();
   input += synapse0xa9df608();
   input += synapse0xa9df630();
   return input;
}

double Test_Side_1500::neuron0xa9df3e8() {
   double input = input0xa9df3e8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9df658() {
   double input = -0.109748;
   input += synapse0xa9df850();
   input += synapse0xa9df878();
   input += synapse0xa9df8a0();
   return input;
}

double Test_Side_1500::neuron0xa9df658() {
   double input = input0xa9df658();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9df8c8() {
   double input = 0.189699;
   input += synapse0xa9dfac0();
   input += synapse0xa9dfae8();
   input += synapse0xa9dfb10();
   return input;
}

double Test_Side_1500::neuron0xa9df8c8() {
   double input = input0xa9df8c8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9dfb38() {
   double input = 0.640774;
   input += synapse0xa9dfd38();
   input += synapse0xa9dfd60();
   input += synapse0xa9dfd88();
   return input;
}

double Test_Side_1500::neuron0xa9dfb38() {
   double input = input0xa9dfb38();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9dfeb8() {
   double input = 5.85528;
   input += synapse0xa9e00b8();
   input += synapse0xa9e00e0();
   input += synapse0xa9e0108();
   return input;
}

double Test_Side_1500::neuron0xa9dfeb8() {
   double input = input0xa9dfeb8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Side_1500::input0xa9de248() {
   double input = -2.67668;
   input += synapse0xa9e01c0();
   input += synapse0xa9e01e8();
   input += synapse0xa9e0210();
   input += synapse0xa9e0238();
   input += synapse0xa9e0260();
   input += synapse0xa9e0288();
   input += synapse0xa9e02b0();
   input += synapse0xa9e02d8();
   input += synapse0xa9e0300();
   input += synapse0xa9e0328();
   input += synapse0xa9e0350();
   input += synapse0xa9e0378();
   return input;
}

double Test_Side_1500::neuron0xa9de248() {
   double input = input0xa9de248();
   return (input * 1)+0;
}

double Test_Side_1500::synapse0xa9e2b18() {
   return (neuron0xa9ddce8()*-0.569663);
}

double Test_Side_1500::synapse0xa9de4f8() {
   return (neuron0xa9dde78()*0.478153);
}

double Test_Side_1500::synapse0xa9de520() {
   return (neuron0xa9de050()*1.93914);
}

double Test_Side_1500::synapse0xa9de720() {
   return (neuron0xa9ddce8()*0.0391599);
}

double Test_Side_1500::synapse0xa9de748() {
   return (neuron0xa9dde78()*-0.266515);
}

double Test_Side_1500::synapse0xa9de770() {
   return (neuron0xa9de050()*1.08551);
}

double Test_Side_1500::synapse0xa9de970() {
   return (neuron0xa9ddce8()*0.803313);
}

double Test_Side_1500::synapse0xa9de998() {
   return (neuron0xa9dde78()*0.576498);
}

double Test_Side_1500::synapse0xa9de9c0() {
   return (neuron0xa9de050()*0.128775);
}

double Test_Side_1500::synapse0xa9debe0() {
   return (neuron0xa9ddce8()*4.01828);
}

double Test_Side_1500::synapse0xa9dec08() {
   return (neuron0xa9dde78()*-0.192983);
}

double Test_Side_1500::synapse0xa9dec30() {
   return (neuron0xa9de050()*9.71096);
}

double Test_Side_1500::synapse0xa9dee50() {
   return (neuron0xa9ddce8()*0.130542);
}

double Test_Side_1500::synapse0xa9dee78() {
   return (neuron0xa9dde78()*0.0180835);
}

double Test_Side_1500::synapse0xa9deea0() {
   return (neuron0xa9de050()*-11.3045);
}

double Test_Side_1500::synapse0xa9df0c0() {
   return (neuron0xa9ddce8()*3.52456);
}

double Test_Side_1500::synapse0xa9df0e8() {
   return (neuron0xa9dde78()*-0.194451);
}

double Test_Side_1500::synapse0xa9df198() {
   return (neuron0xa9de050()*6.59806);
}

double Test_Side_1500::synapse0xa9df370() {
   return (neuron0xa9ddce8()*0.0455052);
}

double Test_Side_1500::synapse0xa9df398() {
   return (neuron0xa9dde78()*-0.269476);
}

double Test_Side_1500::synapse0xa9df3c0() {
   return (neuron0xa9de050()*-5.55112);
}

double Test_Side_1500::synapse0xa9df5e0() {
   return (neuron0xa9ddce8()*-0.233442);
}

double Test_Side_1500::synapse0xa9df608() {
   return (neuron0xa9dde78()*0.0123811);
}

double Test_Side_1500::synapse0xa9df630() {
   return (neuron0xa9de050()*0.680141);
}

double Test_Side_1500::synapse0xa9df850() {
   return (neuron0xa9ddce8()*-0.134326);
}

double Test_Side_1500::synapse0xa9df878() {
   return (neuron0xa9dde78()*-0.0541828);
}

double Test_Side_1500::synapse0xa9df8a0() {
   return (neuron0xa9de050()*0.504204);
}

double Test_Side_1500::synapse0xa9dfac0() {
   return (neuron0xa9ddce8()*-0.0972752);
}

double Test_Side_1500::synapse0xa9dfae8() {
   return (neuron0xa9dde78()*-0.111198);
}

double Test_Side_1500::synapse0xa9dfb10() {
   return (neuron0xa9de050()*-1.766);
}

double Test_Side_1500::synapse0xa9dfd38() {
   return (neuron0xa9ddce8()*0.858686);
}

double Test_Side_1500::synapse0xa9dfd60() {
   return (neuron0xa9dde78()*-0.308873);
}

double Test_Side_1500::synapse0xa9dfd88() {
   return (neuron0xa9de050()*0.193227);
}

double Test_Side_1500::synapse0xa9e00b8() {
   return (neuron0xa9ddce8()*-0.117526);
}

double Test_Side_1500::synapse0xa9e00e0() {
   return (neuron0xa9dde78()*0.225148);
}

double Test_Side_1500::synapse0xa9e0108() {
   return (neuron0xa9de050()*-5.90925);
}

double Test_Side_1500::synapse0xa9e01c0() {
   return (neuron0xa9de368()*-0.977624);
}

double Test_Side_1500::synapse0xa9e01e8() {
   return (neuron0xa9de548()*-2.70834);
}

double Test_Side_1500::synapse0xa9e0210() {
   return (neuron0xa9de798()*0.188397);
}

double Test_Side_1500::synapse0xa9e0238() {
   return (neuron0xa9de9e8()*-2.69945);
}

double Test_Side_1500::synapse0xa9e0260() {
   return (neuron0xa9dec58()*-6.91854);
}

double Test_Side_1500::synapse0xa9e0288() {
   return (neuron0xa9deec8()*2.95758);
}

double Test_Side_1500::synapse0xa9e02b0() {
   return (neuron0xa9df1c0()*6.93083);
}

double Test_Side_1500::synapse0xa9e02d8() {
   return (neuron0xa9df3e8()*-1.59696);
}

double Test_Side_1500::synapse0xa9e0300() {
   return (neuron0xa9df658()*-1.4824);
}

double Test_Side_1500::synapse0xa9e0328() {
   return (neuron0xa9df8c8()*1.27217);
}

double Test_Side_1500::synapse0xa9e0350() {
   return (neuron0xa9dfb38()*0.506543);
}

double Test_Side_1500::synapse0xa9e0378() {
   return (neuron0xa9dfeb8()*7.14613);
}

