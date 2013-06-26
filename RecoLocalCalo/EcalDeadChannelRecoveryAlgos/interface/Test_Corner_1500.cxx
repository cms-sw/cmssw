#include "Test_Corner_1500.h"
#include <cmath>

double Test_Corner_1500::value(int index,double in0,double in1,double in2) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   switch(index) {
     case 0:
         return ((neuron0xa841248()*1)+0);
     default:
         return 0.;
   }
}

double Test_Corner_1500::neuron0xa840ce8() {
   return input0;
}

double Test_Corner_1500::neuron0xa840e78() {
   return input1;
}

double Test_Corner_1500::neuron0xa841050() {
   return input2;
}

double Test_Corner_1500::input0xa841368() {
   double input = 0.95298;
   input += synapse0xa845b18();
   input += synapse0xa8414f8();
   input += synapse0xa841520();
   return input;
}

double Test_Corner_1500::neuron0xa841368() {
   double input = input0xa841368();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa841548() {
   double input = 0.796275;
   input += synapse0xa841720();
   input += synapse0xa841748();
   input += synapse0xa841770();
   return input;
}

double Test_Corner_1500::neuron0xa841548() {
   double input = input0xa841548();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa841798() {
   double input = 1.7544;
   input += synapse0xa841970();
   input += synapse0xa841998();
   input += synapse0xa8419c0();
   return input;
}

double Test_Corner_1500::neuron0xa841798() {
   double input = input0xa841798();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa8419e8() {
   double input = 0.131702;
   input += synapse0xa841be0();
   input += synapse0xa841c08();
   input += synapse0xa841c30();
   return input;
}

double Test_Corner_1500::neuron0xa8419e8() {
   double input = input0xa8419e8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa841c58() {
   double input = -0.203393;
   input += synapse0xa841e50();
   input += synapse0xa841e78();
   input += synapse0xa841ea0();
   return input;
}

double Test_Corner_1500::neuron0xa841c58() {
   double input = input0xa841c58();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa841ec8() {
   double input = -0.640446;
   input += synapse0xa8420c0();
   input += synapse0xa8420e8();
   input += synapse0xa842198();
   return input;
}

double Test_Corner_1500::neuron0xa841ec8() {
   double input = input0xa841ec8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa8421c0() {
   double input = 0.0571014;
   input += synapse0xa842370();
   input += synapse0xa842398();
   input += synapse0xa8423c0();
   return input;
}

double Test_Corner_1500::neuron0xa8421c0() {
   double input = input0xa8421c0();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa8423e8() {
   double input = 0.63165;
   input += synapse0xa8425e0();
   input += synapse0xa842608();
   input += synapse0xa842630();
   return input;
}

double Test_Corner_1500::neuron0xa8423e8() {
   double input = input0xa8423e8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa842658() {
   double input = 3.09874;
   input += synapse0xa842850();
   input += synapse0xa842878();
   input += synapse0xa8428a0();
   return input;
}

double Test_Corner_1500::neuron0xa842658() {
   double input = input0xa842658();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa8428c8() {
   double input = -0.445403;
   input += synapse0xa842ac0();
   input += synapse0xa842ae8();
   input += synapse0xa842b10();
   return input;
}

double Test_Corner_1500::neuron0xa8428c8() {
   double input = input0xa8428c8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa842b38() {
   double input = 0.278495;
   input += synapse0xa842d38();
   input += synapse0xa842d60();
   input += synapse0xa842d88();
   return input;
}

double Test_Corner_1500::neuron0xa842b38() {
   double input = input0xa842b38();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa842eb8() {
   double input = -0.612154;
   input += synapse0xa8430b8();
   input += synapse0xa8430e0();
   input += synapse0xa843108();
   return input;
}

double Test_Corner_1500::neuron0xa842eb8() {
   double input = input0xa842eb8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Corner_1500::input0xa841248() {
   double input = -0.0572088;
   input += synapse0xa8431c0();
   input += synapse0xa8431e8();
   input += synapse0xa843210();
   input += synapse0xa843238();
   input += synapse0xa843260();
   input += synapse0xa843288();
   input += synapse0xa8432b0();
   input += synapse0xa8432d8();
   input += synapse0xa843300();
   input += synapse0xa843328();
   input += synapse0xa843350();
   input += synapse0xa843378();
   return input;
}

double Test_Corner_1500::neuron0xa841248() {
   double input = input0xa841248();
   return (input * 1)+0;
}

double Test_Corner_1500::synapse0xa845b18() {
   return (neuron0xa840ce8()*0.276886);
}

double Test_Corner_1500::synapse0xa8414f8() {
   return (neuron0xa840e78()*-0.0309585);
}

double Test_Corner_1500::synapse0xa841520() {
   return (neuron0xa841050()*-1.51021);
}

double Test_Corner_1500::synapse0xa841720() {
   return (neuron0xa840ce8()*0.02515);
}

double Test_Corner_1500::synapse0xa841748() {
   return (neuron0xa840e78()*0.147998);
}

double Test_Corner_1500::synapse0xa841770() {
   return (neuron0xa841050()*-3.11258);
}

double Test_Corner_1500::synapse0xa841970() {
   return (neuron0xa840ce8()*-0.440272);
}

double Test_Corner_1500::synapse0xa841998() {
   return (neuron0xa840e78()*-0.620079);
}

double Test_Corner_1500::synapse0xa8419c0() {
   return (neuron0xa841050()*2.33571);
}

double Test_Corner_1500::synapse0xa841be0() {
   return (neuron0xa840ce8()*0.0060958);
}

double Test_Corner_1500::synapse0xa841c08() {
   return (neuron0xa840e78()*0.0247543);
}

double Test_Corner_1500::synapse0xa841c30() {
   return (neuron0xa841050()*0.000122817);
}

double Test_Corner_1500::synapse0xa841e50() {
   return (neuron0xa840ce8()*-0.257473);
}

double Test_Corner_1500::synapse0xa841e78() {
   return (neuron0xa840e78()*0.678953);
}

double Test_Corner_1500::synapse0xa841ea0() {
   return (neuron0xa841050()*-0.884389);
}

double Test_Corner_1500::synapse0xa8420c0() {
   return (neuron0xa840ce8()*-0.224937);
}

double Test_Corner_1500::synapse0xa8420e8() {
   return (neuron0xa840e78()*0.354063);
}

double Test_Corner_1500::synapse0xa842198() {
   return (neuron0xa841050()*-0.516678);
}

double Test_Corner_1500::synapse0xa842370() {
   return (neuron0xa840ce8()*-0.0579972);
}

double Test_Corner_1500::synapse0xa842398() {
   return (neuron0xa840e78()*0.342624);
}

double Test_Corner_1500::synapse0xa8423c0() {
   return (neuron0xa841050()*-0.925181);
}

double Test_Corner_1500::synapse0xa8425e0() {
   return (neuron0xa840ce8()*0.301493);
}

double Test_Corner_1500::synapse0xa842608() {
   return (neuron0xa840e78()*-1.05591);
}

double Test_Corner_1500::synapse0xa842630() {
   return (neuron0xa841050()*0.865177);
}

double Test_Corner_1500::synapse0xa842850() {
   return (neuron0xa840ce8()*-0.957436);
}

double Test_Corner_1500::synapse0xa842878() {
   return (neuron0xa840e78()*-0.997926);
}

double Test_Corner_1500::synapse0xa8428a0() {
   return (neuron0xa841050()*3.08568);
}

double Test_Corner_1500::synapse0xa842ac0() {
   return (neuron0xa840ce8()*0.0641783);
}

double Test_Corner_1500::synapse0xa842ae8() {
   return (neuron0xa840e78()*0.377656);
}

double Test_Corner_1500::synapse0xa842b10() {
   return (neuron0xa841050()*1.17774);
}

double Test_Corner_1500::synapse0xa842d38() {
   return (neuron0xa840ce8()*0.80131);
}

double Test_Corner_1500::synapse0xa842d60() {
   return (neuron0xa840e78()*0.1294);
}

double Test_Corner_1500::synapse0xa842d88() {
   return (neuron0xa841050()*-0.424347);
}

double Test_Corner_1500::synapse0xa8430b8() {
   return (neuron0xa840ce8()*0.53206);
}

double Test_Corner_1500::synapse0xa8430e0() {
   return (neuron0xa840e78()*0.314745);
}

double Test_Corner_1500::synapse0xa843108() {
   return (neuron0xa841050()*-0.859277);
}

double Test_Corner_1500::synapse0xa8431c0() {
   return (neuron0xa841368()*2.5844);
}

double Test_Corner_1500::synapse0xa8431e8() {
   return (neuron0xa841548()*-2.15982);
}

double Test_Corner_1500::synapse0xa843210() {
   return (neuron0xa841798()*-2.09542);
}

double Test_Corner_1500::synapse0xa843238() {
   return (neuron0xa8419e8()*-0.00693921);
}

double Test_Corner_1500::synapse0xa843260() {
   return (neuron0xa841c58()*1.09624);
}

double Test_Corner_1500::synapse0xa843288() {
   return (neuron0xa841ec8()*-1.23607);
}

double Test_Corner_1500::synapse0xa8432b0() {
   return (neuron0xa8421c0()*0.726722);
}

double Test_Corner_1500::synapse0xa8432d8() {
   return (neuron0xa8423e8()*0.33204);
}

double Test_Corner_1500::synapse0xa843300() {
   return (neuron0xa842658()*0.835795);
}

double Test_Corner_1500::synapse0xa843328() {
   return (neuron0xa8428c8()*0.652662);
}

double Test_Corner_1500::synapse0xa843350() {
   return (neuron0xa842b38()*-0.263563);
}

double Test_Corner_1500::synapse0xa843378() {
   return (neuron0xa842eb8()*-1.38574);
}

