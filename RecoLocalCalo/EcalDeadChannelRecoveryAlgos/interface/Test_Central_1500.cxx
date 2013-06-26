#include "Test_Central_1500.h"
#include <cmath>

double Test_Central_1500::value(int index,double in0,double in1,double in2) {
   input0 = (in0 - 0)/1;
   input1 = (in1 - 0)/1;
   input2 = (in2 - 0)/1;
   switch(index) {
     case 0:
         return ((neuron0xa7ed248()*1)+0);
     default:
         return 0.;
   }
}

double Test_Central_1500::neuron0xa7ecce8() {
   return input0;
}

double Test_Central_1500::neuron0xa7ece78() {
   return input1;
}

double Test_Central_1500::neuron0xa7ed050() {
   return input2;
}

double Test_Central_1500::input0xa7ed368() {
   double input = -0.491512;
   input += synapse0xa7f1b18();
   input += synapse0xa7ed4f8();
   input += synapse0xa7ed520();
   return input;
}

double Test_Central_1500::neuron0xa7ed368() {
   double input = input0xa7ed368();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7ed548() {
   double input = 11.262;
   input += synapse0xa7ed720();
   input += synapse0xa7ed748();
   input += synapse0xa7ed770();
   return input;
}

double Test_Central_1500::neuron0xa7ed548() {
   double input = input0xa7ed548();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7ed798() {
   double input = -18.9111;
   input += synapse0xa7ed970();
   input += synapse0xa7ed998();
   input += synapse0xa7ed9c0();
   return input;
}

double Test_Central_1500::neuron0xa7ed798() {
   double input = input0xa7ed798();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7ed9e8() {
   double input = -9.84302;
   input += synapse0xa7edbe0();
   input += synapse0xa7edc08();
   input += synapse0xa7edc30();
   return input;
}

double Test_Central_1500::neuron0xa7ed9e8() {
   double input = input0xa7ed9e8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7edc58() {
   double input = -2.9443;
   input += synapse0xa7ede50();
   input += synapse0xa7ede78();
   input += synapse0xa7edea0();
   return input;
}

double Test_Central_1500::neuron0xa7edc58() {
   double input = input0xa7edc58();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7edec8() {
   double input = 12.1114;
   input += synapse0xa7ee0c0();
   input += synapse0xa7ee0e8();
   input += synapse0xa7ee198();
   return input;
}

double Test_Central_1500::neuron0xa7edec8() {
   double input = input0xa7edec8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7ee1c0() {
   double input = 0.103132;
   input += synapse0xa7ee370();
   input += synapse0xa7ee398();
   input += synapse0xa7ee3c0();
   return input;
}

double Test_Central_1500::neuron0xa7ee1c0() {
   double input = input0xa7ee1c0();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7ee3e8() {
   double input = -0.145439;
   input += synapse0xa7ee5e0();
   input += synapse0xa7ee608();
   input += synapse0xa7ee630();
   return input;
}

double Test_Central_1500::neuron0xa7ee3e8() {
   double input = input0xa7ee3e8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7ee658() {
   double input = -11.7324;
   input += synapse0xa7ee850();
   input += synapse0xa7ee878();
   input += synapse0xa7ee8a0();
   return input;
}

double Test_Central_1500::neuron0xa7ee658() {
   double input = input0xa7ee658();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7ee8c8() {
   double input = 1.27801;
   input += synapse0xa7eeac0();
   input += synapse0xa7eeae8();
   input += synapse0xa7eeb10();
   return input;
}

double Test_Central_1500::neuron0xa7ee8c8() {
   double input = input0xa7ee8c8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7eeb38() {
   double input = 17.2662;
   input += synapse0xa7eed38();
   input += synapse0xa7eed60();
   input += synapse0xa7eed88();
   return input;
}

double Test_Central_1500::neuron0xa7eeb38() {
   double input = input0xa7eeb38();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7eeeb8() {
   double input = 15.4448;
   input += synapse0xa7ef0b8();
   input += synapse0xa7ef0e0();
   input += synapse0xa7ef108();
   return input;
}

double Test_Central_1500::neuron0xa7eeeb8() {
   double input = input0xa7eeeb8();
   return ((1/(1+exp(-input))) * 1)+0;
}

double Test_Central_1500::input0xa7ed248() {
   double input = -1.09658;
   input += synapse0xa7ef1c0();
   input += synapse0xa7ef1e8();
   input += synapse0xa7ef210();
   input += synapse0xa7ef238();
   input += synapse0xa7ef260();
   input += synapse0xa7ef288();
   input += synapse0xa7ef2b0();
   input += synapse0xa7ef2d8();
   input += synapse0xa7ef300();
   input += synapse0xa7ef328();
   input += synapse0xa7ef350();
   input += synapse0xa7ef378();
   return input;
}

double Test_Central_1500::neuron0xa7ed248() {
   double input = input0xa7ed248();
   return (input * 1)+0;
}

double Test_Central_1500::synapse0xa7f1b18() {
   return (neuron0xa7ecce8()*1.27225);
}

double Test_Central_1500::synapse0xa7ed4f8() {
   return (neuron0xa7ece78()*-0.318157);
}

double Test_Central_1500::synapse0xa7ed520() {
   return (neuron0xa7ed050()*-4.69136);
}

double Test_Central_1500::synapse0xa7ed720() {
   return (neuron0xa7ecce8()*-0.218843);
}

double Test_Central_1500::synapse0xa7ed748() {
   return (neuron0xa7ece78()*-0.991067);
}

double Test_Central_1500::synapse0xa7ed770() {
   return (neuron0xa7ed050()*-13.33);
}

double Test_Central_1500::synapse0xa7ed970() {
   return (neuron0xa7ecce8()*0.563781);
}

double Test_Central_1500::synapse0xa7ed998() {
   return (neuron0xa7ece78()*0.373979);
}

double Test_Central_1500::synapse0xa7ed9c0() {
   return (neuron0xa7ed050()*21.4407);
}

double Test_Central_1500::synapse0xa7edbe0() {
   return (neuron0xa7ecce8()*-0.612953);
}

double Test_Central_1500::synapse0xa7edc08() {
   return (neuron0xa7ece78()*-0.492245);
}

double Test_Central_1500::synapse0xa7edc30() {
   return (neuron0xa7ed050()*11.5594);
}

double Test_Central_1500::synapse0xa7ede50() {
   return (neuron0xa7ecce8()*-0.541166);
}

double Test_Central_1500::synapse0xa7ede78() {
   return (neuron0xa7ece78()*2.38148);
}

double Test_Central_1500::synapse0xa7edea0() {
   return (neuron0xa7ed050()*-3.85006);
}

double Test_Central_1500::synapse0xa7ee0c0() {
   return (neuron0xa7ecce8()*-0.359131);
}

double Test_Central_1500::synapse0xa7ee0e8() {
   return (neuron0xa7ece78()*0.504872);
}

double Test_Central_1500::synapse0xa7ee198() {
   return (neuron0xa7ed050()*-13.4148);
}

double Test_Central_1500::synapse0xa7ee370() {
   return (neuron0xa7ecce8()*0.966356);
}

double Test_Central_1500::synapse0xa7ee398() {
   return (neuron0xa7ece78()*0.606441);
}

double Test_Central_1500::synapse0xa7ee3c0() {
   return (neuron0xa7ed050()*2.92563);
}

double Test_Central_1500::synapse0xa7ee5e0() {
   return (neuron0xa7ecce8()*1.03962);
}

double Test_Central_1500::synapse0xa7ee608() {
   return (neuron0xa7ece78()*-0.0568441);
}

double Test_Central_1500::synapse0xa7ee630() {
   return (neuron0xa7ed050()*1.46231);
}

double Test_Central_1500::synapse0xa7ee850() {
   return (neuron0xa7ecce8()*-0.492515);
}

double Test_Central_1500::synapse0xa7ee878() {
   return (neuron0xa7ece78()*-0.425518);
}

double Test_Central_1500::synapse0xa7ee8a0() {
   return (neuron0xa7ed050()*14.739);
}

double Test_Central_1500::synapse0xa7eeac0() {
   return (neuron0xa7ecce8()*-1.02258);
}

double Test_Central_1500::synapse0xa7eeae8() {
   return (neuron0xa7ece78()*0.717172);
}

double Test_Central_1500::synapse0xa7eeb10() {
   return (neuron0xa7ed050()*-4.48672);
}

double Test_Central_1500::synapse0xa7eed38() {
   return (neuron0xa7ecce8()*-0.0556568);
}

double Test_Central_1500::synapse0xa7eed60() {
   return (neuron0xa7ece78()*-0.142405);
}

double Test_Central_1500::synapse0xa7eed88() {
   return (neuron0xa7ed050()*-20.6336);
}

double Test_Central_1500::synapse0xa7ef0b8() {
   return (neuron0xa7ecce8()*0.590757);
}

double Test_Central_1500::synapse0xa7ef0e0() {
   return (neuron0xa7ece78()*-0.539049);
}

double Test_Central_1500::synapse0xa7ef108() {
   return (neuron0xa7ed050()*-17.3757);
}

double Test_Central_1500::synapse0xa7ef1c0() {
   return (neuron0xa7ed368()*-3.52295);
}

double Test_Central_1500::synapse0xa7ef1e8() {
   return (neuron0xa7ed548()*3.53713);
}

double Test_Central_1500::synapse0xa7ef210() {
   return (neuron0xa7ed798()*-7.23512);
}

double Test_Central_1500::synapse0xa7ef238() {
   return (neuron0xa7ed9e8()*-13.0572);
}

double Test_Central_1500::synapse0xa7ef260() {
   return (neuron0xa7edc58()*-1.39761);
}

double Test_Central_1500::synapse0xa7ef288() {
   return (neuron0xa7edec8()*9.81101);
}

double Test_Central_1500::synapse0xa7ef2b0() {
   return (neuron0xa7ee1c0()*4.08124);
}

double Test_Central_1500::synapse0xa7ef2d8() {
   return (neuron0xa7ee3e8()*-4.0037);
}

double Test_Central_1500::synapse0xa7ef300() {
   return (neuron0xa7ee658()*10.8453);
}

double Test_Central_1500::synapse0xa7ef328() {
   return (neuron0xa7ee8c8()*-2.65147);
}

double Test_Central_1500::synapse0xa7ef350() {
   return (neuron0xa7eeb38()*-16.3778);
}

double Test_Central_1500::synapse0xa7ef378() {
   return (neuron0xa7eeeb8()*8.23048);
}

