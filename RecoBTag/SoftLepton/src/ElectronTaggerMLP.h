#ifndef ElectronTaggerMLP_h
#define ElectronTaggerMLP_h

class ElectronTaggerMLP { 
public:
   ElectronTaggerMLP() {}
   ~ElectronTaggerMLP() {}
   double Value(int index,double in0,double in1,double in2,double in3);
   double Value(int index, double* input);
private:
   double input0;
   double input1;
   double input2;
   double input3;
   double neuron0x2220ebb0();
   double neuron0x2220eef0();
   double neuron0x2220f230();
   double neuron0x2220f570();
   double input0x2220f9e0();
   double neuron0x2220f9e0();
   double input0x2220fd10();
   double neuron0x2220fd10();
   double input0x22210150();
   double neuron0x22210150();
   double input0x22210590();
   double neuron0x22210590();
   double input0x222109d0();
   double neuron0x222109d0();
   double synapse0x221e7e20();
   double synapse0x221e7ce0();
   double synapse0x2220fc90();
   double synapse0x2220fcd0();
   double synapse0x22210050();
   double synapse0x22210090();
   double synapse0x222100d0();
   double synapse0x22210110();
   double synapse0x22210490();
   double synapse0x222104d0();
   double synapse0x22210510();
   double synapse0x22210550();
   double synapse0x222108d0();
   double synapse0x22210910();
   double synapse0x22210950();
   double synapse0x22210990();
   double synapse0x22210d10();
   double synapse0x221afe70();
   double synapse0x221a2dd0();
   double synapse0x221a2e10();
};

#endif // ElectronTaggerMLP_h

