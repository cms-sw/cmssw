#ifndef MuonTaggerMLP_h
#define MuonTaggerMLP_h

class MuonTaggerMLP { 
public:
   MuonTaggerMLP() {}
   ~MuonTaggerMLP() {}
   double value(int index,double in0,double in1,double in2,double in3);
   double value(int index, double* input);
private:
   double input0;
   double input1;
   double input2;
   double input3;
   double neuron0x16f30a90();
   double neuron0x16f30dd0();
   double neuron0x16f6f2a0();
   double neuron0x16f6f5e0();
   double input0x1711c4b0();
   double neuron0x1711c4b0();
   double input0x1711c760();
   double neuron0x1711c760();
   double input0x16f7a440();
   double neuron0x16f7a440();
   double input0x16f7a880();
   double neuron0x16f7a880();
   double synapse0x16f06360();
   double synapse0x16f09170();
   double synapse0x1711d270();
   double synapse0x171084a0();
   double synapse0x16f7a340();
   double synapse0x16f7a380();
   double synapse0x16f7a3c0();
   double synapse0x16f7a400();
   double synapse0x16f7a780();
   double synapse0x16f7a7c0();
   double synapse0x16f7a800();
   double synapse0x16f7a840();
   double synapse0x16f7abc0();
   double synapse0x16f7ac00();
   double synapse0x16f7ac40();
};

#endif // MuonTaggerMLP_h

