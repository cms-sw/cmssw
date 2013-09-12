#ifndef MuonTaggerMLP_h
#define MuonTaggerMLP_h

class MuonTaggerMLP { 
public:
   MuonTaggerMLP() {}
   ~MuonTaggerMLP() {}
   double Value(int index,double in0,double in1,double in2,double in3);
   double Value(int index, double* input);
private:
   double input0;
   double input1;
   double input2;
   double input3;
   double neuron0x67da870();
   double neuron0x67dabb0();
   double neuron0x67daef0();
   double neuron0x67db230();
   double input0x67db6a0();
   double neuron0x67db6a0();
   double input0x67db9d0();
   double neuron0x67db9d0();
   double input0x67dbe10();
   double neuron0x67dbe10();
   double input0x67dc250();
   double neuron0x67dc250();
   double input0x67dc690();
   double neuron0x67dc690();
   double synapse0x67b3ae0();
   double synapse0x67b39a0();
   double synapse0x67db950();
   double synapse0x67db990();
   double synapse0x67dbd10();
   double synapse0x67dbd50();
   double synapse0x67dbd90();
   double synapse0x67dbdd0();
   double synapse0x67dc150();
   double synapse0x67dc190();
   double synapse0x67dc1d0();
   double synapse0x67dc210();
   double synapse0x67dc590();
   double synapse0x67dc5d0();
   double synapse0x67dc610();
   double synapse0x67dc650();
   double synapse0x67dc9d0();
   double synapse0x677bb20();
   double synapse0x6745bb0();
   double synapse0x6745bf0();
};

#endif // MuonTaggerMLP_h

