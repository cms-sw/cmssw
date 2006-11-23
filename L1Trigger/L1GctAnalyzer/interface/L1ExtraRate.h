

class L1ExtraRate {

 public:

  L1ExtraRate(string name);
  ~L1ExtraRate();

  void fill (L1ExtraParticle& l1);
  void weightedFill(L1ExtraParticle& l1, double weight);

 private:

  TH1F rate_;

}
