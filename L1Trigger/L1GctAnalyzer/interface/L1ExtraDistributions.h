


class L1ExtraDistributions {

 public:

  // name is base for histograms
  L1ExtraDistributions(string name);

  ~L1ExtraDistributions();

  // fill the histos
  void fill(L1ExtraParticle* l1);

 private:

  TH1F etDist_;
  TH1F etaDist_;
  TH1F phiDist_;

}
