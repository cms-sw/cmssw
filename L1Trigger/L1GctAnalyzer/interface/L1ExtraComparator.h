




class L1ExtraComparator {

 public:

  // name is base for histograms
  L1ExtraComparator(string name);

  ~L1ExtraComparator();

  // fill the histos
  void fill(L1ExtraParticle* l1, Particle* mc);

 private:

  // Et resolution
  TH1F etDiff_;  // fill with Et(L1)-Et(MC)
  TH1F scaledEtDiff_;  // fill with Et(L1)-Et(MC) / Et(MC)

  // position resolution
  TH1F etaDiff_; // eta(L1)-eta(MC)
  TH1F phiDiff_; // phi(L1)-phi(MC)
  TH1F rDiff_; // difference in R between L1 and MC

}
