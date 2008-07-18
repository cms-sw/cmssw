#ifndef __PFJetAlgorithm__
#define __PFJetAlgorithm__

#include <TLorentzVector.h>
#include <map>
#include <vector>




class PFJetAlgorithm {

 public:

  class Jet {
    
  private:
    const std::vector<TLorentzVector>*              fAllVecs;
    
    TLorentzVector  fMomentum;
    std::vector<int>    fVecIndexes;

  public:
    Jet() : fAllVecs(0) {}
    Jet( int i, const std::vector<TLorentzVector>* allvecs) : fAllVecs(allvecs) {
      Add(i);
    }
    ~Jet() {}
    
    void Add(int i) {
      fVecIndexes.push_back(i);      
      fMomentum += (*fAllVecs)[i];
    }

    void Clear() { fVecIndexes.clear(); fMomentum *= 0;} 

    Jet& operator+=(const Jet& other) {
      fVecIndexes.insert( fVecIndexes.begin(), other.fVecIndexes.begin(), other.fVecIndexes.end());
      fMomentum += other.fMomentum;
      return *this;
    }

    const TLorentzVector& GetMomentum() const {return fMomentum;}
    const std::vector<int>&    GetIndexes() const {return fVecIndexes;}

    friend ostream& operator<<(ostream& out, const PFJetAlgorithm::Jet& jet);
  };


 private:

  const std::vector<TLorentzVector>*              fAllVecs;
  std::vector< PFJetAlgorithm::Jet >              fJets;
  std::vector< int >                              fAssignedVecs; 
  std::map<double,  int, std::greater<double> >   fEtOrderedSeeds;
  

  double                                    fConeAngle;
  double                                    fSeedEt;
  double                                    fConeMerge;


 public:


  typedef std::map< double, PFJetAlgorithm::Jet, std::greater<double> >::iterator IJ;
  typedef  std::map<double, int, std::greater<double> >::const_iterator IV;

  PFJetAlgorithm() : fConeAngle(0.5), fSeedEt(2),  fConeMerge(0.5) {}

  PFJetAlgorithm(double cone, double et, double conemerge) : 
    fConeAngle(cone), 
    fSeedEt(et),
    fConeMerge(conemerge)
    {}

  virtual ~PFJetAlgorithm() {}

  const std::vector< PFJetAlgorithm::Jet >& 
    FindJets( const std::vector<TLorentzVector>* vecs);


  void SetConeAngle(double coneAngle) {fConeAngle = coneAngle;}
  void SetSeedEt(double et) {fSeedEt = et;}
  void SetConeMerge(double coneMerge) {fConeMerge = coneMerge;}


  static double DeltaR(double eta1, double phi1, double eta2, double phi2);

  void Update();
  
  void Clear() { 
    fJets.clear(); 
    fAssignedVecs.clear(); 
    fEtOrderedSeeds.clear();
  } 

  void CleanUp();
  void 
    MergeJets(std::map< double, PFJetAlgorithm::Jet, 
              std::greater<double> >& etjets);

  double GetConeAngle() const { return fConeAngle;}
  const std::vector< PFJetAlgorithm::Jet >& GetJets() const { return fJets;}
};

#endif
