#ifndef __JetAlgorithmEF__
#define __JetAlgorithmEF__

#include <TLorentzVector.h>
#include <map>
#include <vector>



using namespace std;

class JetAlgorithmEF {

 public:

  class Jet {
    
  private:
    const vector<TLorentzVector>*              fAllVecs;
    
    TLorentzVector  fMomentum;
    vector<int>    fVecIndexes;

  public:
    Jet() : fAllVecs(0) {}
    Jet( int i, const vector<TLorentzVector>* allvecs) : fAllVecs(allvecs) {
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
    const vector<int>&    GetIndexes() const {return fVecIndexes;}

    friend ostream& operator<<(ostream& out, const JetAlgorithmEF::Jet& jet);
  };


 private:

  const vector<TLorentzVector>*              fAllVecs;
  vector< JetAlgorithmEF::Jet >              fJets;
  vector< int >                              fAssignedVecs; 
  map<double,  int, greater<double> >        fEtOrderedSeeds;
  

  double                                    fConeAngle;
  double                                    fSeedEt;
  double                                    fConeMerge;


 public:


  typedef map< double, JetAlgorithmEF::Jet, greater<double> >::iterator IJ;
  typedef  map<double, int, greater<double> >::const_iterator IV;

  JetAlgorithmEF() : fConeAngle(0.4), fSeedEt(5),  fConeMerge(3) {}
  JetAlgorithmEF(double cone, double et, double conemerge) : 
    fConeAngle(cone), 
    fSeedEt(et),
    fConeMerge(conemerge)
    {}
  virtual ~JetAlgorithmEF() {}

  const vector< JetAlgorithmEF::Jet >& FindJets( const vector<TLorentzVector>* vecs);

  void SetCone(double cone) {fConeAngle = cone;}
  void SetSeedEt(double et) {fSeedEt = et;}

  static double DeltaR(double eta1, double phi1, double eta2, double phi2);

  void Update();
  
  void Clear() { 
    fJets.clear(); 
    fAssignedVecs.clear(); 
    fEtOrderedSeeds.clear();
  } 

  void CleanUp();
  void MergeJets(map< double, JetAlgorithmEF::Jet, greater<double> >& etjets);

  double GetConeAngle() const { return fConeAngle;}
  const vector< JetAlgorithmEF::Jet >& GetJets() const { return fJets;}
};

#endif
