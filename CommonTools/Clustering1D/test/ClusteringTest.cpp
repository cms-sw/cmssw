#include "CommonTools/Clustering1D/test/Input.cpp"
#include "CommonTools/Clustering1D/interface/Cluster1DMerger.h"
#include "CommonTools/Clustering1D/interface/TrivialWeightEstimator.h"

#define HaveMtv
#define HaveFsmw
#define HaveDivisive
#ifdef HaveMtv
#include "CommonTools/Clustering1D/interface/MtvClusterizer1D.h"
#endif
#ifdef HaveFsmw
#include "CommonTools/Clustering1D/interface/FsmwClusterizer1D.h"
#endif
#ifdef HaveDivisive
#include "CommonTools/Clustering1D/interface/DivisiveClusterizer1D.h"
#endif

#include <string>
#include <iostream>

using namespace std;

namespace {
  void print(const Cluster1D<string>& obj) {
    cout << "   Cluster1D ";
    vector<const string*> names = obj.tracks();
    for (vector<const string*>::iterator nm = names.begin(); nm != names.end(); ++nm) {
      cout << **nm;
    };
    cout << " at " << obj.position().value() << " +/- " << obj.position().error() << " weight " << obj.weight();
    cout << endl;
  }

  void mergingResult(const Cluster1D<string>& one, const Cluster1D<string>& two) {
    cout << "Merger test:" << endl;
    print(one);
    print(two);

    Cluster1D<string> result = Cluster1DMerger<string>(TrivialWeightEstimator<string>())(one, two);
    cout << "Merge result: " << endl;
    print(result);
  }

  void mergerTest() {
    string one_s = "a";
    vector<const string*> one_names;
    one_names.push_back(&one_s);
    Cluster1D<string> one(Measurement1D(1.0, 0.1), one_names, 1.0);

    vector<const string*> two_names;
    string two_s = "b";
    two_names.push_back(&two_s);
    Cluster1D<string> two(Measurement1D(2.0, 0.2), two_names, 1.0);

    mergingResult(one, two);
  }
}  // namespace

int main(int argc, char** argv) { mergerTest(); }
