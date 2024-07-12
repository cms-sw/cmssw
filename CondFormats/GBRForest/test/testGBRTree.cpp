#include "CondFormats/GBRForest/interface/GBRTree.h"

namespace {

  struct MyTree : public GBRTree {
    MyTree() {
      fCutIndices.resize(1, 0);
      fCutVals.resize(1, 0);
      fLeftIndices.resize(1, 0);
      fRightIndices.resize(1, -1);
      fResponses.resize(2);
      fResponses[0] = 1;
      fResponses[1] = 2;
    }

    double GetResponseNew(const float *vector) const {
      int index = 0;
      do {
        auto r = fRightIndices[index];
        auto l = fLeftIndices[index];
        // the code below is equivalent to the original
        // index = (vector[fCutIndices[index]] > fCutVals[index]) ? r : l;
        // gnenerates non branching code  and it's at least 30% faster
        // see https://godbolt.org/z/xT5dY9Th1   (yes in gcc13 is changed... but in the trunk is back as it was in gcc12)
        unsigned int x = vector[fCutIndices[index]] > fCutVals[index] ? ~0 : 0;
        index = (x & r) | ((~x) & l);
      } while (index > 0);
      return fResponses[-index];
    }

    double GetResponseOld(const float *vector) const {
      int index = 0;
      do {
        auto r = fRightIndices[index];
        auto l = fLeftIndices[index];
        index = (vector[fCutIndices[index]] > fCutVals[index]) ? r : l;
      } while (index > 0);
      return fResponses[-index];
    }
  };

}  // namespace

#include <iostream>

int main() {
  MyTree aTree;

  float val[1];
  double ret;

  val[0] = -1;
  std::cout << "val " << val[0] << std::endl;
  ret = aTree.GetResponse(val);
  std::cout << "def " << ret << std::endl;
  ret = aTree.GetResponseOld(val);
  std::cout << "old " << ret << std::endl;
  ret = aTree.GetResponseOld(val);
  std::cout << "new " << ret << std::endl;

  val[0] = 1;
  std::cout << "val " << val[0] << std::endl;
  ret = aTree.GetResponse(val);
  std::cout << "def " << ret << std::endl;
  ret = aTree.GetResponseOld(val);
  std::cout << "old " << ret << std::endl;
  ret = aTree.GetResponseOld(val);
  std::cout << "new " << ret << std::endl;

  return 0;
}
