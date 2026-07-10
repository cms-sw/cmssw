#include <vector>
#include <list>
#include <cstdint>

template <typename T, typename Greater>
class AccumulatingSort {
private:
  std::vector<std::list<T> > mSortArrays;
  uint32_t mSize;

private:
  void AccumulatorUnit(const std::list<T>& aInput, T& aAcc, std::list<T>& aTail) {
    aTail.clear();

    bool lAccInserted(false);
    const Greater g;
    for (auto const& l : aInput) {
      if (!lAccInserted and
          !g(l,
             aAcc))  // Accumulator greater than or equal to new entry and not previously inserted -> Reinsert accumulator
      {
        aTail.push_back(aAcc);
        lAccInserted = true;
      }
      aTail.push_back(l);
    }

    aAcc = *aTail.begin();
    aTail.erase(aTail.begin());
  }

public:
  AccumulatingSort(const uint32_t& aSize) : mSortArrays(aSize + 1, std::list<T>(aSize)), mSize(aSize) {}

  void Merge(const std::vector<T>& aInput, std::vector<T>& aOutput) {
    aOutput.resize(mSize);

    mSortArrays[0].clear();
    mSortArrays[0].insert(mSortArrays[0].begin(), aInput.begin(), aInput.end());

    for (uint32_t i(0); i != mSize; ++i)
      AccumulatorUnit(mSortArrays[i], aOutput[i], mSortArrays[i + 1]);
  }
};
