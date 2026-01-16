#ifndef treeutil_H
#define treeutil_H

#include "TBranch.h"
#include "TTree.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <variant>

class TreeUtil {
private:
  struct BranchDataBase {
    TBranch* branch;
    bool isLoaded = false;
    explicit BranchDataBase(TBranch* b) : branch(b) {}
    virtual ~BranchDataBase() = default;
    virtual const void* getRaw() const = 0;
    virtual void adoptIfNeeded() = 0;
  };

  template <typename T>
  struct BranchDataHolder : BranchDataBase {
    std::unique_ptr<T> buffer;
    mutable T* raw = nullptr;
    mutable std::unique_ptr<T> owner;

    explicit BranchDataHolder(TBranch* b) : BranchDataBase(b) {
      if constexpr (std::is_fundamental_v<T>) {
        buffer = std::make_unique<T>();
        branch->SetAddress(buffer.get());
      } else {
        branch->SetAddress(&raw);
        branch->SetAutoDelete(false);
      }
    }

    const void* getRaw() const override {
      if constexpr (std::is_fundamental_v<T>) {
        return buffer.get();
      } else {
        return owner ? static_cast<const void*>(owner.get()) : static_cast<const void*>(raw);
      }
    }

    void adoptIfNeeded() override {
      if constexpr (!std::is_fundamental_v<T>) {
        if (!owner && raw) {
          owner.reset(raw);
        }
      }
    }
  };

  TTree* tree;
  unsigned int index;
  std::unordered_map<std::string, std::unique_ptr<BranchDataBase>> data;

public:
  void Init(TTree* tree);
  void GetEntry(unsigned int idx);
  static void progress(int nEventsTotal, int nEventsChain);
  void loadAllBranches();
  bool contains(std::string const& name) const;
  template <typename T>
  T const& get(std::string const& name);
  short const& getS(std::string const& name);
  unsigned short const& getUS(std::string const& name);
  int const& getI(std::string const& name);
  unsigned int const& getU(std::string const& name);
  float const& getF(std::string const& name);
  std::vector<short> const& getVS(std::string const& name);
  std::vector<unsigned short> const& getVUS(std::string const& name);
  std::vector<int> const& getVI(std::string const& name);
  std::vector<unsigned int> const& getVU(std::string const& name);
  std::vector<float> const& getVF(std::string const& name);
  std::vector<std::vector<short>> const& getVVS(std::string const& name);
  std::vector<std::vector<unsigned short>> const& getVVUS(std::string const& name);
  std::vector<std::vector<int>> const& getVVI(std::string const& name);
  std::vector<std::vector<unsigned int>> const& getVVU(std::string const& name);
  std::vector<std::vector<float>> const& getVVF(std::string const& name);
};

#endif
