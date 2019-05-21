
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <TBranch.h>

#include <string>
#include <vector>

class ProductInfo {
public:
  ProductInfo(const edm::Provenance &prov, TBranch &branch, edm::EDGetToken const &token);

  Long64_t size() const { return m_size; }
  const edm::InputTag &tag() const { return m_tag; }
  const edm::TypeID &type() const { return m_type; }
  const edm::EDGetToken &token() const { return m_token; }
  static bool sort(const ProductInfo &x, const ProductInfo &y);

private:
  static void addBranchSizes(TBranch &branch, Long64_t &size);

  edm::InputTag m_tag;
  edm::TypeID m_type;
  edm::EDGetToken m_token;
  Long64_t m_size;
};

ProductInfo::ProductInfo(const edm::Provenance &prov, TBranch &branch, edm::EDGetToken const &token)
    : m_tag(prov.moduleLabel(), prov.productInstanceName(), prov.processName()),
      m_type(prov.branchDescription().unwrappedTypeID()),
      m_token(token),
      m_size(0) {
  addBranchSizes(branch, m_size);
}

void ProductInfo::addBranchSizes(TBranch &branch, Long64_t &size) {
  size += branch.GetTotalSize();  // Includes size of branch metadata
  // Now recurse through any subbranches.
  Long64_t nB = branch.GetListOfBranches()->GetEntries();
  for (Long64_t i = 0; i < nB; ++i) {
    TBranch *btemp = (TBranch *)branch.GetListOfBranches()->At(i);
    if (btemp != nullptr) {
      addBranchSizes(*btemp, size);
    }
  }
}

bool ProductInfo::sort(ProductInfo const &x, ProductInfo const &y) { return x.size() < y.size(); }

typedef std::vector<ProductInfo> ProductInfos;
