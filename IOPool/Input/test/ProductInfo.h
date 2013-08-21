
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Common/interface/Provenance.h"

#include <TBranch.h>

#include <string>
#include <vector>

class ProductInfo {
   public:
      ProductInfo(const edm::Provenance &prov, TBranch & branch);

      Long64_t size() const {return m_size;}
      const edm::InputTag & tag() const { return m_tag;}
      const std::string & className() const { return m_className;}
      static bool sort(const ProductInfo &x, const ProductInfo &y);

   private:
      static void addBranchSizes(TBranch & branch, Long64_t &size);

      std::string m_className;
      edm::InputTag m_tag;
      Long64_t m_size;
};

ProductInfo::ProductInfo(const edm::Provenance &prov, TBranch & branch) :
    m_className(prov.className()),
    m_tag(prov.moduleLabel(), prov.productInstanceName(), prov.processName()),
    m_size(0)
{
   addBranchSizes(branch, m_size);
}

void ProductInfo::addBranchSizes(TBranch & branch, Long64_t &size)
{
   size += branch.GetTotalSize(); // Includes size of branch metadata
   // Now recurse through any subbranches.
   Long64_t nB = branch.GetListOfBranches()->GetEntries();
   for (Long64_t i = 0; i < nB; ++i)
   {
      TBranch *btemp = (TBranch *)branch.GetListOfBranches()->At(i);
      if (btemp != NULL)
      {
         addBranchSizes(*btemp, size);
      }
   }
}

bool ProductInfo::sort(ProductInfo const &x, ProductInfo const &y)
{
   return x.size() < y.size();
}

typedef std::vector<ProductInfo> ProductInfos;

