#ifndef CondFormats_HcalObjects_HcalObjectAddons
#define CondFormats_HcalObjects_HcalObjectAddons

#include <vector>
#include <algorithm>

// functions with common forms
namespace HcalObjectAddons {
  template <class Item, class Less>
  const Item * findByT (const Item* target, const std::vector<const Item*>& itemsByT){
    Less less;
    auto item = std::lower_bound (itemsByT.begin(), itemsByT.end(), target, less);
    if (item == itemsByT.end() || !less.equal(*item,target)){
      return 0;
    }
    return *item;
  }

  //sorting
  template <class Item, class Less>
  static void sortByT(const std::vector<Item>& items, std::vector<const Item*>& itemsByT){
    itemsByT.clear();
    itemsByT.reserve(items.size());
    Less less;
    for(const auto& i : items){
      if (less.good(i)) itemsByT.push_back(&i);
    }
    std::sort (itemsByT.begin(), itemsByT.end(), less);
  }

}

#endif
