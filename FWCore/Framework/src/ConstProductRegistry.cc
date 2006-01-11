/**
   \file
   class impl

   \version $Id: ProductRegistry.cc,v 1.13 2005/10/11 19:28:17 chrjones Exp $
   \date 19 Jul 2005
*/

#include <FWCore/Framework/interface/ConstProductRegistry.h>

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.13 2005/10/11 19:28:17 chrjones Exp $";


namespace edm
{

  std::vector<std::string>
  ConstProductRegistry::allBranchNames() const
  {
    std::vector<std::string> result;
    result.reserve( productList().size() ); 

    ProductList::const_iterator it  = productList().begin();
    ProductList::const_iterator end = productList().end();

    for ( ; it != end; ++it ) result.push_back(it->second.branchName_);

  return result;
  }

  std::vector<BranchDescription const*> 
  ConstProductRegistry::allBranchDescriptions() const
  {
    std::vector<BranchDescription const*> result;
    result.reserve( productList().size() );

    ProductList::const_iterator it  = productList().begin();
    ProductList::const_iterator end = productList().end();
    
    for ( ; it != end; ++it) result.push_back(&(it->second));    
    return result;
  }
  
} // namespace edm


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
