#include <vector>
#include <string>

#include "Rtypes.h"

#include "eve_filter.h"

// show the calorimeter system (ECAL, HCAL)
void calo_filter(void) {
  std::vector< std::pair< std::string, Color_t> > elements;
  elements.push_back( std::make_pair(std::string("/cms:World/cms:CMSE/caloBase:CALO/eregalgo:ECAL/eregalgo:EREG/eealgo:ENCA/eealgo:E[EO][0-9][0-9]"),                   kCyan) );     // .../eealgo:EFRY (except for E[EO]02 which are elementary (?))
  elements.push_back( std::make_pair(std::string("/cms:World/cms:CMSE/caloBase:CALO/eregalgo:ECAL/eregalgo:EBAR/ebalgo:ESPM/eregalgo:EFAW/eregalgo:EHAWR/ebalgo:EWAL"), kCyan) );     // .../ebalgo:EWRA/ebalgo:ECLR/ebalgo:EBRY
  elements.push_back( std::make_pair(std::string("/cms:World/cms:CMSE/caloBase:CALO/hcalalgo:HCal/hcalbarrelalgo:HB"), kRed) );
  elements.push_back( std::make_pair(std::string("/cms:World/cms:CMSE/caloBase:CALO/hcalalgo:HCal/hcalendcapalgo:HE"), kRed) );

  TEveElement * node = get_root_object("cms:World_1");
  if (node) {
    init_filter(elements);
    apply_filter( node, do_hide, true );
  }
}
