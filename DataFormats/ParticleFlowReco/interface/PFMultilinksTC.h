#ifndef __PFMultilinksTC__
#define __PFMultilinksTC__

// Done by Glowinski & Gouzevitch                                                                                                                                                   

#include <vector>

namespace reco {

  /// \brief Abstract This class is used by the KDTree Track / Ecal Cluster                                                                                                         
  /// linker to store all found links.                                                                                                                                              
  ///                                                                                                                                                                               
  typedef std::vector<std::pair<double, double> > PFMultilinksType;
  class PFMultiLinksTC
  {
  public:
    bool                                        isValid;
    PFMultilinksType                            linkedClusters;

  public:
    PFMultiLinksTC(bool isvalid = false) : isValid(isvalid)
      {}
  };
}


#endif
